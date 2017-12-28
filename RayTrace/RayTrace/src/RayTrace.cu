#include "RayTrace.h"
#include "Functions.h"
#include "CudaFuncs.h"
#include "RandomGenerator.h"
#include <device_functions.h>
#include "device_launch_parameters.h"
#include <surface_functions.h>
#include "surface_indirect_functions.h"
#include <cuda_gl_interop.h>
#include<iostream>
#include <algorithm>
#include "ObjReaderSimple.h"
#include <assert.h>

__constant__ float epsilon = 0.00001f;

const int num_bounces = 3;
const int pass_flt3_size = 5;
#define attenuation 8.f

struct mat3
{
	float3 cols[3];
};


// Param. container:
struct DevArrs
{
	// Device data
	float3 *_verts, *_norms;
	uint3* _Vind, *_Nind;
	float3 *_light;

	unsigned int num_tri, num_light;
};


/* Filthy globals:
*/
//Device
DevArrs dev;
float4 *_target;
float3 *_pass_buffer[num_bounces];

//Host data
CU_image frame_buffer;
SimpleMesh mesh;

__host__ GLuint generateTraceContext(unsigned int buffer_width, unsigned int buffer_height, unsigned int num_light, const char *obj_file)
{

	if (!initCudaDevice()) return 0;

	mf::RandomGenerator rnd;

	std::unique_ptr<float3> light_info(new float3[num_light * 2]);
	
	float3* linfo = light_info.get();
	linfo[0] = { 1,-1,1 };
	linfo[num_light] = make_float3(1.f);
	for (int i = 1; i < num_light; i++)
	{
		linfo[i] = rnd.randomNormal()*5;			// Light point
		linfo[i].z = abs(linfo[i].z)/2;
		linfo[num_light + i] = make_float3(0.7f);	// Intensity color
	}
	dev.num_light = num_light;

	if (!allocateTexture_RGBA(buffer_width, buffer_height, frame_buffer))
		return 0;
		
	if (!readObj(obj_file, mesh) || !mesh.valid()) return 0;
	dev.num_tri = mesh._face_ind.size() / 3;
	dev._verts = (float3*)genCpy(mesh._vertices.data(), mesh._vertices.size());
	dev._norms = (float3*)genCpy(mesh._normals.data(), mesh._normals.size());
	dev._Vind = (uint3*)genCpy(mesh._face_ind.data(), mesh._face_ind.size());
	dev._Nind = (uint3*)genCpy(mesh._face_nor.data(), mesh._face_nor.size());
	dev._light = genCpy(linfo, num_light*2);

	size_t num_pixel = buffer_height*buffer_width;
	_target = cudaAlloc<float4>(num_pixel);
	for(int i = 0; i < num_bounces;i++)
		_pass_buffer[i] = cudaAlloc<float3>(num_pixel * pass_flt3_size);

	if (dev._verts == NULL || dev._norms == NULL || dev._Vind == NULL || dev._Nind == NULL || dev._light == NULL)
		closeContext();
	cudaCheck();
	return frame_buffer._textureID; // 0 if closed
}
__host__ void closeContext()
{
	cudaFree(_target);
	for (int i = 0; i < num_bounces; i++)
		cudaFree(_pass_buffer[i]);
	cudaFree(dev._verts);
	cudaFree(dev._norms);
	cudaFree(dev._Vind);
	cudaFree(dev._Nind);
	cudaFree(dev._light);
	frame_buffer.destroy();
}
__global__ void colorKernel(cudaSurfaceObject_t surfaceWrite, float4 color, uint2 dimension)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (x < dimension.x && y < dimension.y)
	{
		surf2Dwrite(color, surfaceWrite, x * sizeof(float4), y, cudaBoundaryModeTrap);
	}
}
__global__ void copyKernel(cudaSurfaceObject_t surfaceWrite, float4 *source, uint2 dimension)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < dimension.x && y < dimension.y)
	{
		//cudaBoundaryModeZero 
		surf2Dwrite(source[x + y *dimension.x], surfaceWrite, x * sizeof(float4), y, cudaBoundaryModeTrap);
	}
}
/* Ray intersection returning the barycentric coordinates.
*/
__device__ bool rayIntersection(const float3* verts, const float3 &ray_dir, const float3 &ray_pos, uint3 indices, float &distance, float3 &uvw)
{
	/*	Ray triangle intersecting implemented according to the barycentric technique
		and implemented from the book Real-Time Rendering third edition (p.750)
	*/
	float3 p0 = verts[indices.x];
	float3 e1 = verts[indices.y] - p0;
	float3 e2 = verts[indices.z] - p0;
	float3 q = cross(ray_dir, e2);

	float a = dot(e1, q);

	float f = 1 / a;

	//Calculate area u
	float3 S = ray_pos - p0;
	uvw.x = f * dot(S, q);

	//Calculate area v
	float3 R = cross(S, e1);
	uvw.y = f * dot(ray_dir, R);


	//Calculate distance
	distance = f * dot(e2, R);

	return
		//Check if triangle is parallel with ray direction
		(-epsilon > a || a > epsilon) &&
		uvw.x >= 0 && 
		uvw.y >= 0 &&
		//Check for area w. Since u & v areas is signed w intersect only if u + v is less then 1
		(uvw.z = 1 - (uvw.x + uvw.y)) >= 0;
}

__device__ bool simpleTrace(const DevArrs &param, const float3& pos, const float3& dir, float &inter_dist, unsigned int &tri_ind, float3 &tri_uvw)
{
	tri_ind = UINT32_MAX;
	for (unsigned int i = 0;i < param.num_tri; i++)
	{
		float3 uvw;
		float dist;
		if (rayIntersection(param._verts, dir, pos, param._Vind[i], dist, uvw) &&
			dist > epsilon &&	dist < inter_dist)
		{
			inter_dist = dist; tri_ind = i; tri_uvw = uvw;
		}
	}
	return tri_ind != UINT32_MAX;
}

__device__ bool traceSimple(const DevArrs &param, const float3& pos, const float3& dir, float trace_len)
{
	unsigned int tri_ind;
	float3 tri_uvw;
	return simpleTrace(param, pos, dir, trace_len, tri_ind, tri_uvw);
}
__device__ float3 calcNorm(const float3 *norms, const uint3 &ind, const float3 &uvw)
{
	return norms[ind.x] * uvw.z + norms[ind.y] * uvw.x + norms[ind.z] * uvw.y;
}
__device__ float3 shadowRay(const DevArrs &param, int light_index, const float3& pos, const float3 &in_dir, const float3& surf_norm, const float3 &diffuseFactor, float traversed_dist)
{
	float3 dir = param._light[light_index] - pos;
	float len = sqrt(dot(dir, dir));
	dir *= 1 / len;
	if (!traceSimple(param, pos, dir, len))
	{
		float fac = (len + traversed_dist) / (attenuation*attenuation);
		return fac * dot(surf_norm, dir) * diffuseFactor * param._light[param.num_light + light_index];
	}
	return{ 0,0,0 };
}

__device__ float traceRay(const DevArrs &param, float3 &pos, float3& in_dir, float3 &surf_norm, float3 &diffuseFactor)
{
	float traverse_dist = FLT_MAX;
	unsigned int tri_ind;
	float3 uvw;
	if (!simpleTrace(param, pos, in_dir, traverse_dist, tri_ind, uvw))
		return 0;
	// Calc. reflected ray
	pos = in_dir * traverse_dist + pos;
	surf_norm = calcNorm(param._norms, param._Nind[tri_ind], uvw);
	diffuseFactor *= tri_ind < 449 ? make_float3(0.2f) : make_float3(0.7f); //Material (simple implementation)
	return traverse_dist;
}

__device__ void storeRayPass(uint pixel, uint pixel_dim, float3* pass, const float3 &pos, const float3 &dir, const float3 &surf_norm, const float3 &diffuse_factor, float traverse_dist)
{
	pass[pixel] = pos;
	pass[pixel + pixel_dim] = dir;
	pass[pixel + 2 * pixel_dim] = surf_norm;
	pass[pixel + 3 * pixel_dim] = diffuse_factor;
	pass[pixel + 4 * pixel_dim].x = traverse_dist; // Padded mem.
}

__device__ void storeDiffuse(uint pixel, uint pixel_dim, float3* pass, const float3 &diffuse_factor)
{
	pass[pixel + 3 * pixel_dim] = diffuse_factor;
}
__global__ void secondaryKernel(DevArrs params, float3* in_pass, float3* out_pass, uint2 dim, float trace_depth)
{
	int pX = blockIdx.x * blockDim.x + threadIdx.x;
	int pY = blockIdx.y * blockDim.y + threadIdx.y;

	if (pX < dim.x && pY < dim.y)
	{
		int num_pixel = dim.x * dim.y;
		unsigned int pixel = pX + pY * dim.x;
		float3 diffuseFactor = in_pass[pixel + 3 * num_pixel];
		if (diffuseFactor.x + diffuseFactor.y + diffuseFactor.z < epsilon)
			return;
		float3 pos =		in_pass[pixel];
		float3 dir =		in_pass[pixel + num_pixel];
		float3 surf_norm =	in_pass[pixel + 2 * num_pixel];
		float3 traversal =	in_pass[pixel + 4 * num_pixel];

		if (traversal.x >= trace_depth)
		{
			storeDiffuse(pixel, num_pixel, out_pass, { 0,0,0 });
			return;
		}

		// Bounce
		dir = reflect(dir, surf_norm);
		float traverse_dist = traceRay(params, pos, dir, surf_norm, diffuseFactor);
		diffuseFactor = traverse_dist < epsilon ? make_float3(0) : diffuseFactor;

		storeRayPass(pixel, num_pixel, out_pass, pos, dir, surf_norm, diffuseFactor, traversal.x + traverse_dist);
	}
}
__global__ void primaryKernel(DevArrs params, float3* pass0, uint2 dimension, float2 d_edge, float3 cam_pos, mat3 cam_basis)
{
	int pX = blockIdx.x * blockDim.x + threadIdx.x;
	int pY = blockIdx.y * blockDim.y + threadIdx.y;
	//Calc. NDC pixel center coordinate 
	float2 xy = { (2 * pX + 0.5f) / dimension.x - 1, (2 * pY + 0.5f) / dimension.y - 1 };

	if (pX < dimension.x && pY < dimension.y)
	{
		//Calc. ray:
		float2 d_xy = xy * d_edge;
		float3 dir = d_xy.x*cam_basis.cols[0] + d_xy.y *cam_basis.cols[1] + cam_basis.cols[2];
		float3 pos = cam_pos;
		dir = normalize(dir);

		// Bounce
		float3 surf_norm, diffuseFactor = { 1,1,1 };
		float traverse_dist = traceRay(params, pos, dir, surf_norm, diffuseFactor);
		diffuseFactor = traverse_dist < epsilon ? make_float3(0) : diffuseFactor;

		unsigned int pixel = pX + pY * dimension.x;
		int num_pixel = dimension.x * dimension.y;
		storeRayPass(pixel, num_pixel, pass0, pos, dir, surf_norm, diffuseFactor, traverse_dist);
	}
}
__global__ void lightKernel(DevArrs params, int light_index, float4* target, float3* pass, uint2 dimension)
{
	int pX = blockIdx.x * blockDim.x + threadIdx.x;
	int pY = blockIdx.y * blockDim.y + threadIdx.y;

	if (pX < dimension.x && pY < dimension.y)
	{
		int num_pixel = dimension.x * dimension.y;
		unsigned int pixel = pX + pY * dimension.x;
		float3 diffuseFactor = pass[pixel + 3 * num_pixel];
		if (diffuseFactor.x + diffuseFactor.y + diffuseFactor.z < epsilon)
			return;
		float3 pos =			pass[pixel];
		float3 dir =			pass[pixel + num_pixel];
		float3 surf_norm =		pass[pixel + 2 * num_pixel];
		float3 traversal =		pass[pixel + 4 * num_pixel];
		//Calc. color
		float3 col = shadowRay(params, light_index, pos, dir, surf_norm, diffuseFactor, traversal.x);
		target[pixel] += make_float4(col, 1);
	}
}

__host__ void rayTrace(unsigned int buffer_width, unsigned int buffer_height, float fov, float trace_depth, unsigned int kernel_dim)
{
	cudaSurfaceObject_t buffer_surface;
	if (frame_buffer.mapSurface(buffer_surface) != cudaSuccess)
		return;

	float aspect = buffer_width / (float)buffer_height;
	fov = fov / 360.f * mf::PIx2;
	float2 d_edge = {aspect*tan(fov/2), tan(fov / 2)};
	mat3 cam_dir;
	cam_dir.cols[0] = { 1, 0, 0 };
	cam_dir.cols[1] = { 0, 0, 1 };
	cam_dir.cols[2] = { 0, 1, 0 };

	float3 cam_pos = { 0,-7, 0 };

	dim3 block(kernel_dim, kernel_dim, 1);
	dim3 grid(div_ceil(buffer_width, block.x), div_ceil(buffer_height, block.y), 1);
	uint2 dim = { buffer_width, buffer_height };

	// Primary camera rays:
	cudaCheck();
	primaryKernel << <grid, block >> > (dev, _pass_buffer[0], dim, d_edge, cam_pos, cam_dir);

	// Secondary (reflection) rays:
	cudaCheck();
	for (int i = 1; i < num_bounces; i++)
	{
		secondaryKernel<<<grid, block>>>(dev, _pass_buffer[i-1], _pass_buffer[i], dim, trace_depth);
		cudaCheck();
	}		

	// Shadow rays and light evaluation
	for (int i = 0; i < num_bounces; i++)
	{
		for (int light_i = 0; light_i < dev.num_light; light_i++)
		{
			lightKernel << <grid, block >> > (dev, light_i, _target, _pass_buffer[i], dim);
			cudaCheck();
		}
	}
	copyKernel << <grid, block >> > (buffer_surface, _target, dim);

	cudaCheck();
	if (frame_buffer.unmap() != cudaSuccess)
		std::cout << "Error unmapping\n";
	cudaDestroySurfaceObject(buffer_surface);
}
