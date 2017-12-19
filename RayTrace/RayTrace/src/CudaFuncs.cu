#include "CudaFuncs.h"
#include <cuda_gl_interop.h>
#include <device_functions.h>
#include "device_launch_parameters.h"
#include<iostream>
#include <assert.h>

#pragma region Functions

__device__ int div2ceil(int value) { return (value & 1) + (value >> 1); }
__device__ int cuda_div_ceil(int nume, int denom) { return nume / denom + ((nume % denom) > 0); }
/* For positive nums with sum less then INT_MAX*/
__device__ int cuda_div_ceil_pos(int nume, int denom) { return (nume + denom - 1) / denom; }

#pragma endregion



#pragma region Device constructs

/* Initiate runtime device. */
bool initCudaDevice()
{
	cudaError_t err = cudaSetDevice(0);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return false;
	}
	return true;
}
__host__ void cudaCheck()
{
	cudaDeviceSynchronize();
	cudaError err = cudaPeekAtLastError();
	if (err != cudaSuccess)
	{
		std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
		cudaDeviceReset();
		exit(0);
	}
}

bool read(int* dev_arr, int* arr, size_t arr_len)
{
	// Copy output vector from GPU buffer to host memory.
	cudaError_t cudaStatus = cudaMemcpy(arr, dev_arr, arr_len * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return false;
	}
	return true;
}
bool read(float* dev_arr, float* arr, size_t arr_len)
{
	// Copy output vector from GPU buffer to host memory.
	cudaError_t cudaStatus = cudaMemcpy(arr, dev_arr, arr_len * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return false;
	}
	return true;
}


#pragma endregion

#ifdef OPEN_GL

bool allocateTexture_RGBA(unsigned int width, unsigned int height, CU_image &image)
{
	// Generate a texture ID
	glGenTextures(1, &image._textureID);
	// Make this the current texture (remember that GL is state-based)
	glBindTexture(GL_TEXTURE_2D, image._textureID);
	// Allocate the texture memory. The last parameter is NULL since we only
	// want to allocate memory, not initialize it
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA,
		GL_FLOAT, NULL);
	// Must set the filter mode, GL_LINEAR enables interpolation when scaling
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	GLenum gl_err = glGetError();
	cudaError err = cudaGraphicsGLRegisterImage(&image._resource, image._textureID, GL_TEXTURE_2D, 
		cudaGraphicsRegisterFlagsSurfaceLoadStore);
	if (err != cudaSuccess)
	{
		glDeleteBuffers(1, &image._textureID);
		fprintf(stderr, "cudaGraphicsGLRegisterImage failed!\n");
		return false;
	}

	return true;
}

cudaError createCudaSurface(cudaArray_t arr, cudaSurfaceObject_t &surfObj)
{
	//Create resource desc.
	struct cudaResourceDesc resDesc; 
	memset(&resDesc, 0, sizeof(resDesc)); 
	resDesc.resType = cudaResourceTypeArray; 
	resDesc.res.array.array = arr;
	
	return cudaCreateSurfaceObject(&surfObj, &resDesc);
}

cudaError CU_image::map(cudaArray_t &arr)
{
	cudaError err = cudaGraphicsMapResources(1, &_resource);
	if (err != cudaSuccess)
		return err;
	return cudaGraphicsSubResourceGetMappedArray(&arr, _resource, 0, 0);
}


cudaError CU_image::mapSurface(cudaSurfaceObject_t &surfObj)
{
	cudaArray_t arr;
	cudaError err = map(arr);
	if (err != cudaSuccess) return err;
	err = createCudaSurface(arr, surfObj);
	return err;
}
cudaError CU_image::unmap()
{
	return cudaGraphicsUnmapResources(1, &_resource);
}
cudaError CU_image::destroy()
{
	cudaError err = cudaGraphicsUnregisterResource(_resource);
	glDeleteBuffers(1, &_textureID);
	_textureID = 0;
	return err;
}



#endif