#include "GaussianDevice.h"
#include "Gaussian.h"
#include "CudaFuncs.h"
#include "device_launch_parameters.h"
#include<iostream>
#include <algorithm>
#include <string>

__global__ void gausEliminate_v2(float* mat, float* b, float* column_k, int rows, int cols, int k)
{
	__shared__ float row_k[threads2D];
	__shared__ float col_k[threads2D];
	__shared__ float pivot;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = k + blockIdx.x * blockDim.x + threadIdx.x;
	//Fetch data:
	// Let first warp fetch row:
	if (threadIdx.y == 0 && col <= cols)
		row_k[threadIdx.x] = col == cols ? b[k] : mat[k * cols + col];
	// Let second warp fetch column:
	else if (threadIdx.y == 1 && blockIdx.y * blockDim.y + threadIdx.x < rows)
		col_k[threadIdx.x] = column_k[blockIdx.y * blockDim.y + threadIdx.x];
	//Let third warp fetch pivot element:
	else if (threadIdx.y == 2 && threadIdx.x == 0)
		pivot = mat[k*cols + k];
	syncthreads();

	if (col <= cols && row < rows)
	{
		if (row != k)
		{
			// Find the value of L_(i,j) related to this thread: 
			float elim = col_k[threadIdx.y] / pivot * row_k[threadIdx.x];
			//printf("(%d, %d, %f, %f), ", row, col, col_k[threadIdx.y], row_k[threadIdx.x]);
			if (col == cols) //The last thread column is used to update the vector!
				b[row] -= elim;
			else
			{
				//printf("i,e: %d, %f\n", block + threadIdx.y * cols + threadIdx.x, elim);
				mat[(blockIdx.y * blockDim.y + threadIdx.y) * cols + blockIdx.x * blockDim.x + threadIdx.x + k] -= elim;
			}
		}
	}
}
/* Sovle the remaining diagonal matrix! */
__global__ void solve(float* mat, float* b, float* x, int rows, int cols)
{
	int n = blockIdx.x*threads1D + threadIdx.x;
	if (n < rows) //Ensure bounds
		x[n] = b[n] / mat[n * cols + n];
}

Vector gaussSolveCudaDevice(Matrix& mat, Vector& b)
{
	int n = mat.row;
	Vector x(n);
	//Allocate
	float* dev_mat = genCpy(mat.arr.get(), mat.col*mat.row);
	float* dev_b = genCpy(b.arr.get(), n);
	float* dev_col = allocFloat(mat.row);
	float* dev_x = allocFloat(n);
	if (dev_mat && dev_b && dev_x)
	{
		dim3 block1D(threads1D, 1, 1);
		dim3 block2D(threads2D, threads2D, 1);
		read(dev_mat, mat.arr.get(), mat.col*mat.row);
		//Execute:
		for (int k = 0; k < n; k++)
		{
			greatestRowK <<<1,block1D>>> (dev_mat, mat.row, mat.col, k);

			dim3 gridSwap(div_ceil(mat.col+1,threads1D), 1, 1);
			swapRow << <gridSwap, block1D >> > (dev_mat, dev_b, dev_col, mat.row, mat.col, k);

			dim3 gridElim(div_ceil(mat.col - k + 1,threads2D), div_ceil(mat.row, threads2D), 1);
			gausEliminate_v2<<<gridElim,block2D>>>(dev_mat, dev_b, dev_col, mat.row, mat.col, k);
			
#ifdef DEBUG
			read(dev_mat, mat.arr.get(), mat.col*mat.row);
			read(dev_b, b.arr.get(), n);
			print(mat);
			print(b);
#endif
		}

		dim3 gridSolve(div_ceil(mat.col, threads1D), 1, 1);
		solve<<<gridSolve, block1D>>> (dev_mat, dev_b, dev_x, mat.row, mat.col);
		//Fetch data
		read(dev_x, x.arr.get(), n);
		read(dev_mat, mat.arr.get(), mat.col*mat.row);
	}
	cudaFree(dev_mat);
	cudaFree(dev_b);
	cudaFree(dev_col);
	cudaFree(dev_x);
	return x;
}



#pragma region Tests

/* Memory test:
 * Access column major mem. per warp instead of in one warp.
 * Conclussion: No difference...
*/
__global__ void singleWarpTransaction(float* arr, int rows, int cols)
{
	__shared__ float shared[64];
	int block = blockIdx.y * blockDim.y * cols + blockIdx.x * blockDim.x;
	if (threadIdx.y < 2)
		shared[threadIdx.x + blockDim.x * threadIdx.y] = arr[block + threadIdx.x*cols];
	syncthreads();
	//Repeat local over block y:
	arr[block + threadIdx.y * cols + threadIdx.x] = shared[threadIdx.x] + shared[blockDim.x + threadIdx.y];
}

__global__ void perWarpTransaction(float* arr, int rows, int cols)
{
	__shared__ float shared[64];
	int block = blockIdx.y * blockDim.y * cols + blockIdx.x * blockDim.x;
	if (threadIdx.x == 0 || threadIdx.y == 0)
		shared[threadIdx.x + threadIdx.y + blockDim.y * (threadIdx.y > 0)] = arr[block + threadIdx.y*cols + threadIdx.x];
	shared[blockDim.x] = shared[0];
	syncthreads();
	//Repeat local over block y:
	arr[block + threadIdx.y * cols + threadIdx.x] = shared[threadIdx.x] + shared[blockDim.x + threadIdx.y];
}

const int size = 16384 * 2;
void singleWarpTransaction()
{
	int arr_size = size * size;
	float* arr = new float[arr_size];
	float* dev_arr = allocFloat(arr_size);
	if (dev_arr)
	{
		//Execute:
		dim3 block(32, 32, 1);
		dim3 grid(size / 32, size / 32, 1);
		singleWarpTransaction<<<grid, block >> > (dev_arr, size, size);
	}
	read(dev_arr, arr, arr_size);
	cudaFree(dev_arr);
	delete[] arr;
}
void perWarpTransaction()
{
	int arr_size = size * size;
	float* arr = new float[arr_size];
	float* dev_arr = allocFloat(arr_size);
	if (dev_arr)
	{
		//Execute:
		dim3 block(32, 32, 1);
		dim3 grid(size/32, size/32, 1);
		perWarpTransaction << <grid, block >> > (dev_arr, size, size);
	}
	read(dev_arr, arr, arr_size);
	cudaFree(dev_arr);
	delete[] arr;
}
#pragma endregion
