#include "CudaFuncs.h"
#include "Gaussian.h"
#include <device_functions.h>
#include<iostream>
#include <algorithm>
#include <string>

/* Variable storing the result of greatestRowK func*/
__device__ int greatest_row;

__device__  void swap(float* arr, int ind_a, int ind_b)
{
	float tmp = arr[ind_a];
	arr[ind_a] = arr[ind_b];
	arr[ind_b] = tmp;
}
/* Swap row k with row i using a single block.
*/
__global__ void swapRow(float* mat, float* b, int rows, int cols, int k)
{
	int row_i = greatest_row;
	if (k != row_i) //If the same row don't swap.
	{
		int row_k = k*cols;
		int swap_row = row_i*cols;
		//Swap matrix
		for (int i = threadIdx.x; i < cols; i += blockDim.x)
			swap(mat, swap_row + i, row_k + i);
		// Swap b
		if(threadIdx.x == 0)
			swap(b, row_i, k);
	}
}
/* Swap row k with row i using multiple blocks. Outputs the k:th column as a separate vector 
*/
__global__ void swapRow(float* mat, float* b, float* column_k, int rows, int cols, int k)
{
	int row_i = greatest_row;
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (k != row_i) //If the same row don't swap.
	{
		if (i < cols) //Ensure bounds
		{
			//Swap:
			float tmp = mat[k*cols + i];
			mat[k*cols + i] = mat[row_i*cols + i];
			mat[row_i*cols + i] = tmp;
		}
		//Swap vector b:
		else if (i == cols)
		{
			float tmp = b[k];
			b[k] = b[row_i];
			b[row_i] = tmp;
		}
	}
	//Store column k in a separate array: (A[k,k] is updated since the same warp swaps it).
	if (i < rows)
		column_k[i] = mat[i*cols + k];
}

__device__ int div2ceil(int value) { return (value & 1) + (value >> 1); }
__global__ void greatestRowK(float* mat, int rows, int cols, int k)
{
	__shared__ float col_k[threads1D];
	__shared__ int indices[threads1D];
	float max = 0.0f;
	int index = k;
	for (int i = k + threadIdx.x; i < rows; i += blockDim.x)
	{
		float value = fabsf(mat[i*cols + k]);
		if (value > max)
		{
			index = i;
			max = value;
		}
	}

	//Iterate the remaining comparisons by using half the threads on each iteration:
	col_k[threadIdx.x] = max;
	indices[threadIdx.x] = index;
	for (int i = div2ceil(threads1D); i > 1; i = div2ceil(i))//Div by 2
	{
		syncthreads();
		if (threadIdx.x < i)
		{
			float value = col_k[threadIdx.x + i];
			if (value > col_k[threadIdx.x])
			{
				indices[threadIdx.x] = indices[threadIdx.x + i];
				col_k[threadIdx.x] = value;
			}
		}
	}
	// Store result globaly, no need to sync last step is within the same warp as last iteration!
	// (only two thread executes the last iteration). Perform evaluation here...
	if (threadIdx.x == 0)
	{
		greatest_row = col_k[1] > col_k[0] ? indices[1] : indices[0];
#ifdef DEBUG
		printf("k, i, %d %d\n", k, greatest_row);
#endif
	}
}


__global__ void gausEliminate(float* A, float* b, float* column_k, int rows, int cols, int k)
{
	__shared__ float row_k[threads2D];
	__shared__ float col_k[threads2D];
	__shared__ float pivot;
	int row = blockIdx.y * blockDim.y + threadIdx.y + k + 1;
	int col = k + blockIdx.x * blockDim.x + threadIdx.x;
	//Fetch data:
	// Let first warp fetch row:
	if (threadIdx.y == 0 && col <= cols)
		row_k[threadIdx.x] = col == cols ? b[k] : A[k * cols + col];
	// Let second warp fetch column:
	else if (threadIdx.y == 1 && blockIdx.y * blockDim.y + threadIdx.x + k + 1 < rows)
		col_k[threadIdx.x] = column_k[blockIdx.y * blockDim.y + threadIdx.x + k + 1];
	//Let third warp fetch pivot element:
	else if (threadIdx.y == 2 && threadIdx.x == 0)
		pivot = A[k*cols + k];
	syncthreads();

	if (col <= cols && row < rows)
	{
		// Find the value of L_(i,j) related to this thread: 
		float elim = col_k[threadIdx.y] / pivot * row_k[threadIdx.x];
		//printf("(%d, %d, %f, %f), ", row, col, col_k[threadIdx.y], row_k[threadIdx.x]);
		if (col == cols) //The last thread column is used to update the vector!
			b[row] -= elim;
		else
		{
			//printf("i,e: %d, %f\n", block + threadIdx.y * cols + threadIdx.x, elim);
			A[row * cols + col] -= elim;
		}
	}
}

Vector gaussSolveCuda(Matrix& mat, Vector& b)
{
	int n = mat.row;
	Vector x;
	//Allocate
	float* dev_A = genCpy(mat.arr.get(), mat.col*mat.row);
	float* dev_b = genCpy(b.arr.get(), n);
	float* dev_col = allocFloat(n);
	if (dev_A && dev_b && dev_col)
	{
		dim3 block1D(threads1D, 1, 1);
		dim3 block2D(threads2D, threads2D, 1);
		dim3 gridSwap(div_ceil(mat.col+1,threads1D), 1, 1);
		//Execute:
		for (int k = 0; k < n; k++)
		{
			greatestRowK <<<1,block1D>>> (dev_A, mat.row, mat.col, k);

			swapRow << <gridSwap, block1D >> > (dev_A, dev_b, dev_col, mat.row, mat.col, k);

			dim3 gridElim(div_ceil(mat.col - k + 1,threads2D), div_ceil(mat.row, threads2D), 1);
			gausEliminate<<<gridElim,block2D>>>(dev_A, dev_b, dev_col, mat.row, mat.col, k);
			
#ifdef DEBUG
			read(dev_A, mat.arr.get(), mat.col*mat.row);
			read(dev_b, b.arr.get(), n);
			print(mat);
			print(b);
#endif
		}
		//Fetch data
		read(dev_A, mat.arr.get(), mat.col*mat.row);
		read(dev_b, b.arr.get(), n);
		x = backSubstitute(mat, b, n);
	}
	cudaFree(dev_A);
	cudaFree(dev_b);
	cudaFree(dev_col);
	return x;
}