#include "CudaFuncs.h"
#include "Gaussian.h"
#include "GaussianMulti.h"
#include <device_functions.h>
#include<iostream>
#include <algorithm>
#include <string>


__global__ void gausEliminate_Row_Wise(float* A, float* b, int rows, int cols, int k, int row_offset)
{
	 //__shared__ float row_k[256];
	 extern __shared__ float row_k[];

	int row = blockIdx.x * blockDim.x + threadIdx.x + k + 1 + row_offset;
	int row_ind = row * cols; // The index of the first element in the row
	float pivot;
	if (row < rows)
		pivot = A[row_ind + k] / A[k*cols + k];

	for (int col = k; col < cols; col += blockDim.x)
	{
		// Iterate the row for blockDim.x times until new data from the k:th row is needed.
		int iter_end = min(blockDim.x, cols - col);
		// Fetch k:th row data
		if (col + threadIdx.x < cols)
			row_k[threadIdx.x] = A[k * cols + col + threadIdx.x];
		syncthreads(); //Wait for all threads to finish reading

		if (row < rows) //Every thread needs to fetch an element!
		{
			// Iterate 
			for (int i = 0; i < iter_end; i++)
				A[row_ind + col + i] -= pivot * row_k[i];
		}
		syncthreads(); // Wait until all rows/threads are done before next iter.
	}

	// Last step apply row elimination on b vector:
	if (row < rows)
		b[row] -= pivot * b[k];
	
}

Vector gaussSolveCudaMulti(Matrix& mat, Vector& b, int threads)
{
	int n = mat.row;
	Vector x;
	//Allocate
	float* dev_A = genCpy(mat.arr.get(), mat.col*mat.row);
	float* dev_b = genCpy(b.arr.get(), n);
	if (dev_A && dev_b)
	{
		dim3 grid(div_ceil(threads, 256), 1, 1);
		int iter_end = div_ceil(n, threads);
		// Resize arr:
		threads = std::min(256, threads);
		dim3 block(threads, 1, 1);

		//Execute:
		for (int k = 0; k < n - 1; k++)
		{
			greatestRowK <<<1, threads1D >>> (dev_A, mat.row, mat.col, k);

			swapRow << <grid, block >> > (dev_A, dev_b, mat.col, grid.x, k);

			//Launch a kernel (synced to limit threads) for every row:
			for (int i = 0; i < iter_end; i++)
				gausEliminate_Row_Wise << <grid, block, threads * sizeof(float) >> > (dev_A, dev_b, mat.row, mat.col, k, i * threads*grid.x);
			cudaDeviceSynchronize();
#ifdef DEBUG
			read(dev_A, mat.arr.get(), mat.col*mat.row);
			read(dev_b, b.arr.get(), n);
			if(mat.col * mat.row < 100)
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
	return x;
}