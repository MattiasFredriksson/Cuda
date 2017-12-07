#include "CudaFuncs.h"
#include "OddEven.h"
#include <device_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>
#include <algorithm>


// Guard the __syncthreads() command as it is not recognized:
#ifdef __INTELLISENSE__
#define syncthreads()
#else
#define syncthreads() __syncthreads()
#endif

// Threads per block and also the number of iterations per kernel
const int threads = 256;

#pragma region simple version

/* Simple sort kernel, executing a single odd even sort
*/
__global__ void sortKernelSimple(int *arr, int arr_len, int odd)
{
	int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x) + odd;
	if (i < arr_len - 1)
	{
		//Even
		int a = arr[i];
		int b = arr[i + 1];
		if (a > b)
		{
			arr[i] = b;
			arr[i + 1] = a;
		}
	}
}

/* Single threaded sort func. */
void oddEvenSortCudaSimple(int* arr, unsigned int arr_len)
{
	// Init device:
	if (!init()) return;
	int* dev_arr = genCpy(arr, arr_len);
	if (dev_arr)
	{
		//Execute:
		int num_block = div_ceil(arr_len, 2*threads);
		dim3 block(threads, 1, 1);
		dim3 grid(num_block, 1, 1);

		for (unsigned int i = 0; i < arr_len; i++)
		{
			int odd = i & 1;
			sortKernelSimple <<<grid, block>>> (dev_arr, arr_len, odd);
		}
	}
	//Transfer result to RAM:
	read(dev_arr, arr, arr_len);
	//Done
	cudaFree(dev_arr);
}

#pragma endregion

#pragma region MultiKernel
//Simple version running multiple items per thread.

__global__ void sortKernelMulti(int *arr, int arr_len, int num_elem, int oddEven)
{
	int i = 2 * (blockIdx.x * blockDim.x * num_elem) + oddEven;
	int iterEnd = min(arr_len - 1, i + 2 * blockDim.x *num_elem);
	// Increment to thread start index:
	i += 2 * threadIdx.x;
	// Every thread in block (warp) step by num_elem
	for (; i < iterEnd; i += 2 * blockDim.x)
	{
		//Even
		int a = arr[i];
		int b = arr[i + 1];
		if (a > b)
		{
			arr[i] = b;
			arr[i + 1] = a;
		}
	}
}

void oddEvenSortCudaMulti(int* arr, unsigned int arr_len, int threads)
{
	// Init device:
	if (!init()) return;
	int* dev_arr = genCpy(arr, arr_len);
	if (dev_arr)
	{
		//Execute:
		int num_elem = div_ceil(arr_len, 2 * threads);
		int num_block = div_ceil(threads, 256);
		dim3 block(256, 1, 1);
		dim3 grid(num_block, 1, 1);

		for (unsigned int i = 0; i < arr_len; i++)
		{
			int odd = i & 1;
			sortKernelMulti <<<grid, block >>>(dev_arr, arr_len, num_elem, odd);
		}
	}
	//Transfer result to RAM:
	read(dev_arr, arr, arr_len);
	//Done
	cudaFree(dev_arr);
}


#pragma endregion

__global__ void sortKernel(int *d_arr, int arr_len, int offset)
{
	__shared__ int arr[2 * threads];

	int block_i = 2 * blockIdx.x * blockDim.x + offset;
	int i = 2* threadIdx.x;

	//Load into shared mem:
	int acc_ind = block_i + threadIdx.x;
	arr[threadIdx.x] = d_arr[acc_ind];
	arr[threadIdx.x + blockDim.x] = d_arr[acc_ind + blockDim.x];
	syncthreads();

	if (block_i + i < arr_len - 1)
	{
		//Repeat threads times = one step (even or uneven) per thread.
		for (int n = 0; n < threads; n++)
		{
			//Even
			int a = arr[i];
			int b = arr[i + 1];
			if (a > b)
			{
				arr[i] = b;
				b = a;
			}
			syncthreads();
			if (threadIdx.x < blockDim.x - 1 && block_i + i < arr_len - 2) //Not if last block thread or out of bounds!
			{
				//Uneven
				int c = arr[i + 2];

				if (b > c)
				{
					arr[i + 2] = b;
					b = c;
				}
			}
			// Only this thread accesses this element and it's unlikely no thread in the warp swaps it.
			arr[i + 1] = b;
			syncthreads();
		}
	}
	//Write back to global:
	d_arr[acc_ind] = arr[threadIdx.x];
	d_arr[acc_ind + blockDim.x] = arr[threadIdx.x + blockDim.x];
}

/* Single threaded sort func. */
void oddEvenSortCuda(int* arr, unsigned int arr_len)
{
	int num_block = div_ceil(arr_len, 2 * threads);
	int num_step = div_ceil(arr_len, threads);
	// Init device:
	if (!init()) return;
	// Alloc arr, padding it removes access checks for threads out of bounds:
	int* dev_arr = genCpy(num_block * 2 * threads, arr, arr_len);
	if (dev_arr)
	{
		//Execute:
		dim3 block(threads, 1, 1);
		dim3 grid_even(num_block, 1, 1);
		dim3 grid_uneven(num_block - 1, 1, 1);

		// Sync. blocks using odd/even every n:th it!
		for (int i = 0; i < num_step; i++)
		{
			int odd = i & 1;
			int offset = odd * threads;
			// Launch a kernel on the GPU with one thread for every ~2 elements.
			if (odd)
				sortKernel << <grid_uneven, block >> > (dev_arr, arr_len, offset);
			else
				sortKernel << <grid_even, block >> > (dev_arr, arr_len, offset);

		}
	}
	//Transfer result to RAM:
	read(dev_arr, arr, arr_len);
	//Done
	cudaFree(dev_arr);
}