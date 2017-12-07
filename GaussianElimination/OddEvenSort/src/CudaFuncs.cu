#include "CudaFuncs.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>


/* Initiate device. */
bool init()
{
	cudaError_t err = cudaSetDevice(0);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return false;
	}
	return true;
}
/* Allocate an integer array. */
int* allocInt(int n)
{
	int* dev_arr = 0;
	//Alloc
	cudaError_t err = cudaMalloc((void**)&dev_arr, n * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return NULL;
	}
	return dev_arr;
}
/* Allocate an float array. */
float* allocFloat(int n)
{
	float* dev_arr = 0;
	//Alloc
	cudaError_t err = cudaMalloc((void**)&dev_arr, n * sizeof(float));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return NULL;
	}
	return dev_arr;
}
/* Transfer data to device array. */
bool transferDevice(int* arr, int* dev_arr, unsigned int n)
{
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus = cudaMemcpy(dev_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return false;
	}
	return true;
}
/* Transfer data to device array. */
bool transferDevice(float* arr, float* dev_arr, unsigned int n)
{
	// Copy input vectors from host memory to GPU buffers.
	cudaError_t cudaStatus = cudaMemcpy(dev_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return false;
	}
	return true;
}
/* Generate a copy of the data on the device. */
int* genCpy(int* arr, unsigned int arr_len)
{
	int* dev_arr = 0;
	if (!(dev_arr = allocInt(arr_len))) return NULL;
	if (!transferDevice(arr, dev_arr, arr_len)) return NULL;
	return dev_arr;
}
/* Generate a copy of the data on the device. */
float* genCpy(float* arr, unsigned int arr_len)
{
	float* dev_arr = 0;
	if (!(dev_arr = allocFloat(arr_len))) return NULL;
	if (!transferDevice(arr, dev_arr, arr_len)) return NULL;
	return dev_arr;
}
/* Generate a copy of the data on the device. */
int* genCpy(unsigned int num_alloc, int* arr, unsigned int arr_len)
{
	int* dev_arr = 0;
	if (!(dev_arr = allocInt(num_alloc))) return NULL;
	if (!transferDevice(arr, dev_arr, arr_len)) return NULL;
	return dev_arr;
}

bool read(int* dev_arr, int* arr, unsigned int arr_len)
{
	// Copy output vector from GPU buffer to host memory.
	cudaError_t cudaStatus = cudaMemcpy(arr, dev_arr, arr_len * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return false;
	}
	return true;
}
bool read(float* dev_arr, float* arr, unsigned int arr_len)
{
	// Copy output vector from GPU buffer to host memory.
	cudaError_t cudaStatus = cudaMemcpy(arr, dev_arr, arr_len * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return false;
	}
	return true;
}