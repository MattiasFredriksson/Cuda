#pragma once
#include "GaussianSingle.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#define DEBUG

// Threads per block
const int threads1D = 256;
const int threads2D = 16;

// Guard the __syncthreads() command as it is not recognized:
#ifdef __INTELLISENSE__
#define syncthreads()
#else
#define syncthreads() __syncthreads()
#endif

inline int div_ceil(int numerator, int denominator);

Vector gaussSolveCuda(Matrix& mat, Vector& v);


__global__ void swapRow(float* mat, float* b, float* column_k, int rows, int cols, int k);
__global__ void greatestRowK(float* mat, int rows, int cols, int k);