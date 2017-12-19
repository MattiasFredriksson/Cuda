#pragma once
#include "GaussianSingle.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#define DEBUG

// Guard the __syncthreads() command as it is not recognized:
#ifdef __INTELLISENSE__
#define syncthreads()
#else
#define syncthreads() __syncthreads()
#endif

Vector gaussSolveCudaRowWise(Matrix& mat, Vector& v, int threads);