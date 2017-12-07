#pragma once
#include "GaussianSingle.h"



inline int div_ceil(int numerator, int denominator)
{
	std::div_t res = std::div(numerator, denominator);
	return res.quot + (res.rem != 0);
}

Vector gaussSolveCudaDevice(Matrix& mat, Vector& v);



//Some mem. test funcs:


void perWarpTransaction();
void singleWarpTransaction();