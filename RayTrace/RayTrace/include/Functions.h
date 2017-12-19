#pragma once
#include <cstdlib>


inline int div_ceil(int numerator, int denominator)
{
	std::div_t res = std::div(numerator, denominator);
	return res.quot + (res.rem != 0);
}