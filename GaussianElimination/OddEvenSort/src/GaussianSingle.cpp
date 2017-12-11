#include "GaussianSingle.h"
#include <algorithm>
#include <iostream>

/* Find the greatest element in column k of the matrix and 
 * then swap the row containing the element to the k:th row.
 */
bool greatestRowK(Matrix mat, Vector v, int k)
{
	//Find:
	float max = std::abs(mat.getRow(k)[k]);
	int index = k;
	for (int row_i = k + 1; row_i < mat.row; row_i++)
	{
		float* row = mat.getRow(row_i);
		if (std::abs(row[k]) > max)
		{
			max = std::abs(row[k]);
			index = row_i;
		}
	}
	// Permutate the matrix:
	if (index == k) return true;		// Same row
	if (max < 0.000001f) return false;	// Singular matrix, failure
	float* a = mat.getRow(k);
	float* b = mat.getRow(index);
	for (int i = 0; i < mat.row; i++)
		std::swap(a[i], b[i]);

	// Permute vector
	std::swap(v[k], v[index]);
	return true;
}
void gaussEliminate(Matrix mat, Vector v, int k)
{
	float* row_k = mat.getRow(k);
	float pivot = row_k[k];
	for (int row_i = k + 1; row_i < mat.row; row_i++)
	{
		float* row = mat.getRow(row_i);
		float mult = row[k] / pivot;

		//Eliminate row pivot elem: (apply Lk_inv)
		for (int i = k; i < mat.col; i++)
			row[i] -= row_k[i] * mult;
		//Apply Lk_inv on vec:
		v[row_i] -= v[k] * mult;
	}
}

Vector backSubstitute(Matrix mat, Vector b, int n)
{
	Vector x = Vector(n);

	// Backsubstitute n rows:
	for (int k = n - 1; k >= 0; k--)
	{
		float* row = mat.getRow(k);
		float sum = b[k];
		for (int i = k + 1; i < mat.col; i++)
			sum -= x[i] * row[i];
		x[k] = sum / row[k];
	}
	return x;
}

Vector gaussSolve(Matrix mat, Vector v)
{
	int max_iter = std::min(mat.col, mat.row);
	for (int i = 0; i < max_iter; i++)
	{
		if (!greatestRowK(mat, v, i))
			return Vector(v.length);
		gaussEliminate(mat, v, i);
	}
	return backSubstitute(mat, v, max_iter);
}



void print(Matrix mat)
{
	std::cout << "Mat:\n";
	for (int row = 0; row < mat.row; row++)
	{
		float* row_v = mat.getRow(row);
		for (int col = 0; col < mat.col; col++)
			std::cout << row_v[col] << ",\t";
		std::cout << "\n";
	}
}
void print(Vector vec)
{
	std::cout << "Vec: ";
	for (int i = 0; i < vec.length; i++)
		std::cout << vec[i] << ",\t";
	std::cout << "\n";
}

void print(Vector vec, int n)
{
	std::cout << "Vec: ";
	for (int i = 0; i < vec.length && i < n; i++)
		std::cout << vec[i] << ",\t";
	if (n < vec.length)
	{
		std::cout << " . . . ";
		n /= 2;
		for (int i = vec.length - n; i < vec.length; i++)
			std::cout << vec[i] << ",\t";
	}
	std::cout << "\n";

}