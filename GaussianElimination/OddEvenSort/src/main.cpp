#include "GaussianSingle.h"
#include "Gaussian.h"
#include "GaussianDevice.h"
#include "RandomGenerator.h"
#include <iostream>
#include <algorithm>
#include<chrono>
#include "Log.h"


void exampleA(Matrix& mat, Vector& vec);
void exampleB(Matrix& mat, Vector& vec, int n, float scale);

int main()
{
	mf::RandomGenerator rnd;
	mf::Log log("Counters.txt", false, false, true);

	Matrix mat;
	Vector vec;
	//exampleA(mat, vec);
	exampleB(mat, vec, 4096, 1);
#ifdef DEBUG
	print(mat);
#endif

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	Vector x;
	//Execute:
	for(int i = 0; i < 10; i++)
		//x = gaussSolveCuda(mat, vec);
		x = gaussSolveCudaDevice(mat, vec);
	//x = gaussSolve(mat, vec);

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << std::endl;

	if(mat.col < 5)
		print(mat);
	std::cout << "Solution x: ";
	print(x);

	std::cout << "\n";
	//Time measure: cast to microseconds/nanoseconds for better perf:
	double sec = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0;
	//log.logMsg("sec/n: " + std::to_string(sec) + "\t" + std::to_string(arr_len));
	std::cout << "Execution time = " << sec << " s" <<std::endl;

	std::getchar();
}


void memTest()
{
	for (int i = 0; i < 10; i++)
		//singleWarpTransaction();
		perWarpTransaction();
}

void exampleA(Matrix& mat, Vector& vec)
{
	mat = Matrix(3, 3);
	mat[0] = 2; mat[1] = -4; mat[2] = 4;
	mat[3] = 6; mat[4] = -10; mat[5] = 7;
	mat[6] = -1; mat[7] = -4; mat[8] = 8;

	vec = Vector(3);
	vec[0] = -2; vec[1] = -3; vec[2] = 0;
}
void exampleB(Matrix& mat, Vector& vec, int n, float scale)
{
	mat = Matrix(n, n);
	vec = Vector(n);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
			mat[i * n + j] = (j + 1 + std::max(0, j-i)) / scale;
		vec[i] = (i+1) / scale;
	}
}