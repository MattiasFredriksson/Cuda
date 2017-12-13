#include "GaussianSingle.h"
#include "Gaussian.h"
#include "GaussianMulti.h"
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


	bool thread_mode = false;
	int num_repeat = 5;
	int mode_0 = 2, mode_1 = 3;
	int num_mode = thread_mode ? 3 : mode_1;
	int max_elem = thread_mode ? 12 : 14;
	int min_elem = 5;

	for (int mode = thread_mode ? 2 : mode_0; mode < num_mode; mode++)
	{
		log.logMsg("Mode: " + std::to_string(mode));
		std::cout << "Executing mode: " << std::to_string(mode) << std::endl;
		for (int num = min_elem; num < max_elem; num++)
		{
			//Num elem
			int elem_count = thread_mode ? 4096 : 2 << num;
			if (mode == 0 && elem_count > 4096*2) continue;
			int threads = thread_mode ? 2 << num : elem_count;
			//Print run info
			std::cout << "N: " << elem_count;
			if (thread_mode) std::cout << "  T: " << threads;
			std::cout << std::endl;

			double total_time = 0;
#ifdef DEBUG
			if (mat.col * mat.row < 100)
				print(mat);
#endif

			for (int i = 0; i < num_repeat; i++)
			{
				Matrix mat;
				Vector vec;
				//exampleA(mat, vec);
				exampleB(mat, vec, elem_count, 2);
				std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

				Vector x;
				//Execute:
				//for(int i = 0; i < 10; i++)
				switch (mode)
				{
				case 1:
					x = gaussSolveCuda(mat, vec);
					break;
				case 2:
					x = gaussSolveCudaMulti(mat, vec, threads);
					break;
					/*
				case 3:
					x = gaussSolveCudaDevice(mat, vec);
					break;
					*/
				case 0:
				default:
					x = gaussSolve(mat, vec);
					break;
				}

				std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
				std::cout << std::endl;

				std::cout << "\n";
				//Time measure: cast to microseconds/nanoseconds for better perf:
				double sec = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0;
				total_time += sec;
				//log.logMsg("sec/n: " + std::to_string(sec) + "\t" + std::to_string(arr_len));
				std::cout << "Execution time = " << sec << " s" << std::endl;
#ifdef Debug
				if (mat.col < 5)
					print(mat);
#endif
				std::cout << "Solution x:\n";
				print(x, 5);
			}
			double avg_sec = total_time / num_repeat;
			//Construct message:
			std::string msg = (mode == 2 ? "sec/n/threads" : "sec/n");
			msg += "\t" + std::to_string(avg_sec) + "\t" + std::to_string(elem_count);
			if (mode == 2) msg += "\t" + std::to_string(threads);
			log.logMsg(msg);

			std::cout << "Test complete!\n";
			std::cout << "Avg. Time difference = " << avg_sec << " s" << std::endl;
		}
	}


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