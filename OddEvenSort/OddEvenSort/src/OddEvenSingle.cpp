#include "OddEvenSingle.h"
#include <algorithm>
#include <iostream>




void oddEvenSortSingle(int* arr, unsigned int arr_len)
{	
	int end_index = arr_len - 1; //Exclude last element!
	int iter_sorted = 0;
	for (int i = 0; i < arr_len; i++)
	{
		//Sort:
		int num_sorted = 0;
		// Iterate elems:
		for (unsigned int ii = i & 1; ii < end_index; ii += 2)
		{
			if (arr[ii] > arr[ii + 1])
			{
				std::swap(arr[ii], arr[ii + 1]);
				num_sorted++;
			}
		}
		//Terminate if two consecutive iterations are sorted:
		iter_sorted = num_sorted ? 0 : iter_sorted + 1;
		if (iter_sorted == 2)
			break;
	}
}

void verify(int* arr, unsigned int arr_len) 
{
	int num_unsorted = 0;
	for (unsigned int i = 0; i < arr_len - 1; i++)
	{
		if (arr[i] > arr[i + 1])
		{
			num_unsorted++;
			//Output segment
			std::cout << i << ": ";
			if(i-1 > 0) std::cout << arr[i-1] << ", ";
			std::cout << arr[i] << ", ";
			if (i + 1 < arr_len) std::cout << arr[i + 1] << std::endl;
		}
	}
	
	if (num_unsorted == 0)
		std::cout << "Sorted arr!\n";
	else
		std::cout << "Sort failed: " << num_unsorted << " variables remain to be sorted!\n";
}


void printArr(int* arr, unsigned int from, unsigned int to)
{
	for (unsigned int i = from; i < to; i++)
		std::cout << arr[i] << ", ";
	std::cout << std::endl;
}