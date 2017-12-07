#pragma once

/* Initiate device. */
bool init();
/* Allocate an integer array. */
int* allocInt(int n);
/* Transfer data to device array. */
bool transferDevice(int* arr, int* dev_arr, int n);
/* Generate a copy of the data on the device. */
int* genCpy(int* arr, unsigned int arr_len);
/* Generate a copy of the data on the device. 
num_alloc	<<	Number of elements allocated, including padding elements.
arr			<<	The actual data array.
arr_len		<<	The length of the data array.
*/
int* genCpy(unsigned int num_alloc, int* arr, unsigned int arr_len);

bool read(int* dev_arr, int* arr, int arr_len);