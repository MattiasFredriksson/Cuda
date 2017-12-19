#include "RandomGenerator.h"


#pragma region HEADER
//--------------------------------------------------------------------------------------
// File: RandomGenerator.h
// Project: Function library
//
// Author Mattias Fredriksson 2017.
//--------------------------------------------------------------------------------------

#include"RandomGenerator.h"
#include<memory>
#include<cmath>
#include<algorithm>
#include<iostream>
#pragma endregion

namespace mf {
	/* Minimum number of seeds allowed to be generated
	*/
	const unsigned int mt32MinSeedSize = 4;
	/*	Number of 32 bit ints used by the Mersenne Twister (ie. the actual number of 32 int seeds required).
	*/
	const unsigned int mt32MaxSeedSize = 624;
	

	/* Generate a random initializer list with random numbers generated trough the random_device.
	*/
	std::vector<std::random_device::result_type> generateSeedSeq(unsigned int seedCount) {
		std::random_device seeder;
		std::vector<std::random_device::result_type> vec(seedCount);
		//Fetch seeds.
		std::cout << "Seed: ";
		for (unsigned int i = 0; i < seedCount; i++)
		{
			vec[i] = seeder();
			std::cout << vec[i] << ", ";
		}
		std::cout << std::endl;
		return vec;
	}


	/* Construct a seeded random generator
	*/
	RandomGenerator::RandomGenerator()
		:rng() {
		seedGenerator();
	}
	/* Construct a seeded random generator.
	seedCount	<<	Number of seeds used to seed the number engine.
	*/
	RandomGenerator::RandomGenerator(unsigned char seedCount)
		: rng() {
		seedGenerator(seedCount);
	}
	/* Construct a random generator seeded from a list beginning and ending at the pointers. Should use atleast 64 bit for seeding.
	*/
	RandomGenerator::RandomGenerator(const unsigned int* begin, const unsigned int* end)
		: rng() {
		setSeed(begin, end);
	}
	/* Construct a random generator seeded from a list. Should use atleast 64 bit for seeding.
	*/
	RandomGenerator::RandomGenerator(const std::initializer_list<unsigned int>& list)
		: rng() {
		setSeed(list);
	}

	/* Construct a random generator seeded from another random distributor.
	*/
	RandomGenerator::RandomGenerator(const mf::RandomGenerator& seeder)
		: rng() {
		std::vector<unsigned int> seed = seeder.generateSeed(mt32MaxSeedSize);
		setSeed(seed);
	}
	RandomGenerator::~RandomGenerator() {
	}


	/*	Seed the generator with a new seed.
	*/
	void RandomGenerator::seedGenerator() {
		//Generate a seed sequence of 128 bits
		std::vector<std::random_device::result_type> vec = generateSeedSeq(mt32MinSeedSize);
		rng.seed(std::seed_seq(vec.begin(), vec.end()));
	}
	/*	Seed the generator by generating a set of seeds for the random engine.
	*/
	void RandomGenerator::seedGenerator(unsigned int seedCount) {
		std::vector<std::random_device::result_type> vec = generateSeedSeq(std::min(std::max(seedCount, mt32MinSeedSize), mt32MaxSeedSize));
		rng.seed(std::seed_seq(vec.begin(), vec.end()));
	}
	/* Seed the engine from the seeds specified in list beginning and ending at the pointers. Should use atleast 128 bit for seeding.
	*/
	void RandomGenerator::setSeed(const unsigned int* begin, const unsigned int* end) {
		rng.seed(std::seed_seq(begin, end));
	}
	/*	Seed the generator from the seeds specified in the list. Should use atleast 128 bit for seeding.
	*/
	void RandomGenerator::setSeed(const std::initializer_list<unsigned int>& list) {
		rng.seed(std::seed_seq(list));
	}
	/*	Seed the generator from the seeds specified in the list. Should use atleast 128 bit for seeding.
	*/
	void RandomGenerator::setSeed(const std::vector<unsigned int>& vec) {
		rng.seed(std::seed_seq(vec.begin(), vec.end()));
	}



	/* Generate a seed as a set of unsigned integers. */
	std::vector<unsigned int> RandomGenerator::generateSeed(unsigned int numSeedInt) const {
		std::vector<unsigned int> list;
		std::uniform_int_distribution<unsigned int> uni;
		for (unsigned int i = 0; i < numSeedInt; i++)
			list.push_back(uni(rng));
		return list;
	}
	/*
	Generate a random unsigned byte
	*/
	char RandomGenerator::randomUByte() const {
		std::uniform_int_distribution<int> uni(0, UCHAR_MAX);
		return (char)uni(rng);
	}

	int RandomGenerator::randomInt(int min, int max) const {
		std::uniform_int_distribution<int> uni(min, max);
		return uni(rng);
	}
	int RandomGenerator::randomInt(int max) const {
		std::uniform_int_distribution<int> uni(0, max);
		return uni(rng);
	}
	int RandomGenerator::randomInt() const {
		std::uniform_int_distribution<int> uni(INT_MIN, INT_MAX);
		return uni(rng);
	}

	float RandomGenerator::randomFloat(float min, float max) const {
		std::uniform_real_distribution<float> uni(min, max);
		return uni(rng);
	}
	float RandomGenerator::randomFloat(float max) const {
		std::uniform_real_distribution<float> uni(0, max);
		return uni(rng);
	}
	float RandomGenerator::randomFloat() const {
		static const std::uniform_real_distribution<float> uni(FLT_MIN, FLT_MAX);
		return uni(rng);
	}
	float RandomGenerator::randomUnitFloat() const
	{
		static const std::uniform_real_distribution<float> uni(0, 1);
		return uni(rng);
	}

	/*	Generates a random highprecision floating point value between [min-max]
	*/
	double RandomGenerator::randomDouble(double min, double max) const {
		std::uniform_real_distribution<double> uni(min, max);
		return uni(rng);
	}
	double RandomGenerator::randomUnitDouble() const {
		static const std::uniform_real_distribution<double> uni(0.0, 1.0);
		return uni(rng);
	}

	/* Permutate the array using the Knuth shuffle distributed by the generator.
	*/
	void RandomGenerator::knuthShuffle(int* arr, int size) const
	{
		int tmp, j;
		//Step through the array swapping an index with an index in the remaining greater range of [i,n]
		for (int i = 0; i < size - 2; i++)
		{
			//Generate a int in the remaining range: 
			j = i + randomInt(size - i - 1);
			//Swap
			tmp = arr[i];
			arr[i] = arr[j];
			arr[j] = tmp;
		}
	}

	/* Permutate the array using the Knuth shuffle distributed by the generator.
	*  Shuffle a specified number of times.
	*/
	void RandomGenerator::knuthShuffle(int* arr, int size, unsigned int num_times) const
	{
		for (unsigned int i = 0; i < num_times; i++)
			knuthShuffle(arr, size);
	}

	/* Generate an biased distribution by repeating the range over the array size and then shuffled using the Knuth algorithm.
	*/
	void RandomGenerator::shuffledIntDistribution(int* arr, unsigned int size, int min, int max) const
	{
		//Generate repeated distribution:
		for (unsigned int i = 0; i < size; i++)
			arr[i] = i % (max - min + 1) + min;
		//Shuffle
		knuthShuffle(arr, size);
	}

	/* Generate an biased distribution by repeating the range over the array size and then shuffled using the Knuth algorithm.
	* Range is defined as [min, max].
	*/
	std::unique_ptr<int> RandomGenerator::shuffledIntDistribution(unsigned int size, int min, int max) const
	{
		std::unique_ptr<int> arr(new int[size]);
		shuffledIntDistribution(arr.get(), size, min, max);
		return std::move(arr);
	}
	/* Generates a normalized vector from a set of random float values.
	*/
	float3 RandomGenerator::randomNormal() const {
		/* Uniform distribution over a unit sphere.
		*	1.	Distribute values over a surface S^2 with bounds X : [0,2*Pi), Y : [-1,1].
		Where Y is defined from [0,Pi] where cosine of spherical coordinates gives uniform distribution in [-1,1].
		The formula can be viewed as a distribution over a rectangle representing the surface of a hollow cylinder and y = z.
		*	2.	Use the distributed z coordinate to find the radius of the circle intersected by the plane orthogonal to the z axis.
		*	3.	Find the normal vector by using the radius and distributed (X) theta angle to calculate the remaining x,y coordinates.
		*/

		// Distribution ranges: 
		const static std::uniform_real_distribution<float> negone_one(-1, 1);
		const static std::uniform_real_distribution<float> zero_2Pi(0, std::nextafter(PIx2, 0.0f)); //[0, 2pi) -> Find the next value from 2Pi toward 0
																									// Distribute values over the sphere
		float z = negone_one(rng);
		float theta = zero_2Pi(rng);

		float circ_rad = sqrt(1 - z*z); //Radius of the circle defined by the intersection of the plane offset u along z axis.
		return{ circ_rad * cos(theta), circ_rad * sin(theta), z };
	}
	/* Generate a set of distributed values in the range for the array using the randomInt(min, max) function.
	*/
	void RandomGenerator::randomSetofInt(int* arr, unsigned int size, int min, int max) const
	{
		for (unsigned int i = 0; i < size; i++)
			arr[i] = randomInt(min, max);
	}

	/* Generate an array with a set of distributed values in the range using the randomInt(min, max) function.
	*/
	std::unique_ptr<int> RandomGenerator::randomSetofInt(unsigned int size, int min, int max) const
	{
		std::unique_ptr<int> arr(new int[size]);
		randomSetofInt(arr.get(), size, min, max);
		return std::move(arr);
	}
}
