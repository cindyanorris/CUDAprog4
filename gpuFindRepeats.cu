#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "CHECK.h"
#include "gpuFindRepeats.h"
#include "gpuScan.h"  //exclusive scan prototype       


/*
 * findRepeats
 * You need to write this code.
 * At a minimum, its parameters will be an
 * input array, the length of that array,
 * and an array to be initialized to the
 * repeat indices.
 *
 * You'll need to use your exclusive scan
 * as well as other kernels.
 * 
 * It should return the number of repeats.
*/
int findRepeats(.................)
{

}

/* 
 * gpuFindRepeats
 *
 * Uses the GPU to perform a find repeats on the input array.
 * The repeats array is initialized to the indices within
 * the input array that correspond to the first of a pair
 * of repeats.  For example, if the input array contains: 
 * {1,2,2,7,4,3,3,3,6}
 * then the repeats array is set to:
 * {1,5,6}
 * Note that the 1 is the index of the first 2 in
 * in the input array, the 5 is the index of
 * the first 3 in the input array, and the 6
 * is the input of the second 3 in the input array.
 * @param - input: array of length integers  
 * @param - repeats: array to hold the indices of repeats
 * @param - length: number of elements in input array
 * @param - repeatLength: pointer to a variable that is set
 *                   to the number of repeats 
 * @returns the amount of time it takes to perform the
 *          find repeats 
*/
float gpuFindRepeats(int * input, int * repeats, int length, int * repeatLength)
{
    int * d_input, * d_repeats;
    //create input array for GPU
    CHECK(cudaMalloc((void **)&d_input, sizeof(int) * length));
    CHECK(cudaMemcpy(d_input, input, length * sizeof(int), 
               cudaMemcpyHostToDevice));
    //create array to hold repeat indices
    CHECK(cudaMalloc((void **)&d_repeats, sizeof(int) * length));

    //you may want to cudaMalloc other arrays

    float gpuMsecTime = -1;
    cudaEvent_t start_cpu, stop_cpu;
    //start the timing
    CHECK(cudaEventCreate(&start_cpu));
    CHECK(cudaEventCreate(&stop_cpu));
    CHECK(cudaEventRecord(start_cpu));

    //you need to decide upon the parameters to findRepeats
    //your find repeats needs to return the number of repeats
    //int numRepeats = findRepeats(.....);

    //stop the timing
    CHECK(cudaEventRecord(stop_cpu));
    CHECK(cudaEventSynchronize(stop_cpu));
    CHECK(cudaEventElapsedTime(&gpuMsecTime, start_cpu, stop_cpu));
   
    //copy the output of the GPU to the CPU array
    cudaMemcpy(repeats, d_repeats, numRepeats * sizeof(int),
               cudaMemcpyDeviceToHost);
    (*repeatLength) = numRepeats;

    //release the space for the GPU arrays
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_repeats));

    return gpuMsecTime;
}

