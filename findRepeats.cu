#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "CHECK.h"
#include "gpuFindRepeats.h"

//prototypes for functions in this file
void initOnes(int * array, int length);
void initRandom(int * array, int length);
void compare(int * h_output, int * d_output, int length);
float cpuFindRepeats(int * input, int * repeats, int length,
                     int * numRepeats);

#define NUMTESTS 5
typedef struct
{
    int length;        //number of data elements
    float speedupGoal; //minimum speedup that you should aim for
    const char * type; //how array is initialized
} testType;

testType tests[NUMTESTS] = {{1 << 18, 1.5, "random"},
                            {1 << 18, 4.0, "random"},
                            {1 << 21, 4.5, "random"},
                            {1 << 24, 5.0, "random"},
                            {1 << 28, 5.5, "random"}};

/*
   driver for the find repeats program.  
   The main calls the functions to perform the tests
   specified in the tests array.
*/
int main()
{
    int i;
    float cpuTime;
    float gpuTime;
    float speedup;
    printf("%10s\t%8s\t%8s\t%8s\t%8s\t%8s\n", 
           "Length", "Data", "CPU ms", "GPU ms", "Speedup", "Goal");
    for (i = 0; i < NUMTESTS; i++)
    {
        int * input = (int *) malloc(sizeof(int) * tests[i].length);
        int * repeats = (int *) malloc(sizeof(int) * tests[i].length);
        int * output = (int *) malloc(sizeof(int) * tests[i].length);
        int * d_output = (int *) malloc(sizeof(int) * tests[i].length);
        int * d_repeats = (int *) malloc(sizeof(int) * tests[i].length);
        int numRepeats, d_numRepeats;

        //initialize the array to all 1s 
        //or random small numbers
        if (strcmp(tests[i].type, "ones") == 0)
            initOnes(input, tests[i].length);
        else
            initRandom(input, tests[i].length);

        //for convenience, set the output to the input and then
        //the find repeat routines can just operate on the output
        memcpy(output, input, sizeof(int) * tests[i].length);
        memcpy(d_output, input, sizeof(int) * tests[i].length);

        //perform the find repeats using the CPU
        cpuTime = cpuFindRepeats(output, repeats, tests[i].length, &numRepeats);       
        //perform the find repeats using the GPU
        gpuTime = gpuFindRepeats(d_output, d_repeats, tests[i].length, &d_numRepeats);       
        speedup = cpuTime / gpuTime;
  
        if (numRepeats != d_numRepeats)
        {
           printf("Number of repeats differ. host: %d, gpu: %d\n", 
                   numRepeats, d_numRepeats); 
        }

        //make sure the gpuScan produced the correct results
        compare(repeats, d_repeats, numRepeats);

        //print the output
        printf("%10d\t%8s\t%8.4f\t%8.4f\t%8.4f\t%8.1f\n", 
               tests[i].length, tests[i].type, cpuTime, gpuTime, 
               speedup, tests[i].speedupGoal);

        //free the dynamically allocated data
        free(input);
        free(output);
        free(repeats);
        free(d_output); 
        free(d_repeats);
    }
}    

/*
   cpuFindRepeats
   Performs the find repeats on an array of integers
   with length elements.  The find repeats sets the
   repeats array to the indices of the first of
   a pair of repeats in the input array.  For example,
   if the input array contains {1,2,2,7,4,3,3,3,6}
   then the repeats array is set to {1,5,6}.
   Note that the 1 is the index of the first 2 in
   in the input array, the 5 is the index of
   the first 3 in the input array, and the 6
   is the input of the second 3 in the input array.
@param - input: array of length integers  
@param - repeats: array to hold the indices of repeats
@param - length: number of elements in input array
@param - numRepeats: point to a variable that is set
                     to the number of repeats
@returns time in msec it takes to perform repeats
*/
float cpuFindRepeats(int * input, int * repeats, int length,
                     int * numRepeats)
{
    cudaEvent_t start_cpu, stop_cpu;
    int count = 0, idx = 0;
    float cpuMsecTime = -1;

    //time the scan
    CHECK(cudaEventCreate(&start_cpu));
    CHECK(cudaEventCreate(&stop_cpu));
    CHECK(cudaEventRecord(start_cpu));

    while(idx < length - 1){ 
        if(input[idx] == input[idx + 1]){
            repeats[count] = idx;
            count++;
        }   
        idx++;
    }   
    (*numRepeats) = count;
    CHECK(cudaEventRecord(stop_cpu));
    CHECK(cudaEventSynchronize(stop_cpu));
    CHECK(cudaEventElapsedTime(&cpuMsecTime, start_cpu, stop_cpu));
    return cpuMsecTime;
}

/* 
   compare
   Compares two arrays of integers with length
   elements to see if the elements are equal.
   If the arrays differ, outputs an error
   message and exits the program.
*/
void compare(int * h_output, int * d_output, int length)
{
   for (int i = 0; i < length; i++)
   {
      if (h_output[i] != d_output[i])
      {
         printf("Compare failed: h_output[%d] = %d, d_output[%d] = %d\n", 
                i, h_output[i], i, d_output[i]);
         exit(1);
      }
   }
}

/* 
   initRandom
   initializes an array of integers of size
   length to ones
*/
void initOnes(int * array, int length)
{
   int i;
   for (i = 0; i < length; i++)
      array[i] = 1;
}

/* 
   initRandom
   initializes an array of integers of size
   length to random values between 0 and 4,
   inclusive
*/
void initRandom(int * array, int length)
{
   int i;
   for (i = 0; i < length; i++)
      array[i] = rand() % 5;
}

