CC = /usr/local/cuda-8.0//bin/nvcc
GENCODE_FLAGS = -arch=sm_30
CC_FLAGS = -c --compiler-options -Wall,-Wextra,-O3,-m64
NVCCFLAGS = -m64 

findRepeats: findRepeats.o gpuFindRepeats.o gpuScan.o
	$(CC) $(GENCODE_FLAGS) findRepeats.o gpuFindRepeats.o gpuScan.o -o findRepeats

findRepeats.o: findRepeats.cu CHECK.h gpuFindRepeats.h
	$(CC) $(CC_FLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) findRepeats.cu -o findRepeats.o

gpuFindRepeats.o: gpuFindRepeats.cu CHECK.h gpuFindRepeats.h
	$(CC) $(CC_FLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) gpuFindRepeats.cu -o gpuFindRepeats.o

gpuScan.o: gpuScan.cu CHECK.h gpuScan.h
	$(CC) $(CC_FLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) gpuScan.cu -o gpuScan.o

clean:
	rm findRepeats findRepeats.o gpuFindRepeats.o gpuScan.o
