#include <cstdio>
#include <cuda_runtime.h>

extern __global__ void
matrixMulCUDA(int *C, int *A, int *B, int wA, int wB);