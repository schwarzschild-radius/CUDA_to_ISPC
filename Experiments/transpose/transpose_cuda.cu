#include <stdio.h>
#include "transpose_ispc.h"
#include "transpose_cuda.cuh"

__global__ void 
transpose_parallel_per_element(int in[], int out[], size_t N, size_t K)
{
	int i = blockIdx.x * K + threadIdx.x;
	int j = blockIdx.y * K + threadIdx.y;

	out[j + i*N] = in[i + j*N]; // out(j,i) = in(i,j)
}