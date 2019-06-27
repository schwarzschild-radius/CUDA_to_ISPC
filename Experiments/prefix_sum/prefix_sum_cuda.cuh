#include <cuda_runtime.h>
#include <cstdio>

__global__ void prefix_sum_cuda(int *a, size_t N);