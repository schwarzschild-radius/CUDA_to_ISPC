#include <cuda_runtime.h>

__global__ void radix_sort_cuda(size_t n_bits, int *arr, size_t N);