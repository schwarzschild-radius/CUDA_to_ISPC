#include <cuda_runtime.h>

extern __global__ void atomic_cuda(int *d_bins, const int *d_in, const int BIN_COUNT);