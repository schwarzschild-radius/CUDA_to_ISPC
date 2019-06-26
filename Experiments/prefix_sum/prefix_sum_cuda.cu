#include "prefix_sum_cuda.cuh"

__global__ void prefix_sum_cuda(int *a, size_t N) {
    int tid = threadIdx.x;
    int i = 0;
    for (i = 2; i <= N; i *= 2) {
        if (((i - tid % i) == 1) && tid != 0) {
            a[tid] = a[tid] + a[tid - i / 2];
        }
        __syncthreads();
    }
    if (tid == N - 1) {
        a[tid] = 0;
    }
    for (; i > 1; i /= 2) {
        if (((i - tid % i) == 1) && tid != 0) {
            int temp = a[tid - i / 2];
            a[tid - i / 2] = a[tid];
            a[tid] = a[tid] + temp;
        }
        __syncthreads();
    }
    // a[tid] += d_in[tid];
}