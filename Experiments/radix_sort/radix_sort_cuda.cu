#include "radix_sort_cuda.cuh"

__global__ void radix_sort_cuda(size_t n_bits, int *arr, size_t N){
    __shared__ int pred_0[N], pred_1[N];
    size_t tid = threadIdx.x;
    for(size_t i = 0; i < n_bits; i++){
        // predicate
        pred_0[tid] = (arr[tid] & (1 << i) == 0);
        pred_1[tid] = (arr[tid] & (1 << i) == 1);

        // scan
        for(size_t j = 1; j <= N; j *= 2){
            if((tid & 1) && tid >= j){
                pred_0[tid] = pred_0[tid + j];
                pred_1[tid] = pred_1[tid + j];
            }
            __syncthreads(); // wait
        }
        size_t offset_1 = pred_0[N - 1];
        if((arr[tid] & (1 << i) == 0)){
            int temp = arr[tid];
            __syncthreads();
            if((arr[tid] & (1 << i) == 0)){
                arr[pred[tid]] = arr[tid];
            }else{
                arr[offset_1 + pred[tid]] = arr[tid];
            }
        }
        __syncthreads(); // wait
    }
}