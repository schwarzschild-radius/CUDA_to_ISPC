#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include "cuda_utils.cuh"
#include "reduce_cuda.cuh"
#include "reduce_ispc.h"

template <typename T> T executeReference(std::vector<T> &v) {
    return std::accumulate(v.begin(), v.end(), 0);
}

int executeCUDA(std::vector<int> &v) {
    std::vector<int> u(v.begin(), v.end());
    int *d_a = nullptr, *d_partial_sum;
    size_t N = u.size();
    size_t nthreads = N > 1024 ? 1024 : N;
    size_t nblocks = N > 1024 ? (N - 1) / 1024 + 1 : 1;
    cudaCheck(cudaMalloc((void **)&d_a, N * sizeof(int)));
    cudaCheck(cudaMalloc((void **)&d_partial_sum, nblocks * sizeof(int)));
    cudaCheck(
        cudaMemcpy(d_a, u.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheckLaunch(reduce, nblocks, nthreads, d_a, d_partial_sum, nthreads);
    cudaDeviceSynchronize();
    int partial_sum[nblocks];
    cudaCheck(cudaMemcpy(partial_sum, d_partial_sum, sizeof(int) * nblocks, cudaMemcpyDeviceToHost));
    int sum = 0;
    for(int i = 0; i < nblocks; i++)
        sum += partial_sum[i];
    return sum;
}

int executeISPC(std::vector<int> &v) {
    std::vector<int> u(v.begin(), v.end());
    size_t N = u.size();
    int nthreads = N > 1024 ? 1024 : N;
    int nblocks = N > 1024 ? (N - 1) / 1024 + 1 : 1;
    std::cout << nblocks << ", " << nthreads << '\n';
    ispc::Dim3 grid = {nblocks, 1, 1};
    ispc::Dim3 block = {nthreads, 1, 1};
    int *partial_sum = new int[nblocks];
    ispc::reduce(grid, block, u.data(), partial_sum, nthreads);
    int sum = 0;
    for(int i = 0; i < nblocks; i++)
        sum += partial_sum[i];
    return sum;
}

int main(int argc, char *argv[]) {
    size_t N = 1 << 14;
    std::vector<int> v(N);
    std::iota(v.begin(), v.end(), 0);
    std::cout << executeReference(v) << '\n';
    std::cout << executeCUDA(v) << '\n';
    std::cout << executeISPC(v) << '\n';
    return 0;
}