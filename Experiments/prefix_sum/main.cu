#include "prefix_sum_cuda.cuh"
#include "prefix_sum_ispc.h"
#include <cuda_utils.cuh>
#include <iostream>
#include <numeric>
#include <vector>

#define uint size_t

template <typename T> void fill_vector(T *A, uint n) { std::iota(A, A + n, 1); }

template <typename T> void executeCUDA(T *a, size_t N) {
    T *d_a = nullptr;
    uint nbytes = sizeof(T) * N;

    cudaCheck(cudaMalloc((void **)&d_a, nbytes));

    cudaCheck(cudaMemcpy(d_a, a, nbytes, cudaMemcpyHostToDevice));

    dim3 block = {static_cast<unsigned int>(N), 1, 1};
    dim3 grid = {1, 1, 1};

    cudaCheckLaunch(prefix_sum_cuda, grid, block, d_a, N);

    cudaCheck(cudaMemcpy(a, d_a, nbytes, cudaMemcpyDeviceToHost));
    cudaFree(d_a);
}

template <typename T> void executeISPC(T *a, size_t N) {
    ispc::Dim3 grid{1, 1, 1};
    ispc::Dim3 block{static_cast<int>(N), 1, 1};
    ispc::prefix_sum_ispc(grid, block, a, N);
}

template <typename T> void executeReference(T *a, size_t N) {
    T *temp = new T[N];
    temp[0] = 0;
    temp[1] = a[0];
    for (int i = 2; i < N; i++) {
        temp[i] = a[i - 1] + temp[i - 1];
    }
    std::copy(temp, temp + N, a);
}

template <typename Ref, typename... T>
void compareResults(int N, std::vector<Ref> &ref, std::vector<T> &... rest) {
    for (int i = 0;  i < N; i++) {
        if (((ref[i] != rest[i]) || ...)) {
            std::cerr << "error at " << i << " " << ref[i] << " ";
            (std::cerr << ... << rest[i]) << '\n';
        }
    }
}

template <typename T> void print(std::vector<T> v) {
    for (const auto &i : v) {
        std::cout << i << ' ';
    }
    std::cout << '\n';
}

int main(int argc, char *argv[]) {
    size_t N = 1024;
    if (argc == 2) {
        N = strtoul(argv[argc - 1], nullptr, 10);
    }
    std::vector<int> ref(N), cuda(N), ispc(N);
    fill_vector(ref.data(), N);
    fill_vector(cuda.data(), N);
    fill_vector(ispc.data(), N);

    executeCUDA(cuda.data(), N);
    executeISPC(ispc.data(), N);
    executeReference(ref.data(), N);
    compareResults(N, ref, cuda, ispc);
    return 0;
}