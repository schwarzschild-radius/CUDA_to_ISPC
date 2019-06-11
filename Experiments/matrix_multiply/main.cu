#include "matrix_multiply_cuda.cuh"
#include "matrix_multiply_ispc.h"
#include <cuda_utils.cuh>
#include <iostream>
#include <numeric>
#include <vector>

#define uint size_t

template <typename T> void fill_matrix(T *A, uint m, uint n) {
    std::iota(A, A + m * n, 0);
}

template <typename T>
void executeCUDA(T *cuda_C, T *A, T *B, unsigned int N,
                 unsigned int block_size) {
    int *d_A = nullptr, *d_B = nullptr, *d_cuda_C = nullptr;
    uint nbytes = sizeof(T) * N * N;

    cudaCheck(cudaMalloc((void **)&d_A, nbytes));
    cudaCheck(cudaMalloc((void **)&d_B, nbytes));
    cudaCheck(cudaMalloc((void **)&d_cuda_C, nbytes));

    cudaCheck(cudaMemcpy(d_A, A, nbytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_B, B, nbytes, cudaMemcpyHostToDevice));

    dim3 block = {block_size, block_size, 1};
    dim3 grid = {N / block_size, N / block_size, 1};

    cudaCheckLaunch(matrixMulCUDA, grid, block, d_cuda_C, d_A, d_B, N, N)

    cudaCheck(cudaMemcpy(cuda_C, d_cuda_C, nbytes, cudaMemcpyDeviceToHost));
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_cuda_C);
}

template <typename T>
void executeISPC(T *ispc_C, T *A, T *B, uint16_t N, uint16_t block_size) {
    ispc::blockDim block{1, block_size, block_size};
      ispc::gridDim grid{1, static_cast<uint16_t>(N / block_size),
                         static_cast<uint16_t>(N / block_size)};
    // ispc::gridDim grid{1, 1, 1};
    ispc::matrixMulISPC(grid, block, ispc_C, A, B, N, N, block_size);
}

template <typename T>
void executeReference(T *C, T *A, T *B, size_t m, size_t n, size_t o) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            for (size_t k = 0; k < o; k++) {
                C[i * n + j] += A[i * o + k] * B[k * n + j];
            }
        }
    }
}

template <typename T> void compareResults(T *A, T *B, T *C, uint N) {
    for (uint i = 0; i < N; i++) {
        if (A[i] != B[i] && B[i] != C[i]) {
            std::cerr << "error at i: " << A[i] << ", " << B[i] << ", " << C[i]
                      << ", " << i << "\n";
        }
    }
}

int main(int argc, char *argv[]) {
    size_t N = 8;
    if (argc == 2) {
        N = strtoul(argv[argc - 1], nullptr, 10);
    }
    std::vector<int> A(N * N, 1), B(N * N, 1), C(N * N, 0), cuda_C(N * N, 0),
        ispc_C(N * N, 0), ref_C(N * N, 0);
    fill_matrix(A.data(), N, N);
    fill_matrix(B.data(), N, N);
    size_t block_size = 4;
    executeCUDA(cuda_C.data(), A.data(), B.data(), N, block_size);
    executeISPC(ispc_C.data(), A.data(), B.data(), N, block_size);
    executeReference(ref_C.data(), A.data(), B.data(), N, N, N);
    compareResults(cuda_C.data(), ispc_C.data(), ref_C.data(), N * N);

    /* for(size_t i = 0; i < N; i++){
        for(size_t j = 0; j < N; j++){
            std::cout << ispc_C[i * N + j] << " ";
        }
        std::cout << "\n";
    } */

    return 0;
}