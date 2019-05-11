#include "transpose_cuda.cuh"
#include "transpose_ispc.h"
#include <cuda_utils.cuh>
#include <iostream>
#include <numeric>

#define uint size_t

template <typename T> void fill_matrix(T *A, uint m, uint n) {
  std::iota(A, A + m * n, 0);
}

template <typename T> void executeCUDA(T *A, T *cuda_A, size_t N) {
  int *d_A = nullptr, *d_out_A = nullptr;
  uint nbytes = sizeof(T) * N * N;

  cudaCheck(cudaMalloc((void **)&d_A, nbytes));
  cudaCheck(cudaMalloc((void **)&d_out_A, nbytes));

  cudaCheck(cudaMemcpy(d_A, A, nbytes, cudaMemcpyHostToDevice));

  uint threads = N > 1024 ? 1024 : N;
  uint blocks = N > 1024 ? N / 1024 + 1 : 1;

  transpose_parallel_per_element<<<blocks, threads>>>(d_A, d_out_A, N, 32);

  cudaCheck(cudaMemcpy(cuda_A, d_out_A, nbytes, cudaMemcpyDeviceToHost));
  cudaFree(d_A);
  cudaFree(d_out_A);
}

template <typename T> void executeISPC(T *A, T *ispc_A, uint N) {
  uint16_t threads = N > 1024 ? 1024 : N;
  uint16_t blocks = N > 1024 ? N / 1024 + 1 : 1;
  ispc::gridDim grid{blocks, 1, 1};
  ispc::blockDim block{threads, 1, 1};
  ispc::transpose_parallel_per_element_ispc(grid, block, A, ispc_A, N, 32);
}

template <typename T> void compareResults(T *A, T *B, uint N) {
  for (uint i = 0; i < N; i++) {
    for (uint j = 0; j < N; j++) {
      if (A[i * N + j] != B[i * N + j]) {
        std::cerr << "error at (i, j)=(" << i << ", " << j << ")\n";
      }
    }
  }
}

int main(int argc, char *argv[]) {
  size_t N = 32;
  if (argc == 2) {
    N = strtoul(argv[argc - 1], nullptr, 10);
  }
  int *A = new int[N * N], *cuda_A = new int[N * N], *ispc_A = new int[N * N];

  executeCUDA(A, cuda_A, N);
  executeISPC(A, ispc_A, N);
  compareResults(cuda_A, ispc_A, N);

  return 0;
}