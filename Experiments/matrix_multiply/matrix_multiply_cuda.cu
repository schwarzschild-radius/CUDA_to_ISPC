#include "matrix_multiply_cuda.cuh"
#include <stdio.h>

#define BLOCK_SIZE 2

__global__ void matrixMulCUDA(int *C, int *A, int *B, int wA, int wB,
                              int block_size) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = wA * BLOCK_SIZE * by;

    int aEnd = aBegin + wA - 1;

    int aStep = BLOCK_SIZE;

    int bBegin = BLOCK_SIZE * bx;

    int bStep = BLOCK_SIZE * wB;

    float Csub = 0;

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {

        __shared__ int As[BLOCK_SIZE][BLOCK_SIZE];

        __shared__ int Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        __syncthreads();

#pragma unroll

    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[ty][k] * Bs[k][tx];
    }

    __syncthreads();
  }

  int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  C[c + wB * ty + tx] = Csub;
}
