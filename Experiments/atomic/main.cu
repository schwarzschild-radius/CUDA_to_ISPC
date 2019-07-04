#include "atomic_cuda.cuh"
#include "atomic_ispc.h"
#include <iostream>

int log2(int i) {
    int r = 0;
    while (i >>= 1)
        r++;
    return r;
}

int bit_reverse(int w, int bits) {
    int r = 0;
    for (int i = 0; i < bits; i++) {
        int bit = (w & (1 << i)) >> i;
        r |= bit << (bits - i - 1);
    }
    return r;
}

template <typename T>
void executeCUDA(size_t ARRAY_SIZE, size_t BIN_COUNT, const T *h_in,
                 const T *h_bins, T *cuda_bins) {
    T *d_in;
    T *d_bins;

    size_t BIN_BYTES = BIN_COUNT * sizeof(T);
    size_t ARRAY_BYTES = ARRAY_SIZE * sizeof(T);

    cudaMalloc((void **)&d_in, ARRAY_BYTES);
    cudaMalloc((void **)&d_bins, BIN_BYTES);

    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bins, h_bins, BIN_BYTES, cudaMemcpyHostToDevice);

    atomic_cuda<<<ARRAY_SIZE / 64, 64>>>(d_bins, d_in, BIN_COUNT);

    cudaMemcpy(cuda_bins, d_bins, BIN_BYTES, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_bins);
}

template <typename T>
void executeISPC(size_t ARRAY_SIZE, size_t BIN_COUNT, const T *h_in,
                 const int *h_bins, int *ispc_bins) {
                     ispc::Dim3 grid_dim{static_cast<uint32_t>(ARRAY_SIZE / 64), 1, 1};
                     ispc::Dim3 block_dim{64, 1, 1};
                     ispc::atomic_ispc(grid_dim, block_dim, ispc_bins, h_in, BIN_COUNT);
}

void checkResults(int *cuda, int *ispc, size_t N) {
    for(size_t i = 0; i < N; i++){
        if(cuda[i] != ispc[i]){
            std::cerr << "Mismatch at index : " << i << " " << cuda[i] << ", " << ispc[i] << "\n";
        }
    }
}

int main(int argc, char **argv) {
    const int ARRAY_SIZE = 65536;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);
    const int BIN_COUNT = 16;
    const int BIN_BYTES = BIN_COUNT * sizeof(int);

    // generate the input array on the host
    int h_in[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_in[i] = bit_reverse(i, log2(ARRAY_SIZE));
    }
    int h_bins[BIN_COUNT], cuda_bins[BIN_COUNT], ispc_bins[BIN_COUNT];
    for (int i = 0; i < BIN_COUNT; i++) {
        h_bins[i] = 0;
        ispc_bins[i] = 0;
    }
    executeCUDA(ARRAY_SIZE, BIN_COUNT, h_in, h_bins, cuda_bins);
    executeISPC(ARRAY_SIZE, BIN_COUNT, h_in, h_bins, ispc_bins);
    checkResults(cuda_bins, ispc_bins, BIN_COUNT);

    return 0;
}