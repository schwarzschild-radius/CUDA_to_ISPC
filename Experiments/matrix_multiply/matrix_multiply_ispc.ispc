struct gridDim {
    uniform unsigned int16 gridDim_z;
    uniform unsigned int16 gridDim_y;
    uniform unsigned int16 gridDim_x;
};

struct blockDim {
    uniform unsigned int16 blockDim_z;
    uniform unsigned int16 blockDim_y;
    uniform unsigned int16 blockDim_x;
};

export void matrixMulISPC(uniform gridDim &grid_dim,
                          uniform blockDim &block_dim, uniform int C[],
                          uniform int A[], uniform int B[], uniform int wA,
                          uniform int wB, uniform int BLOCK_SIZE) {
    for (size_t blockIdx_z = 0; blockIdx_z < grid_dim.gridDim_z; blockIdx_z++) {
        for (size_t blockIdx_y = 0; blockIdx_y < grid_dim.gridDim_y;
             blockIdx_y++) {
            for (size_t blockIdx_x = 0; blockIdx_x < grid_dim.gridDim_x;
                 blockIdx_x++) {
                for (size_t threadIdx_z = 0; threadIdx_z < block_dim.blockDim_z;
                     threadIdx_z++) {
                    for (size_t threadIdx_y = 0;
                         threadIdx_y < block_dim.blockDim_y; threadIdx_y++) {
                        for (size_t threadIdx_x = programIndex;
                             threadIdx_x < block_dim.blockDim_x;
                             threadIdx_x += programCount) {
                            unsigned int bx = blockIdx_x;
                            unsigned int by = blockIdx_y;
                            varying unsigned int tx = threadIdx_x;
                            varying unsigned int ty = threadIdx_y;

                            varying unsigned int aBegin = wA * BLOCK_SIZE * by;

                            varying unsigned int aEnd = aBegin + wA - 1;

                            varying unsigned int aStep = BLOCK_SIZE;

                            varying unsigned int bBegin = BLOCK_SIZE * bx;

                            varying unsigned int bStep = BLOCK_SIZE * wB;

                            varying float Csub = 0;

                            for (int a = aBegin, b = bBegin; a <= aEnd;
                                 a += aStep, b += bStep) {
                                uniform int *varying As = &A[a + ty * wA + tx];
                                uniform int *varying Bs = &B[b + ty * wB + tx];
                                for (int k = 0; k < BLOCK_SIZE; ++k) {
                                    Csub += A[a + ty * wA + k] *
                                            B[b + k * wB + tx];
                                }
                            }
                            varying int c =
                                wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
                            C[c + wB * ty + tx] = Csub;
                        }
                    }
                }
            }
        }
    }
}