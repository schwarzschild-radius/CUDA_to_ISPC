struct gridDim {
    uniform unsigned int16 gridDim_x;
    uniform unsigned int16 gridDim_y;
    uniform unsigned int16 gridDim_z;
};

struct blockDim {
    uniform unsigned int16 blockDim_x;
    uniform unsigned int16 blockDim_y;
    uniform unsigned int16 blockDim_z;
};

export void matrixMulISPC(uniform gridDim &grid_dim,
                          uniform blockDim &block_dim, uniform int As[],
                          uniform int Bs[], uniform int C[], uniform int A[],
                          uniform int B[], uniform int wA, uniform int wB,
                          uniform int BLOCK_SIZE) {
    for (size_t blockIdx_z = 0; blockIdx_z < grid_dim.gridDim_z; blockIdx_z++) {
        for (size_t blockIdx_y = 0; blockIdx_y < grid_dim.gridDim_y;
             blockIdx_y++) {
            for (size_t blockIdx_x = 0; blockIdx_x < grid_dim.gridDim_x;
                 blockIdx_x++) {
                int bx = blockIdx_x;
                int by = blockIdx_y;

                int aBegin = wA * BLOCK_SIZE * by;

                int aEnd = aBegin + wA - 1;

                int aStep = BLOCK_SIZE;

                int bBegin = BLOCK_SIZE * bx;

                int bStep = BLOCK_SIZE * wB;
                uniform int Csub[1024];
                for (size_t threadIdx_z = 0; threadIdx_z < block_dim.blockDim_z;
                     threadIdx_z++) {
                    for (size_t threadIdx_y = 0;
                         threadIdx_y < block_dim.blockDim_y; threadIdx_y++) {
                        for (size_t threadIdx_x = programIndex;
                             threadIdx_x < block_dim.blockDim_x;
                             threadIdx_x += programCount) {
                            int tx = threadIdx_x;
                            int ty = threadIdx_y;
                            Csub[ty * BLOCK_SIZE + tx] = 0;
                        }
                    }
                }

                for (int a = aBegin, b = bBegin; a <= aEnd;
                     a += aStep, b += bStep) {

                    for (size_t threadIdx_z = 0;
                         threadIdx_z < block_dim.blockDim_z; threadIdx_z++) {
                        for (size_t threadIdx_y = 0;
                             threadIdx_y < block_dim.blockDim_y;
                             threadIdx_y++) {
                            for (size_t threadIdx_x = programIndex;
                                 threadIdx_x < block_dim.blockDim_x;
                                 threadIdx_x += programCount) {

                                int tx = threadIdx_x;
                                int ty = threadIdx_y;

                                As[ty * BLOCK_SIZE + tx] = A[a + wA * ty + tx];
                                Bs[ty * BLOCK_SIZE + tx] = B[b + wB * ty + tx];
                            }
                        }
                    }
                    for (size_t threadIdx_z = 0;
                         threadIdx_z < block_dim.blockDim_z; threadIdx_z++) {
                        for (size_t threadIdx_y = 0;
                             threadIdx_y < block_dim.blockDim_y;
                             threadIdx_y++) {
                            for (size_t threadIdx_x = programIndex;
                                 threadIdx_x < block_dim.blockDim_x;
                                 threadIdx_x += programCount) {

                                int tx = threadIdx_x;
                                int ty = threadIdx_y;
                                for (int k = 0; k < BLOCK_SIZE; ++k) {
                                    Csub[ty * BLOCK_SIZE + tx] +=
                                        As[ty * BLOCK_SIZE + k] *
                                        Bs[k * BLOCK_SIZE + tx];
                                }
                            }
                        }
                    }
                }
                for (size_t threadIdx_z = 0; threadIdx_z < block_dim.blockDim_z;
                     threadIdx_z++) {
                    for (size_t threadIdx_y = 0;
                         threadIdx_y < block_dim.blockDim_y; threadIdx_y++) {
                        for (size_t threadIdx_x = programIndex;
                             threadIdx_x < block_dim.blockDim_x;
                             threadIdx_x += programCount) {

                            int tx = threadIdx_x;
                            int ty = threadIdx_y;

                            int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
                            C[c + wB * ty + tx] = Csub[ty * BLOCK_SIZE + tx];
                        }
                    }
                }
            }
        }
    }
}