#define ISPC_GRID                                                              \
    for (blockIdx.z = 0; blockIdx.z < gridDim.z; blockIdx.z++) {               \
        for (blockIdx.y = 0; blockIdx.y < gridDim.y; blockIdx.y++) {           \
            for (blockIdx.x = 0; blockIdx.x < gridDim.x; blockIdx.x++) {

#define ISPC_BLOCK                                                             \
    for (threadIdx.z = 0; threadIdx.z < blockDim.z; threadIdx.z++) {           \
        for (threadIdx.y = 0; threadIdx.y < blockDim.y; threadIdx.y++) {       \
            for (threadIdx.x = programIndex; threadIdx.x < blockDim.x;         \
                 threadIdx.x += programCount) {

#define ISPC_START                                                             \
    ISPC_GRID                                                                  \
    ISPC_BLOCK

#define ISPC_GRID_END                                                          \
    }                                                                          \
    }                                                                          \
    }

#define ISPC_BLOCK_END                                                         \
    }                                                                          \
    }                                                                          \
    }

#define ISPC_END                                                               \
    ISPC_GRID_END                                                              \
    ISPC_BLOCK_END

#define SYNCTHREADS()                                                           \
    ISPC_BLOCK_END                                                                   \
    ISPC_BLOCK



struct Dim3{
    unsigned int x, y, z;
};

export void matrixMultiplyISPC(const uniform Dim3& gridDim,
                          const uniform Dim3& blockDim, uniform int C[],
                          uniform int A[], uniform int B[], uniform int wA,
                          uniform int wB, uniform int BLOCK_SIZE) {
                            Dim3 blockIdx, threadIdx;
ISPC_START
                            unsigned int bx = blockIdx.x;
                            unsigned int by = blockIdx.y;
                            varying unsigned int tx = threadIdx.x;
                            varying unsigned int ty = threadIdx.y;

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
ISPC_END
}