#define ISPC_GRID_START                                                              \
    for (blockIdx.z = 0; blockIdx.z < gridDim.z; blockIdx.z++) {               \
        for (blockIdx.y = 0; blockIdx.y < gridDim.y; blockIdx.y++) {           \
            for (blockIdx.x = 0; blockIdx.x < gridDim.x; blockIdx.x++) {

#define ISPC_BLOCK_START                                                             \
    for (threadIdx.z = 0; threadIdx.z < blockDim.z; threadIdx.z++) {           \
        for (threadIdx.y = 0; threadIdx.y < blockDim.y; threadIdx.y++) {       \
            for (threadIdx.x = programIndex; threadIdx.x < blockDim.x;         \
                 threadIdx.x += programCount) {

#define ISPC_START                                                             \
    ISPC_GRID_START                                                                  \
    ISPC_BLOCK_START

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
    ISPC_BLOCK_START


#define BLOCK_SIZE 2

struct Dim3{
    unsigned int x, y, z;
};

export void matrixMulISPC(const uniform Dim3 &gridDim,
                          const uniform Dim3 &blockDim, uniform int C[], uniform int A[],
                          uniform int B[], uniform int wA, uniform int wB) {
    // shared memory declarations
    uniform int As[BLOCK_SIZE][BLOCK_SIZE];
    uniform int Bs[BLOCK_SIZE][BLOCK_SIZE];
    Dim3 blockIdx, threadIdx;
ISPC_GRID_START
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int aBegin = wA * BLOCK_SIZE * by;

    int aEnd = aBegin + wA - 1;

    int aStep = BLOCK_SIZE;

    int bBegin = BLOCK_SIZE * bx;

    int bStep = BLOCK_SIZE * wB;
    
    uniform int Csub[1024];
ISPC_BLOCK_START
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    Csub[ty * BLOCK_SIZE + tx] = 0;
ISPC_BLOCK_END

    for (int a = aBegin, b = bBegin; a <= aEnd;
         a += aStep, b += bStep) {
ISPC_BLOCK_START
            int tx = threadIdx.x;
            int ty = threadIdx.y;

            As[ty][tx] = A[a + wA * ty + tx];
            Bs[ty][tx] = B[b + wB * ty + tx];
SYNCTHREADS()
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                Csub[ty * BLOCK_SIZE + tx] +=
                    As[ty][ k] *
                    Bs[k][tx];
            }
ISPC_BLOCK_END
    }
ISPC_BLOCK_START
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
            C[c + wB * ty + tx] = Csub[ty * BLOCK_SIZE + tx];
ISPC_BLOCK_END
ISPC_GRID_END
}
