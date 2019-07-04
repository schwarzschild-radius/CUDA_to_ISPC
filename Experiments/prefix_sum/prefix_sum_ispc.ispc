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

#define SYNCTHREADS()                                                          \
    ISPC_END                                                                   \
    ISPC_BLOCK

struct Dim3 {
    int x, y, z;
};

export void prefix_sum_ispc(const uniform Dim3 &gridDim,
                            const uniform Dim3 &blockDim, uniform int a[],
                            uniform size_t N) {
    Dim3 blockIdx, threadIdx;

    ISPC_GRID
    int tid = threadIdx.x;
    int i = 0;
    for (i = 2; i <= N; i *= 2) {
        ISPC_BLOCK
        tid = threadIdx.x;
        if (((i - tid % i) == 1) && tid != 0) {
            a[tid] = a[tid] + a[tid - i / 2];
        }
        ISPC_BLOCK_END
    }

    ISPC_BLOCK
    tid = threadIdx.x;
    if (tid == N - 1) {
        a[tid] = 0;
    }
    ISPC_BLOCK_END

    for (; i > 1; i /= 2) {
        ISPC_BLOCK
        tid = threadIdx.x;
        if (((i - tid % i) == 1) && tid != 0) {
            int temp = a[tid - i / 2];
            a[tid - i / 2] = a[tid];
            a[tid] = a[tid] + temp;
        }
        ISPC_BLOCK_END
    }
    ISPC_GRID_END
}