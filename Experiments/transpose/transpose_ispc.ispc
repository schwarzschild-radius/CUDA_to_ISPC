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

task void transpose_parallel_per_element_ispc_task_(
    const uniform Dim3 &gridDim, const uniform Dim3 &blockDim, uniform int a[],
    uniform int b[], uniform unsigned int32 N, uniform unsigned int32 K) {

    Dim3 blockIdx, threadIdx;
    blockIdx.z = taskIndex2;
    blockIdx.y = taskIndex1;
    blockIdx.x = taskIndex0;
    ISPC_BLOCK
    int i = blockIdx.x * K + threadIdx.x;
    int j = blockIdx.y * K + threadIdx.y;
    b[j + i * N] = a[i + j * N];
    ISPC_BLOCK_END
}

export void transpose_parallel_per_element_ispc_task(
    const uniform Dim3 &gridDim, const uniform Dim3 &blockDim, uniform int a[],
    uniform int b[], uniform unsigned int32 N, uniform unsigned int32 K) {
    launch[gridDim.x, gridDim.y,
           gridDim.z] transpose_parallel_per_element_ispc_task_(gridDim,
                                                                blockDim, a, b,
                                                                N, K);
}

export void transpose_parallel_per_element_ispc(
    const uniform Dim3 &gridDim, const uniform Dim3 &blockDim, uniform int a[],
    uniform int b[], uniform unsigned int32 N, uniform unsigned int32 K) {
    Dim3 blockIdx, threadIdx;
    ISPC_START
    int i = blockIdx.x * K + threadIdx.x;
    int j = blockIdx.y * K + threadIdx.y;
    b[j + i * N] = a[i + j * N];
    ISPC_END
}