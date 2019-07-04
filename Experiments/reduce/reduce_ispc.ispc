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
    ISPC_END                                                                   \
    ISPC_BLOCK

struct Dim3 {
    int x, y, z;
};

export void reduce(uniform Dim3 &gridDim, uniform Dim3 &blockDim,
                   uniform int arr[], uniform int partial_sum[],
                   uniform unsigned int n) {
    Dim3 blockIdx, threadIdx;
ISPC_GRID
    for (int s = n / 2; s > 0; s >>= 1) {
ISPC_BLOCK
        size_t gid = threadIdx.x + blockDim.x * blockIdx.x;
        if (threadIdx.x < s) {
            arr[gid] += arr[gid + s];
        }
ISPC_BLOCK_END
    }
    partial_sum[blockIdx.x] = arr[blockIdx.x * blockDim.x];
ISPC_GRID_END
}