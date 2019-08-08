#define ISPC_GRID_START                                                        \
    Dim3 blockIdx, threadIdx;                                                  \
    for (blockIdx.z = 0; blockIdx.z < gridDim.z; blockIdx.z++) {               \
        for (blockIdx.y = 0; blockIdx.y < gridDim.y; blockIdx.y++) {           \
            for (blockIdx.x = 0; blockIdx.x < gridDim.x; blockIdx.x++) {

#define ISPC_BLOCK_START                                                       \
    for (threadIdx.z = 0; threadIdx.z < blockDim.z; threadIdx.z++) {           \
        for (threadIdx.y = 0; threadIdx.y < blockDim.y; threadIdx.y++) {       \
            for (threadIdx.x = programIndex; threadIdx.x < blockDim.x;         \
                 threadIdx.x += programCount) {

#define ISPC_GRID_END                                                          \
    }                                                                          \
    }                                                                          \
    }

#define ISPC_BLOCK_END                                                         \
    }                                                                          \
    }                                                                          \
    }

#define ISPC_START                                                             \
    ISPC_GRID_START                                                            \
    ISPC_BLOCK_START

#define ISPC_END                                                               \
    ISPC_GRID_END                                                              \
    ISPC_BLOCK_END

#define SYNCTHREADS()                                                          \
    ISPC_BLOCK_END                                                             \
    ISPC_BLOCK_START

#define ISPC_KERNEL(function, ...)                                             \
    export void function(                                                      \
        const uniform Dim3 &gridDim, const uniform Dim3 &blockDim,             \
        const uniform size_t &shared_memory_size, __VA_ARGS__)

#define ISPC_DEVICE_FUNCTION(rety, function, ...)                              \
    rety function(const uniform Dim3 &gridDim, const uniform Dim3 &blockDim,   \
                  const Dim3 &blockIdx, const Dim3 &threadIdx, __VA_ARGS__)

#define ISPC_DEVICE_CALL(function, ...)                                        \
    function(gridDim, blockDim, blockIdx, threadIdx, __VA_ARGS__)

#define NS(x, y) x##_##y
#define NS3(x, y, z) x##_##y##_##z
#define ENUM(x, y) const int x##_##y

struct Dim3 {
    int x, y, z;
};

#define k_2powneg32 2.3283064E-10F
#define n_directions 32

unsigned int __ffs(unsigned int i) {
    return ((i != 0) ? count_trailing_zeros(i) + 1 : 0);
}

ISPC_KERNEL(sobol, uniform unsigned int n_vectors,
            uniform unsigned int n_dimensions,
            uniform unsigned int directions[], uniform float output[]) {
    uniform int stride = gridDim.x * blockDim.x;
    ISPC_GRID_START
    uniform unsigned int v[n_directions];
    unsigned int *d_directions = directions + n_directions * blockIdx.y;
    float *d_output = output + n_vectors * blockIdx.y;
    ISPC_BLOCK_START
    if (threadIdx.x < n_directions) {
        v[threadIdx.x] = d_directions[threadIdx.x];
    }
    SYNCTHREADS()
    int i0 = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int g = i0 ^ (i0 >> 1);
    unsigned int X = 0;
    unsigned int mask;

    for (unsigned int k = 0; k < __ffs(stride) - 1; k++) {
        mask = -(g & 1);
        X ^= mask & v[k];
        g = g >> 1;
    }

    if (i0 < n_vectors) {
        d_output[i0] = (float)X * k_2powneg32;
    }

    unsigned int v_log2stridem1 = v[__ffs(stride) - 2];
    unsigned int v_stridemask = stride - 1;

    for (unsigned int i = i0 + stride; i < n_vectors; i += stride) {
        X ^= v_log2stridem1 ^ v[__ffs(~((i - stride) | v_stridemask)) - 1];
        d_output[i] = (float)X * k_2powneg32;
    }
    ISPC_BLOCK_END
    ISPC_GRID_END
}
