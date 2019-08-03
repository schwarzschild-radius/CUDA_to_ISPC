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

#define TIME_STEPS 16
#define CACHE_DELTA (2 * TIME_STEPS)
#define CACHE_SIZE (256)
#define CACHE_STEP (CACHE_SIZE - CACHE_DELTA)

#define NUM_STEPS 2048

#define MAX_OPTIONS 1024

#if NUM_STEPS % CACHE_DELTA
#error Bad constants
#endif

struct __TOptionData {
    double S;
    double X;
    double T, R, V;
    double vDt;
    double puByDf;
    double pdByDf;
};

ISPC_DEVICE_FUNCTION(double, expiryCallValue, double S, double X, double vDt,
                     int i) {
    double d = S * exp(vDt * (2.0 * i - NUM_STEPS)) - X;
    return (d > 0) ? d : 0;
}

ISPC_KERNEL(binomialOptionsKernel, uniform __TOptionData d_OptionData[],
            uniform double d_CallBuffer[], uniform float d_CallValue[]) {
    uniform double callA[CACHE_SIZE + 1];
    uniform double callB[CACHE_SIZE + 1];

    ISPC_GRID_START

    double *const d_Call = &d_CallBuffer[blockIdx.x * (NUM_STEPS + 16)];
    const double S = d_OptionData[blockIdx.x].S;
    const double X = d_OptionData[blockIdx.x].X;
    const double vDt = d_OptionData[blockIdx.x].vDt;
    const double puByDf = d_OptionData[blockIdx.x].puByDf;
    const double pdByDf = d_OptionData[blockIdx.x].pdByDf;

    ISPC_BLOCK_START
    const int tid = threadIdx.x;
    for (int i = tid; i <= NUM_STEPS; i += CACHE_SIZE) {
        d_Call[i] = ISPC_DEVICE_CALL(expiryCallValue, S, X, vDt, i);
    }
    ISPC_BLOCK_END

    for (int i = NUM_STEPS; i > 0; i -= CACHE_DELTA) {
        for (int c_base = 0; c_base < i; c_base += CACHE_STEP) {
            int c_start = min(CACHE_SIZE - 1, i - c_base);
            int c_end = c_start - CACHE_DELTA;
            ISPC_BLOCK_START
            const int tid = threadIdx.x;
            if (tid <= c_start) {
                callA[tid] = d_Call[c_base + tid];
            }
            ISPC_BLOCK_END

            for (int k = c_start - 1; k >= c_end;) {
                ISPC_BLOCK_START
                const int tid = threadIdx.x;
                callB[tid] = puByDf * callA[tid + 1] + pdByDf * callA[tid];
                ISPC_BLOCK_END
                k--;
                ISPC_BLOCK_START
                const int tid = threadIdx.x;
                callA[tid] = puByDf * callB[tid + 1] + pdByDf * callB[tid];
                ISPC_BLOCK_END
                k--;
            }

            ISPC_BLOCK_START
            const int tid = threadIdx.x;
            if (tid <= c_end) {
                d_Call[c_base + tid] = callA[tid];
            }
            ISPC_BLOCK_END
        }
    }

    ISPC_BLOCK_START
    const int tid = threadIdx.x;
    if (threadIdx.x == 0) {
        d_CallValue[blockIdx.x] = (float)callA[0];
    }
    ISPC_BLOCK_END

    ISPC_GRID_END
}