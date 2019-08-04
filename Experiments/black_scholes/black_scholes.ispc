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

ISPC_DEVICE_FUNCTION(float, cndGPU, float d) {
    const float A1 = 0.31938153f;
    const float A2 = -0.356563782f;
    const float A3 = 1.781477937f;
    const float A4 = -1.821255978f;
    const float A5 = 1.330274429f;
    const float RSQRT2PI = 0.39894228040143267793994605993438f;

    float K = 1.0f / (1.0f + 0.2316419f * abs(d));

    float cnd = RSQRT2PI * exp(-0.5f * d * d) *
                (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0f - cnd;

    return cnd;
}

ISPC_DEVICE_FUNCTION(void, BlackScholesBodyGPU, float *CallResult, float *PutResult,
                     float S, float X, float T, float R, float V) {
    float sqrtT, expRT;
    float d1, d2, CNDD1, CNDD2;

    sqrtT = sqrt(T);
    d1 = (log(S / X) + (R + 0.5f * V * V) * T) / (V * sqrtT);
    d2 = d1 - V * sqrtT;

    CNDD1 = ISPC_DEVICE_CALL(cndGPU, d1);
    CNDD2 = ISPC_DEVICE_CALL(cndGPU, d2);
    expRT = exp(-R * T);
    *CallResult = S * CNDD1 - X * expRT * CNDD2;
    *PutResult = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
}

ISPC_KERNEL(BlackScholesGPU, uniform float d_CallResult[],
            uniform float d_PutResult[], uniform float d_StockPrice[],
            uniform float d_OptionStrike[], uniform float d_OptionYears[],
            uniform float Riskfree, uniform float Volatility,
            uniform int optN) {
    ISPC_GRID_START
    ISPC_BLOCK_START
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    // Total number of threads in execution grid
    const int THREAD_N = blockDim.x * gridDim.x;

    // No matter how small is execution grid or how large OptN is,
    // exactly OptN indices will be processed with perfect memory coalescing
    for (int opt = tid; opt < optN; opt += THREAD_N)
        ISPC_DEVICE_CALL(BlackScholesBodyGPU, &d_CallResult[opt], &d_PutResult[opt],
                            d_StockPrice[opt], d_OptionStrike[opt],
                            d_OptionYears[opt], Riskfree, Volatility);
    ISPC_BLOCK_END
    ISPC_GRID_END
}
