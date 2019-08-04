#include <cuda_runtime.h>

__global__ void BlackScholesGPU(float *d_CallResult, float *d_PutResult,
                                float *d_StockPrice, float *d_OptionStrike,
                                float *d_OptionYears, float Riskfree,
                                float Volatility, int optN);