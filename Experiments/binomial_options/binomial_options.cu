#include "binomial_options.cuh"


__device__ inline double expiryCallValue(double S, double X, double vDt, int i)
{
    double d = S * exp(vDt * (2.0 * i - NUM_STEPS)) - X;
    return (d > 0) ? d : 0;
}

__global__ void binomialOptionsKernel(__TOptionData *d_OptionData, double *d_CallBuffer, float *d_CallValue)
{
    __shared__ double callA[CACHE_SIZE+1];
    __shared__ double callB[CACHE_SIZE+1];
    //Global memory frame for current option (thread block)
    double *const d_Call = &d_CallBuffer[blockIdx.x * (NUM_STEPS + 16)];

    const int       tid = threadIdx.x;
    const double      S = d_OptionData[blockIdx.x].S;
    const double      X = d_OptionData[blockIdx.x].X;
    const double    vDt = d_OptionData[blockIdx.x].vDt;
    const double puByDf = d_OptionData[blockIdx.x].puByDf;
    const double pdByDf = d_OptionData[blockIdx.x].pdByDf;

    //Compute values at expiry date
    for (int i = tid; i <= NUM_STEPS; i += CACHE_SIZE)
    {
        d_Call[i] = expiryCallValue(S, X, vDt, i);
    }

    //Walk down binomial tree
    //So double-buffer and synchronize to avoid read-after-write hazards.
    for (int i = NUM_STEPS; i > 0; i -= CACHE_DELTA)
        for (int c_base = 0; c_base < i; c_base += CACHE_STEP)
        {
            //Start and end positions within shared memory cache
            int c_start = min(CACHE_SIZE - 1, i - c_base);
            int c_end   = c_start - CACHE_DELTA;

            //Read data(with apron) to shared memory
            __syncthreads();

            if (tid <= c_start)
            {
                callA[tid] = d_Call[c_base + tid];
            }

            //Calculations within shared memory
            for (int k = c_start - 1; k >= c_end;)
            {
                //Compute discounted expected value
                __syncthreads();
                callB[tid] = puByDf * callA[tid + 1] + pdByDf * callA[tid];
                k--;

                //Compute discounted expected value
                __syncthreads();
                callA[tid] = puByDf * callB[tid + 1] + pdByDf * callB[tid];
                k--;
            }

            //Flush shared memory cache
            __syncthreads();

            if (tid <= c_end)
            {
                d_Call[c_base + tid] = callA[tid];
            }
        }

    //Write the value at the top of the tree to destination buffer
    if (threadIdx.x == 0)
    {
        d_CallValue[blockIdx.x] = (float)callA[0];
    }
}