#include <algorithm>
#include <ctime>
#include <iostream>
#include <vector>

#include "binomial_options.cuh"
#include "binomial_options.h"
#include "cuda_utils.cuh"

float randData(float low, float high) {
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

static double CND(double d) {
    const double A1 = 0.31938153;
    const double A2 = -0.356563782;
    const double A3 = 1.781477937;
    const double A4 = -1.821255978;
    const double A5 = 1.330274429;
    const double RSQRT2PI = 0.39894228040143267793994605993438;

    double K = 1.0 / (1.0 + 0.2316419 * fabs(d));

    double cnd = RSQRT2PI * exp(-0.5 * d * d) *
                 (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0 - cnd;

    return cnd;
}

void executeCUDA(const size_t optN, float *callValue,
                 __TOptionData *optionData) {
    __TOptionData *h_OptionData = new __TOptionData[MAX_OPTIONS];
    __TOptionData *d_OptionData;
    cudaCheck(cudaMalloc((void **)&d_OptionData,
                         MAX_OPTIONS * sizeof(__TOptionData)));
    float *d_CallValue;
    cudaCheck(cudaMalloc((void **)&d_CallValue, sizeof(float) * MAX_OPTIONS));
    double *d_CallBuffer;
    cudaCheck(cudaMalloc((void **)&d_CallBuffer,
                         sizeof(double) * (MAX_OPTIONS * (NUM_STEPS + 16))));

    for (int i = 0; i < optN; i++) {
        const double T = optionData[i].T;
        const double R = optionData[i].R;
        const double V = optionData[i].V;

        const double dt = T / (double)NUM_STEPS;
        const double vDt = V * sqrt(dt);
        const double rDt = R * dt;
        // Per-step interest and discount factors
        const double If = exp(rDt);
        const double Df = exp(-rDt);
        // Values and pseudoprobabilities of upward and downward moves
        const double u = exp(vDt);
        const double d = exp(-vDt);
        const double pu = (If - d) / (u - d);
        const double pd = 1.0 - pu;
        const double puByDf = pu * Df;
        const double pdByDf = pd * Df;

        h_OptionData[i].S = (double)optionData[i].S;
        h_OptionData[i].X = (double)optionData[i].X;
        h_OptionData[i].vDt = (double)vDt;
        h_OptionData[i].puByDf = (double)puByDf;
        h_OptionData[i].pdByDf = (double)pdByDf;
    }
    cudaCheck(cudaMemcpy(d_OptionData, h_OptionData,
                         optN * sizeof(__TOptionData), cudaMemcpyHostToDevice));

    cudaCheckLaunch(binomialOptionsKernel, optN, CACHE_SIZE, 0, d_OptionData,
                    d_CallBuffer, d_CallValue);

    cudaDeviceSynchronize();
    cudaCheck(cudaMemcpy(callValue, d_CallValue, sizeof(float) * optN,
                         cudaMemcpyDeviceToHost));
}

static double expiryCallValue(double S, double X, double vDt, int i) {
    double d = S * exp(vDt * (2.0 * i - NUM_STEPS)) - X;
    return (d > 0) ? d : 0;
}

void executeReference(const unsigned int N, float *callResult,
                      __TOptionData *optionData) {
    for (int i = 0; i < N; i++) {
        static double Call[NUM_STEPS + 1];

        const double S = optionData[i].S;
        const double X = optionData[i].X;
        const double T = optionData[i].T;
        const double R = optionData[i].R;
        const double V = optionData[i].V;

        const double dt = T / (double)NUM_STEPS;
        const double vDt = V * sqrt(dt);
        const double rDt = R * dt;
        // Per-step interest and discount factors
        const double If = exp(rDt);
        const double Df = exp(-rDt);
        // Values and pseudoprobabilities of upward and downward moves
        const double u = exp(vDt);
        const double d = exp(-vDt);
        const double pu = (If - d) / (u - d);
        const double pd = 1.0 - pu;
        const double puByDf = pu * Df;
        const double pdByDf = pd * Df;

        ///////////////////////////////////////////////////////////////////////
        // Compute values at expiration date:
        // call option value at period end is V(T) = S(T) - X
        // if S(T) is greater than X, or zero otherwise.
        // The computation is similar for put options.
        ///////////////////////////////////////////////////////////////////////
        for (int i = 0; i <= NUM_STEPS; i++)
            Call[i] = expiryCallValue(S, X, vDt, i);

        ////////////////////////////////////////////////////////////////////////
        // Walk backwards up binomial tree
        ////////////////////////////////////////////////////////////////////////
        for (int i = NUM_STEPS; i > 0; i--)
            for (int j = 0; j <= i - 1; j++)
                Call[j] = puByDf * Call[j + 1] + pdByDf * Call[j];

        callResult[i] = (float)Call[0];
    }
}

void executeISPC(const size_t optN, float *callValue,
                 __TOptionData *optionData) {
    __TOptionData *h_OptionData = new __TOptionData[MAX_OPTIONS];
    ispc::__TOptionData *ispc_OptionData = new ispc::__TOptionData[MAX_OPTIONS];
    double *CallBuffer = new double[MAX_OPTIONS * (NUM_STEPS + 16)];

    for (int i = 0; i < optN; i++) {
        const double T = optionData[i].T;
        const double R = optionData[i].R;
        const double V = optionData[i].V;

        const double dt = T / (double)NUM_STEPS;
        const double vDt = V * sqrt(dt);
        const double rDt = R * dt;
        // Per-step interest and discount factors
        const double If = exp(rDt);
        const double Df = exp(-rDt);
        // Values and pseudoprobabilities of upward and downward moves
        const double u = exp(vDt);
        const double d = exp(-vDt);
        const double pu = (If - d) / (u - d);
        const double pd = 1.0 - pu;
        const double puByDf = pu * Df;
        const double pdByDf = pd * Df;

        h_OptionData[i].S = (double)optionData[i].S;
        h_OptionData[i].X = (double)optionData[i].X;
        h_OptionData[i].vDt = (double)vDt;
        h_OptionData[i].puByDf = (double)puByDf;
        h_OptionData[i].pdByDf = (double)pdByDf;
    }
    memcpy(ispc_OptionData, h_OptionData,
           sizeof(ispc::__TOptionData) * MAX_OPTIONS);
    ispc::binomialOptionsKernel({static_cast<int32_t>(optN), 1, 1},
                                {static_cast<int32_t>(CACHE_SIZE), 1, 1}, 0,
                                ispc_OptionData, CallBuffer, callValue);
}

bool checkResult(size_t N, std::vector<float> ref, std::vector<float> cuda,
                 std::vector<float> ispc) {
    float sumDeltaCUDA = 0;
    float sumDeltaISPC = 0;
    float sumRef = 0;
    float errorValCUDA = 0.0f;
    float errorValISPC = 0.0f;
    for (int i = 0; i < N; i++) {
        sumDeltaCUDA += fabs(cuda[i] - ref[i]);
/*         std::cout << "Results: " << cuda[i] << ", " << ref[i] << ", " << ispc[i]
                  << "\n"; */
        sumDeltaISPC += fabs(ispc[i] - ref[i]);
        sumRef += ref[i];
    }
    if (sumRef > 1E-5) {
        printf("L1 norm: %E\n", errorValCUDA = sumDeltaCUDA / sumRef);
        printf("L1 norm: %E\n", errorValISPC = sumDeltaISPC / sumRef);
    } else {
        printf("Avg. diff: %E\n", errorValCUDA = sumDeltaCUDA / (double)N);
        printf("Avg. diff: %E\n", errorValISPC = sumDeltaISPC / (double)N);
    }
    if (errorValCUDA > 5e-4 || errorValISPC > 5e-4) {
        std::cout << "Error!\n";
        return true; // true when something goes wrong
    }
    return false;
}

int main() {
    const unsigned int OPT_N_MAX = 512;
    unsigned int useDoublePrecision;
    const size_t OPT_N = OPT_N_MAX;

    __TOptionData optionData[OPT_N_MAX];
    float callValueBS[OPT_N_MAX], callValueGPU[OPT_N_MAX],
        callValueCPU[OPT_N_MAX];

    double sumDelta, sumRef, gpuTime, errorVal;
    srand(time(NULL));

    for (int i = 0; i < OPT_N; i++) {
        optionData[i].S = randData(5.0f, 30.0f);
        optionData[i].X = randData(1.0f, 100.0f);
        optionData[i].T = randData(0.25f, 10.0f);
        optionData[i].R = 0.06f;
        optionData[i].V = 0.10f;
    }

    std::vector<float> ref(OPT_N), cuda(OPT_N), ispc(OPT_N);

    executeCUDA(OPT_N, cuda.data(), optionData);
    executeReference(OPT_N, ref.data(), optionData);
    executeISPC(OPT_N, ispc.data(), optionData);
    if (checkResult(OPT_N, ref, cuda, ispc))
        return 1;
    return 0;
}
