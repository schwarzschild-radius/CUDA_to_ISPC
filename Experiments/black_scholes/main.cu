#include <cmath>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

#include "black_scholes.cuh"
#include "black_scholes.h"
#include "cuda_utils.cuh"

const int OPT_N = 40000;
const int NUM_ITERATIONS = 128;

const int OPT_SZ = OPT_N * sizeof(float);
const float RISKFREE = 0.02f;
const float VOLATILITY = 0.30f;

double CND(double d) {
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

void BlackScholesBodyCPU(float &callResult, float &putResult, float Sf,
                         float Xf, float Tf, float Rf, float Vf) {
    double S = Sf, X = Xf, T = Tf, R = Rf, V = Vf;

    double sqrtT = sqrt(T);
    double d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
    double d2 = d1 - V * sqrtT;
    double CNDD1 = CND(d1);
    double CNDD2 = CND(d2);

    // Calculate Call and Put simultaneously
    double expRT = exp(-R * T);
    callResult = (float)(S * CNDD1 - X * expRT * CNDD2);
    putResult = (float)(X * expRT * (1.0 - CNDD2) - S * (1.0 - CNDD1));
}

void executeReference(float *h_CallResult, float *h_PutResult,
                      float *h_StockPrice, float *h_OptionStrike,
                      float *h_OptionYears) {
    for (int opt = 0; opt < OPT_N; opt++)
        BlackScholesBodyCPU(h_CallResult[opt], h_PutResult[opt],
                            h_StockPrice[opt], h_OptionStrike[opt],
                            h_OptionYears[opt], RISKFREE, VOLATILITY);
}

float RandFloat(float low, float high) {
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

void executeCUDA(float *h_CallResultGPU, float *h_PutResultGPU,
                 float *h_StockPrice, float *h_OptionStrike,
                 float *h_OptionYears) {
    float *d_CallResult, *d_PutResult, *d_StockPrice, *d_OptionStrike,
        *d_OptionYears;
    cudaCheck(cudaMalloc((void **)&d_CallResult, OPT_SZ));
    cudaCheck(cudaMalloc((void **)&d_PutResult, OPT_SZ));
    cudaCheck(cudaMalloc((void **)&d_StockPrice, OPT_SZ));
    cudaCheck(cudaMalloc((void **)&d_OptionStrike, OPT_SZ));
    cudaCheck(cudaMalloc((void **)&d_OptionYears, OPT_SZ));

    cudaCheck(
        cudaMemcpy(d_StockPrice, h_StockPrice, OPT_SZ, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_OptionStrike, h_OptionStrike, OPT_SZ,
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_OptionYears, h_OptionYears, OPT_SZ,
                         cudaMemcpyHostToDevice));

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        cudaCheckLaunch(BlackScholesGPU, 480, 128, 0, d_CallResult, d_PutResult,
                        d_StockPrice, d_OptionStrike, d_OptionYears, RISKFREE,
                        VOLATILITY, OPT_N);
    }

    cudaDeviceSynchronize();

    cudaCheck(cudaMemcpy(h_CallResultGPU, d_CallResult, OPT_SZ,
                               cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_PutResultGPU, d_PutResult, OPT_SZ,
                               cudaMemcpyDeviceToHost));
    cudaCheck(cudaFree(d_OptionYears));
    cudaCheck(cudaFree(d_OptionStrike));
    cudaCheck(cudaFree(d_StockPrice));
    cudaCheck(cudaFree(d_PutResult));
    cudaCheck(cudaFree(d_CallResult));
}

void executeISPC(float *h_CallResult, float *h_PutResult,
                 float *h_StockPrice, float *h_OptionStrike,
                 float *h_OptionYears) {
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        ispc::BlackScholesGPU({480, 1, 1}, {128, 1, 1}, 0, h_CallResult, h_PutResult,
                        h_StockPrice, h_OptionStrike, h_OptionYears, RISKFREE,
                        VOLATILITY, OPT_N);
    }
}

bool checkResult(int N, std::vector<float> ref, std::vector<float> cuda,
                 std::vector<float> ispc) {
    double v_ref, sum_ref;
    double deltaCUDA, sum_deltaCUDA, max_deltaCUDA, L1normCUDA;
    double deltaISPC, sum_deltaISPC, max_deltaISPC, L1normISPC;

    sum_deltaCUDA = 0;
    sum_deltaISPC = 0;

    sum_ref = 0;

    max_deltaCUDA = 0;
    max_deltaISPC = 0;

    for (int i = 0; i < N; i++) {
        // std::cerr << ref[i] << ", " << cuda[i] << ", " << ispc[i] << "\n";
        v_ref = ref[i];
        deltaCUDA = fabs(ref[i] - cuda[i]);
        deltaISPC = fabs(ref[i] - ispc[i]);

        if (deltaCUDA > max_deltaCUDA) {
            max_deltaCUDA = deltaCUDA;
        }

        if (deltaISPC > max_deltaISPC) {
            max_deltaISPC = deltaISPC;
        }

        sum_deltaCUDA += deltaCUDA;
        sum_deltaISPC += deltaISPC;
        sum_ref += fabs(v_ref);
    }

    L1normCUDA = sum_deltaCUDA / sum_ref;
    L1normISPC = sum_deltaISPC / sum_ref;
    printf("\nL1 norm: %E\n", L1normCUDA);
    printf("L1 norm: %E\n", L1normISPC);
    printf("Max absolute error: %E\n", max_deltaCUDA);
    printf("Max absolute error: %E\n", max_deltaISPC);
    if (L1normCUDA > 1e-6 || L1normISPC > 1e-6) {
        std::cerr << "Error!\n";
        return true;
    }
    return false;
}

int main() {
    std::vector<float> h_PutResultCPU(OPT_N), h_PutResultGPU(OPT_N),
        h_StockPrice(OPT_N), h_OptionStrike(OPT_N), h_OptionYears(OPT_N);

    std::vector<float> ref(OPT_N), cuda(OPT_N), ispc(OPT_N);

    srand(time(NULL));

    for (int i = 0; i < OPT_N; i++) {
        ref[i] = 0.0f;
        cuda[i] = 0.0f;
        ispc[i] = 0.0f;
        h_PutResultCPU[i] = -1.0f;
        h_StockPrice[i] = RandFloat(5.0f, 30.0f);
        h_OptionStrike[i] = RandFloat(1.0f, 100.0f);
        h_OptionYears[i] = RandFloat(0.25f, 10.0f);
    }
    std::cout << "Executing GPU:\n";
    executeCUDA(cuda.data(), h_PutResultGPU.data(), h_StockPrice.data(), h_OptionStrike.data(), h_OptionYears.data());
    std::cout << "Executing CPU:\n";
    executeReference(ref.data(), h_PutResultCPU.data(), h_StockPrice.data(), h_OptionStrike.data(), h_OptionYears.data());
    std::cout << "Executing ISPC:\n";
    executeISPC(ispc.data(), h_PutResultCPU.data(), h_StockPrice.data(), h_OptionStrike.data(), h_OptionYears.data());

    if (checkResult(OPT_N, ref, cuda, ispc)) {
        return 1;
    }

    return 0;
}