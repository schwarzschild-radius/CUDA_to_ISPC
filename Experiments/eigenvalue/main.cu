#include <algorithm>
#include <cmath>
#include <float.h>
#include <fstream>
#include <string>
#include <vector>

// Custom includes
#include "bisect_large.cuh"
#include "config.h"
#include "structs.h"
#include "util.h"

void initInputData(InputData &input, const unsigned int mat_size) {
    input.a = (float *)malloc(sizeof(float) * mat_size);
    input.b = (float *)malloc(sizeof(float) * mat_size);

    unsigned int input_data_size = mat_size;
    std::fstream file("./data/diagonal.dat");
    double temp = 0.0;
    for (size_t i = 0; i < input_data_size; i++) {
        file >> temp;
        input.a[i] = temp;
    }
    file.close();

    file.open("./data/superdiagonal.dat");
    for (size_t i = 0; i < input_data_size; i++) {
        file >> temp;
        input.b[i] = temp;
    }
    cudaMalloc((void **)&(input.g_b_raw), sizeof(float) * mat_size);
    cudaMalloc((void **)&(input.g_a), sizeof(float) * mat_size);

    cudaMemcpy(input.g_b_raw, input.b, sizeof(float) * mat_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(input.g_a, input.a, sizeof(float) * mat_size,
               cudaMemcpyHostToDevice);

    input.g_b = input.g_b_raw + 1;
}

void cleanupInputData(InputData &input) {

    freePtr(input.a);
    freePtr(input.b);

    cudaFree(input.g_a);
    input.g_a = NULL;
    cudaFree(input.g_b_raw);
    input.g_b_raw = NULL;
    input.g_b = NULL;
}

void computeGerschgorin(float *d, float *s, unsigned int n, float &lg,
                        float &ug) {

    lg = FLT_MAX;
    ug = -FLT_MAX;

    for (unsigned int i = 1; i < (n - 1); ++i) {

        float sum_abs_ni = fabsf(s[i - 1]) + fabsf(s[i]);

        lg = min(lg, d[i] - sum_abs_ni);
        ug = max(ug, d[i] + sum_abs_ni);
    }

    lg = min(lg, d[0] - fabsf(s[0]));
    ug = max(ug, d[0] + fabsf(s[0]));

    lg = min(lg, d[n - 1] - fabsf(s[n - 2]));
    ug = max(ug, d[n - 1] + fabsf(s[n - 2]));

    float bnorm = max(fabsf(ug), fabsf(lg));

    float psi_0 = 11 * FLT_EPSILON * bnorm;
    float psi_n = 11 * FLT_EPSILON * bnorm;

    lg = lg - bnorm * 2 * n * FLT_EPSILON - psi_0;
    ug = ug + bnorm * 2 * n * FLT_EPSILON + psi_n;

    ug = max(lg, ug);
}

bool runTest() {
    bool bCompareResult = false;

    unsigned int mat_size = 2048;
    float precision = 0.00001f;
    unsigned int iters_timing = 10;
    const char *result_file = "eigenvalues.dat";

    InputData input;
    initInputData(input, mat_size);
    float lg = FLT_MAX;
    float ug = -FLT_MAX;
    computeGerschgorin(input.a, input.b + 1, mat_size, lg, ug);
    printf("Gerschgorin interval: %f / %f\n", lg, ug);

    ResultDataLarge result;
    initResultDataLargeMatrix(result, mat_size);
    computeEigenvaluesLargeMatrix(input, result, mat_size, precision, lg, ug,
                                  iters_timing);

    bCompareResult =
        processResultDataLargeMatrix(input, result, mat_size, result_file);

    cleanupResultDataLargeMatrix(result);
    cleanupInputData(input);

    return bCompareResult;
}

int main() {
    bool bQAResults = false;

    printf("Starting eigenvalues\n");

    bQAResults = runTest();
    printf("Test %s\n", bQAResults ? "Succeeded!" : "Failed!");

    exit(bQAResults ? EXIT_SUCCESS : EXIT_FAILURE);

    // 1
    initData();

    // 2
    executeCUDA();

    // 3
    executeISPC();

    // 4
    executeRef();

    compareResults(N, ref, cuda, ispc);
}