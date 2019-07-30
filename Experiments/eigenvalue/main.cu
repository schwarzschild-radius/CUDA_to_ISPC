#include <algorithm>
#include <cmath>
#include <float.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// CUDA headers
#include "ISPCUtils.cuh"
#include "bisect_kernel_large.cuh"
#include "bisect_kernel_large_multi.cuh"
#include "bisect_kernel_large_onei.cuh"

// ISPC headers
#include "bisect_kernel_large.h"
#include "bisect_kernel_large_multi.h"
#include "bisect_kernel_large_onei.h"

// Custom headers
#include "config.h"
#include "structs.h"
#include "util.h"

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

void initResultDataLargeMatrix(ResultDataLarge &result,
                               const unsigned int mat_size) {

    // helper variables to initialize memory
    unsigned int zero = 0;
    unsigned int mat_size_f = sizeof(float) * mat_size;
    unsigned int mat_size_ui = sizeof(unsigned int) * mat_size;

    float *tempf = (float *)malloc(mat_size_f);
    unsigned int *tempui = (unsigned int *)malloc(mat_size_ui);

    for (unsigned int i = 0; i < mat_size; ++i) {
        tempf[i] = 0.0f;
        tempui[i] = 0;
    }

    // number of intervals containing only one eigenvalue after the first step
    cudaMalloc((void **)&result.g_num_one, sizeof(unsigned int));
    cudaMemcpy(result.g_num_one, &zero, sizeof(unsigned int),
               cudaMemcpyHostToDevice);

    // number of (thread) blocks of intervals with multiple eigenvalues after
    // the first iteration
    cudaMalloc((void **)&result.g_num_blocks_mult, sizeof(unsigned int));
    cudaMemcpy(result.g_num_blocks_mult, &zero, sizeof(unsigned int),
               cudaMemcpyHostToDevice);

    cudaMalloc((void **)&result.g_left_one, mat_size_f);
    cudaMalloc((void **)&result.g_right_one, mat_size_f);
    cudaMalloc((void **)&result.g_pos_one, mat_size_ui);

    cudaMalloc((void **)&result.g_left_mult, mat_size_f);
    cudaMalloc((void **)&result.g_right_mult, mat_size_f);
    cudaMalloc((void **)&result.g_left_count_mult, mat_size_ui);
    cudaMalloc((void **)&result.g_right_count_mult, mat_size_ui);

    cudaMemcpy(result.g_left_one, tempf, mat_size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(result.g_right_one, tempf, mat_size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(result.g_pos_one, tempui, mat_size_ui, cudaMemcpyHostToDevice);

    cudaMemcpy(result.g_left_mult, tempf, mat_size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(result.g_right_mult, tempf, mat_size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(result.g_left_count_mult, tempui, mat_size_ui,
               cudaMemcpyHostToDevice);
    cudaMemcpy(result.g_right_count_mult, tempui, mat_size_ui,
               cudaMemcpyHostToDevice);

    cudaMalloc((void **)&result.g_blocks_mult, mat_size_ui);
    cudaMemcpy(result.g_blocks_mult, tempui, mat_size_ui,
               cudaMemcpyHostToDevice);
    cudaMalloc((void **)&result.g_blocks_mult_sum, mat_size_ui);
    cudaMemcpy(result.g_blocks_mult_sum, tempui, mat_size_ui,
               cudaMemcpyHostToDevice);

    cudaMalloc((void **)&result.g_lambda_mult, mat_size_f);
    cudaMemcpy(result.g_lambda_mult, tempf, mat_size_f, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&result.g_pos_mult, mat_size_ui);
    cudaMemcpy(result.g_pos_mult, tempf, mat_size_ui, cudaMemcpyHostToDevice);
}

std::vector<float> executeCUDA(size_t mat_size, float precision,
                               size_t iterations,
                               std::vector<float> diagonal_data,
                               std::vector<float> superdiagonal_data) {
    InputData input;
    ResultDataLarge result;
    initResultDataLargeMatrix(result, mat_size);

    // 1. initialisation
    input.a = (float *)malloc(sizeof(float) * mat_size);
    input.b = (float *)malloc(sizeof(float) * mat_size);
    std::copy(diagonal_data.begin(), diagonal_data.end(), input.a);
    std::copy(superdiagonal_data.begin(), superdiagonal_data.end(), input.b);

    cudaMalloc((void **)&(input.g_b_raw), sizeof(float) * mat_size);
    cudaMalloc((void **)&(input.g_a), sizeof(float) * mat_size);

    cudaMemcpy(input.g_b_raw, input.b, sizeof(float) * mat_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(input.g_a, input.a, sizeof(float) * mat_size,
               cudaMemcpyHostToDevice);

    input.g_b = input.g_b_raw + 1;

    // 2. execute

    float lg = FLT_MAX;
    float ug = -FLT_MAX;
    computeGerschgorin(diagonal_data.data(), superdiagonal_data.data() + 1,
                       mat_size, lg, ug);
    printf("CUDA Gerschgorin interval: %f / %f\n", lg, ug);

    dim3 blocks(1, 1, 1);
    dim3 threads(MAX_THREADS_BLOCK, 1, 1);
    for (unsigned int iter = 0; iter < iterations; ++iter) {
        bisectKernelLarge<<<blocks, threads>>>(
            input.g_a, input.g_b, mat_size, lg, ug, 0, mat_size, precision,
            result.g_num_one, result.g_num_blocks_mult, result.g_left_one,
            result.g_right_one, result.g_pos_one, result.g_left_mult,
            result.g_right_mult, result.g_left_count_mult,
            result.g_right_count_mult, result.g_blocks_mult,
            result.g_blocks_mult_sum);
        cudaDeviceSynchronize();

        // get the number of intervals containing one eigenvalue after the first
        // processing step
        unsigned int num_one_intervals;
        cudaMemcpy(&num_one_intervals, result.g_num_one, sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);

        dim3 grid_onei;
        grid_onei.x = getNumBlocksLinear(num_one_intervals, MAX_THREADS_BLOCK);
        dim3 threads_onei;
        // use always max number of available threads to better balance load
        // times for matrix data
        threads_onei.x = MAX_THREADS_BLOCK;

        // compute eigenvalues for intervals that contained only one eigenvalue
        // after the first processing step

        bisectKernelLarge_OneIntervals<<<grid_onei, threads_onei>>>(
            input.g_a, input.g_b, mat_size, num_one_intervals,
            result.g_left_one, result.g_right_one, result.g_pos_one, precision);

        cudaDeviceSynchronize();

        // process intervals that contained more than one eigenvalue after
        // the first processing step

        // get the number of blocks of intervals that contain, in total when
        // each interval contains only one eigenvalue, not more than
        // MAX_THREADS_BLOCK threads
        unsigned int num_blocks_mult = 0;
        cudaMemcpy(&num_blocks_mult, result.g_num_blocks_mult,
                   sizeof(unsigned int), cudaMemcpyDeviceToHost);

        // setup the execution environment
        dim3 grid_mult(num_blocks_mult, 1, 1);
        dim3 threads_mult(MAX_THREADS_BLOCK, 1, 1);

        bisectKernelLarge_MultIntervals<<<grid_mult, threads_mult>>>(
            input.g_a, input.g_b, mat_size, result.g_blocks_mult,
            result.g_blocks_mult_sum, result.g_left_mult, result.g_right_mult,
            result.g_left_count_mult, result.g_right_count_mult,
            result.g_lambda_mult, result.g_pos_mult, precision);
    }

    // 3. process data
    const unsigned int mat_size_ui = sizeof(unsigned int) * mat_size;
    const unsigned int mat_size_f = sizeof(float) * mat_size;

    float *lambda_mult = (float *)malloc(sizeof(float) * mat_size);
    cudaMemcpy(lambda_mult, result.g_lambda_mult, sizeof(float) * mat_size,
               cudaMemcpyDeviceToHost);
    unsigned int *pos_mult =
        (unsigned int *)malloc(sizeof(unsigned int) * mat_size);
    cudaMemcpy(pos_mult, result.g_pos_mult, sizeof(unsigned int) * mat_size,
               cudaMemcpyDeviceToHost);

    unsigned int *blocks_mult_sum =
        (unsigned int *)malloc(sizeof(unsigned int) * mat_size);
    cudaMemcpy(blocks_mult_sum, result.g_blocks_mult_sum,
               sizeof(unsigned int) * mat_size, cudaMemcpyDeviceToHost);

    unsigned int num_one_intervals;
    cudaMemcpy(&num_one_intervals, result.g_num_one, sizeof(unsigned int),
               cudaMemcpyDeviceToHost);

    unsigned int sum_blocks_mult = mat_size - num_one_intervals;

    float *left_one = (float *)malloc(mat_size_f);
    float *right_one = (float *)malloc(mat_size_f);
    unsigned int *pos_one = (unsigned int *)malloc(mat_size_ui);
    cudaMemcpy(left_one, result.g_left_one, mat_size_f, cudaMemcpyDeviceToHost);
    cudaMemcpy(right_one, result.g_right_one, mat_size_f,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(pos_one, result.g_pos_one, mat_size_ui, cudaMemcpyDeviceToHost);

    std::vector<float> eigenvals(mat_size_f);

    // singleton intervals generated in the second step
    for (unsigned int i = 0; i < sum_blocks_mult; ++i) {
        eigenvals[pos_mult[i] - 1] = lambda_mult[i];
    }
    // singleton intervals generated in the first step
    unsigned int index = 0;

    for (unsigned int i = 0; i < num_one_intervals; ++i, ++index) {

        eigenvals[pos_one[i] - 1] = left_one[i];
    }
    freePtr(lambda_mult);
    freePtr(pos_mult);
    freePtr(blocks_mult_sum);
    freePtr(left_one);
    freePtr(right_one);
    freePtr(pos_one);

    return eigenvals;
}

std::vector<float> executeReference(size_t mat_size) {
    unsigned int input_data_size = mat_size;
    std::fstream file("./data/reference.dat");
    std::vector<float> reference(input_data_size);
    float temp;
    for (size_t i = 0; i < input_data_size; i++) {
        file >> temp;
        reference[i] = temp;
    }
    return reference;
}

inline bool sdkCompareL2fe(const float *reference, const float *data,
                           const unsigned int len, const float epsilon) {

    float error = 0;
    float ref = 0;

    for (unsigned int i = 0; i < len; ++i) {

        float diff = reference[i] - data[i];
        // std::cout << reference[i] << " " << data[i] << '\n';
        error += diff * diff;
        ref += reference[i] * reference[i];
    }

    float normRef = sqrtf(ref);
    if (fabs(ref) < 1e-7) {
        std::cerr << "ERROR, reference l2-norm is 0\n";
        return false;
    }
    float normError = sqrtf(error);
    error = normError / normRef;
    bool result = error < epsilon;
    if (!result) {
        std::cerr << "ERROR, l2-norm error " << error
                  << " is greater than epsilon " << epsilon << "\n";
    }

    return result;
}

void compareResults(int mat_size, std::vector<float> &reference,
                    std::vector<float> cuda, std::vector<float> ispc) {
    float tolerance = 1.0e-5f + 5.0e-6f;
    if (sdkCompareL2fe(reference.data(), cuda.data(), mat_size, tolerance) ==
        false)
        std::cout << "Error!\n";
    if (sdkCompareL2fe(reference.data(), ispc.data(), mat_size, tolerance) == false){
        std::cout << "Error!\n";
    }
}

void initISPCResultDataLargeMatrix(ResultDataLarge &result,
                                   const unsigned int mat_size) {

    // helper variables to initialize memory
    unsigned int zero = 0;
    unsigned int mat_size_f = sizeof(float) * mat_size;
    unsigned int mat_size_ui = sizeof(unsigned int) * mat_size;

    float *tempf = (float *)malloc(mat_size_f);
    unsigned int *tempui = (unsigned int *)malloc(mat_size_ui);

    for (unsigned int i = 0; i < mat_size; ++i) {
        tempf[i] = 0.0f;
        tempui[i] = 0;
    }

    // number of intervals containing only one eigenvalue after the first step
    result.g_num_one = (unsigned int *)malloc(sizeof(unsigned int));
    memcpy(result.g_num_one, &zero, sizeof(unsigned int));

    // number of (thread) blocks of intervals with multiple eigenvalues after
    // the first iteration
    result.g_num_blocks_mult = (unsigned int *)malloc(sizeof(unsigned int));
    memcpy(result.g_num_blocks_mult, &zero, sizeof(unsigned int));

    result.g_left_one = (float *)malloc(mat_size_f);
    result.g_right_one = (float *)malloc(mat_size_f);
    result.g_pos_one = (unsigned int *)malloc(mat_size_ui);

    result.g_left_mult = (float *)malloc(mat_size_f);
    result.g_right_mult = (float *)malloc(mat_size_f);
    result.g_left_count_mult = (unsigned int *)malloc(mat_size_ui);
    result.g_right_count_mult = (unsigned int *)malloc(mat_size_ui);

    memcpy(result.g_left_one, tempf, mat_size_f);
    memcpy(result.g_right_one, tempf, mat_size_f);
    memcpy(result.g_pos_one, tempui, mat_size_ui);

    memcpy(result.g_left_mult, tempf, mat_size_f);
    memcpy(result.g_right_mult, tempf, mat_size_f);
    memcpy(result.g_left_count_mult, tempui, mat_size_ui);
    memcpy(result.g_right_count_mult, tempui, mat_size_ui);

    result.g_blocks_mult = (unsigned int *)malloc(mat_size_ui);
    memcpy(result.g_blocks_mult, tempui, mat_size_ui);
    result.g_blocks_mult_sum = (unsigned int *)malloc(mat_size_ui);
    memcpy(result.g_blocks_mult_sum, tempui, mat_size_ui);

    result.g_lambda_mult = (float *)malloc(mat_size_f);
    memcpy(result.g_lambda_mult, tempf, mat_size_f);
    result.g_pos_mult = (unsigned int *)malloc(mat_size_ui);
    memcpy(result.g_pos_mult, tempf, mat_size_ui);
}

std::vector<float> executeISPC(size_t mat_size, float precision,
                               size_t iterations,
                               std::vector<float> diagonal_data,
                               std::vector<float> superdiagonal_data) {
    InputData input;
    ResultDataLarge result;
    initISPCResultDataLargeMatrix(result, mat_size);

    input.a = (float *)malloc(sizeof(float) * mat_size);
    input.b = (float *)malloc(sizeof(float) * mat_size);
    std::copy(diagonal_data.begin(), diagonal_data.end(), input.a);
    std::copy(superdiagonal_data.begin(), superdiagonal_data.end(), input.b);

    float lg = FLT_MAX;
    float ug = -FLT_MAX;
    computeGerschgorin(diagonal_data.data(), superdiagonal_data.data() + 1,
                       mat_size, lg, ug);
    printf("CUDA Gerschgorin interval: %f / %f\n", lg, ug);

    dim3 blocks(1, 1, 1);
    dim3 threads(MAX_THREADS_BLOCK, 1, 1);
    for (unsigned int iter = 0; iter < iterations; ++iter) {
        ISPC_LAUNCH_KERNEL(
            bisectKernelLarge, {blocks.x, blocks.y, blocks.z},
            {threads.x, threads.y, threads.z}, 0, input.a, input.b, mat_size,
            lg, ug, 0, mat_size, precision, result.g_num_one,
            result.g_num_blocks_mult, result.g_left_one, result.g_right_one,
            result.g_pos_one, result.g_left_mult, result.g_right_mult,
            result.g_left_count_mult, result.g_right_count_mult,
            result.g_blocks_mult, result.g_blocks_mult_sum);

        unsigned int num_one_intervals;
        memcpy(&num_one_intervals, result.g_num_one, sizeof(unsigned int));

        dim3 grid_onei;
        grid_onei.x = getNumBlocksLinear(num_one_intervals, MAX_THREADS_BLOCK);
        dim3 threads_onei;
        // use always max number of available threads to better balance load
        // times for matrix data
        threads_onei.x = MAX_THREADS_BLOCK;

        // compute eigenvalues for intervals that contained only one eigenvalue
        // after the first processing step

        ISPC_LAUNCH_KERNEL(bisectKernelLarge_OneIntervals,
                           {grid_onei.x, grid_onei.y, grid_onei.z},
                           {threads_onei.x, threads_onei.y, threads_onei.z}, 0,
                           input.g_a, input.g_b, mat_size, num_one_intervals,
                           result.g_left_one, result.g_right_one,
                           result.g_pos_one, precision);

        unsigned int num_blocks_mult = 0;
        memcpy(&num_blocks_mult, result.g_num_blocks_mult,
               sizeof(unsigned int));

        // setup the execution environment
        dim3 grid_mult(num_blocks_mult, 1, 1);
        dim3 threads_mult(MAX_THREADS_BLOCK, 1, 1);

        ISPC_LAUNCH_KERNEL(bisectKernelLarge_MultIntervals,
                           {grid_mult.x, grid_mult.y, grid_mult.z},
                           {threads_mult.x, threads_mult.y, threads_mult.z}, 0,
                           input.g_a, input.g_b, mat_size, result.g_blocks_mult,
                           result.g_blocks_mult_sum, result.g_left_mult,
                           result.g_right_mult, result.g_left_count_mult,
                           result.g_right_count_mult, result.g_lambda_mult,
                           result.g_pos_mult, precision);
    }

    const unsigned int mat_size_ui = sizeof(unsigned int) * mat_size;
    const unsigned int mat_size_f = sizeof(float) * mat_size;

    float *lambda_mult = (float *)malloc(sizeof(float) * mat_size);
    memcpy(lambda_mult, result.g_lambda_mult, sizeof(float) * mat_size);
    unsigned int *pos_mult =
        (unsigned int *)malloc(sizeof(unsigned int) * mat_size);
    memcpy(pos_mult, result.g_pos_mult, sizeof(unsigned int) * mat_size);

    unsigned int *blocks_mult_sum =
        (unsigned int *)malloc(sizeof(unsigned int) * mat_size);
    memcpy(blocks_mult_sum, result.g_blocks_mult_sum,
           sizeof(unsigned int) * mat_size);

    unsigned int num_one_intervals;
    memcpy(&num_one_intervals, result.g_num_one, sizeof(unsigned int));

    unsigned int sum_blocks_mult = mat_size - num_one_intervals;

    float *left_one = (float *)malloc(mat_size_f);
    float *right_one = (float *)malloc(mat_size_f);
    unsigned int *pos_one = (unsigned int *)malloc(mat_size_ui);
    memcpy(left_one, result.g_left_one, mat_size_f);
    memcpy(right_one, result.g_right_one, mat_size_f);
    memcpy(pos_one, result.g_pos_one, mat_size_ui);

    std::vector<float> eigenvals(mat_size_f);

    // singleton intervals generated in the second step
    for (unsigned int i = 0; i < sum_blocks_mult; ++i) {
        if(pos_mult[i] > 0)
            eigenvals[pos_mult[i] - 1] = lambda_mult[i];
    }
    // singleton intervals generated in the first step
    unsigned int index = 0;

    for (unsigned int i = 0; i < num_one_intervals; ++i, ++index) {
        if(pos_mult[i] > 0)
            eigenvals[pos_one[i] - 1] = left_one[i];
    }
    delete lambda_mult;
    delete pos_mult;
    delete blocks_mult_sum;
    delete left_one;
    delete right_one;
    delete pos_one;

    return eigenvals;
}

int main() {
    // setting up common data
    unsigned int mat_size = 2048;
    float precision = 0.00001f;
    unsigned int iterations = 10;
    size_t input_data_size = mat_size;

    std::vector<float> diagonal_data(input_data_size),
        superdiagonal_data(input_data_size);

    std::fstream file("./data/diagonal.dat");
    double temp = 0.0;
    for (size_t i = 0; i < input_data_size; i++) {
        file >> temp;
        diagonal_data[i] = temp;
        // std::cout << diagonal_data[i] << '\n';
    }
    file.close();

    file.open("./data/superdiagonal.dat");
    for (size_t i = 0; i < input_data_size; i++) {
        file >> temp;
        superdiagonal_data[i] = temp;
    }

    // 2
    auto cuda = executeCUDA(mat_size, precision, iterations, diagonal_data,
                            superdiagonal_data);

    // 3
    auto ispc = executeISPC(mat_size, precision, iterations, diagonal_data,
                            superdiagonal_data);

    // 4
    auto ref = executeReference(mat_size);

    compareResults(mat_size, ref, cuda, ispc);
}