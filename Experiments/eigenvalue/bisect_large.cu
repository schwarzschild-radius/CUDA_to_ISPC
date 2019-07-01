/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Computation of eigenvalues of a large symmetric, tridiagonal matrix */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>

#include "config.h"
#include "structs.h"
#include "util.h"

#include "bisect_large.cuh"

// includes, kernels
#include "bisect_kernel_large.cu"
// #include "bisect_kernel_large_onei.cu"
// #include "bisect_kernel_large_multi.cu"


////////////////////////////////////////////////////////////////////////////////
//! Initialize variables and memory for result
//! @param  result handles to memory
//! @param  matrix_size  size of the matrix
////////////////////////////////////////////////////////////////////////////////
void
initResultDataLargeMatrix(ResultDataLarge &result, const unsigned int mat_size)
{

    // helper variables to initialize memory
    unsigned int zero = 0;
    unsigned int mat_size_f = sizeof(float) * mat_size;
    unsigned int mat_size_ui = sizeof(unsigned int) * mat_size;

    float *tempf = (float *) malloc(mat_size_f);
    unsigned int *tempui = (unsigned int *) malloc(mat_size_ui);

    for (unsigned int i = 0; i < mat_size; ++i)
    {
        tempf[i] = 0.0f;
        tempui[i] = 0;
    }

    // number of intervals containing only one eigenvalue after the first step
    cudaMalloc((void **) &result.g_num_one,
                               sizeof(unsigned int));
    cudaMemcpy(result.g_num_one, &zero, sizeof(unsigned int),
                               cudaMemcpyHostToDevice);

    // number of (thread) blocks of intervals with multiple eigenvalues after
    // the first iteration
    cudaMalloc((void **) &result.g_num_blocks_mult,
                               sizeof(unsigned int));
    cudaMemcpy(result.g_num_blocks_mult, &zero,
                               sizeof(unsigned int),
                               cudaMemcpyHostToDevice);


    cudaMalloc((void **) &result.g_left_one, mat_size_f);
    cudaMalloc((void **) &result.g_right_one, mat_size_f);
    cudaMalloc((void **) &result.g_pos_one, mat_size_ui);

    cudaMalloc((void **) &result.g_left_mult, mat_size_f);
    cudaMalloc((void **) &result.g_right_mult, mat_size_f);
    cudaMalloc((void **) &result.g_left_count_mult,
                               mat_size_ui);
    cudaMalloc((void **) &result.g_right_count_mult,
                               mat_size_ui);

    cudaMemcpy(result.g_left_one, tempf, mat_size_f,
                               cudaMemcpyHostToDevice);
    cudaMemcpy(result.g_right_one, tempf, mat_size_f,
                               cudaMemcpyHostToDevice);
    cudaMemcpy(result.g_pos_one, tempui, mat_size_ui,
                               cudaMemcpyHostToDevice);

    cudaMemcpy(result.g_left_mult, tempf, mat_size_f,
                               cudaMemcpyHostToDevice);
    cudaMemcpy(result.g_right_mult, tempf, mat_size_f,
                               cudaMemcpyHostToDevice);
    cudaMemcpy(result.g_left_count_mult, tempui, mat_size_ui,
                               cudaMemcpyHostToDevice);
    cudaMemcpy(result.g_right_count_mult, tempui, mat_size_ui,
                               cudaMemcpyHostToDevice);

    cudaMalloc((void **) &result.g_blocks_mult, mat_size_ui);
    cudaMemcpy(result.g_blocks_mult, tempui, mat_size_ui,
                               cudaMemcpyHostToDevice);
    cudaMalloc((void **) &result.g_blocks_mult_sum, mat_size_ui);
    cudaMemcpy(result.g_blocks_mult_sum, tempui, mat_size_ui,
                               cudaMemcpyHostToDevice);

    cudaMalloc((void **) &result.g_lambda_mult, mat_size_f);
    cudaMemcpy(result.g_lambda_mult, tempf, mat_size_f,
                               cudaMemcpyHostToDevice);
    cudaMalloc((void **) &result.g_pos_mult, mat_size_ui);
    cudaMemcpy(result.g_pos_mult, tempf, mat_size_ui,
                               cudaMemcpyHostToDevice);
}

////////////////////////////////////////////////////////////////////////////////
//! Cleanup result memory
//! @param result  handles to memory
////////////////////////////////////////////////////////////////////////////////
void
cleanupResultDataLargeMatrix(ResultDataLarge &result)
{

    cudaFree(result.g_num_one);
    cudaFree(result.g_num_blocks_mult);
    cudaFree(result.g_left_one);
    cudaFree(result.g_right_one);
    cudaFree(result.g_pos_one);
    cudaFree(result.g_left_mult);
    cudaFree(result.g_right_mult);
    cudaFree(result.g_left_count_mult);
    cudaFree(result.g_right_count_mult);
    cudaFree(result.g_blocks_mult);
    cudaFree(result.g_blocks_mult_sum);
    cudaFree(result.g_lambda_mult);
    cudaFree(result.g_pos_mult);
}

////////////////////////////////////////////////////////////////////////////////
//! Run the kernels to compute the eigenvalues for large matrices
//! @param  input   handles to input data
//! @param  result  handles to result data
//! @param  mat_size  matrix size
//! @param  precision  desired precision of eigenvalues
//! @param  lg  lower limit of Gerschgorin interval
//! @param  ug  upper limit of Gerschgorin interval
//! @param  iterations  number of iterations (for timing)
////////////////////////////////////////////////////////////////////////////////
void
computeEigenvaluesLargeMatrix(const InputData &input, const ResultDataLarge &result,
                              const unsigned int mat_size, const float precision,
                              const float lg, const float ug,
                              const unsigned int iterations)
{
    dim3  blocks(1, 1, 1);
    dim3  threads(MAX_THREADS_BLOCK, 1, 1);

    // do for multiple iterations to improve timing accuracy
    for (unsigned int iter = 0; iter < iterations; ++iter)
    {

        bisectKernelLarge<<< blocks, threads >>>
        (input.g_a, input.g_b, mat_size,
         lg, ug, 0, mat_size, precision,
         result.g_num_one, result.g_num_blocks_mult,
         result.g_left_one, result.g_right_one, result.g_pos_one,
         result.g_left_mult, result.g_right_mult,
         result.g_left_count_mult, result.g_right_count_mult,
         result.g_blocks_mult, result.g_blocks_mult_sum
        );
        cudaDeviceSynchronize();

        // get the number of intervals containing one eigenvalue after the first
        // processing step
        unsigned int num_one_intervals;
        cudaMemcpy(&num_one_intervals, result.g_num_one,
                                   sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost);

        dim3 grid_onei;
        grid_onei.x = getNumBlocksLinear(num_one_intervals, MAX_THREADS_BLOCK);
        dim3 threads_onei;
        // use always max number of available threads to better balance load times
        // for matrix data
        threads_onei.x = MAX_THREADS_BLOCK;

        // compute eigenvalues for intervals that contained only one eigenvalue
        // after the first processing step

       /*  bisectKernelLarge_OneIntervals<<< grid_onei , threads_onei >>>
        (input.g_a, input.g_b, mat_size, num_one_intervals,
         result.g_left_one, result.g_right_one, result.g_pos_one,
         precision
        );

        cudaDeviceSynchronize(); */

        // process intervals that contained more than one eigenvalue after
        // the first processing step

        // get the number of blocks of intervals that contain, in total when
        // each interval contains only one eigenvalue, not more than
        // MAX_THREADS_BLOCK threads
        unsigned int  num_blocks_mult = 0;
        cudaMemcpy(&num_blocks_mult, result.g_num_blocks_mult,
                                   sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost);

        // setup the execution environment
        dim3  grid_mult(num_blocks_mult, 1, 1);
        dim3  threads_mult(MAX_THREADS_BLOCK, 1, 1);

/*         bisectKernelLarge_MultIntervals<<< grid_mult, threads_mult >>>
        (input.g_a, input.g_b, mat_size,
         result.g_blocks_mult, result.g_blocks_mult_sum,
         result.g_left_mult, result.g_right_mult,
         result.g_left_count_mult, result.g_right_count_mult,
         result.g_lambda_mult, result.g_pos_mult,
         precision
        ); */
    }
}

inline bool
sdkCompareL2fe( const float* reference, const float* data,
                const unsigned int len, const float epsilon ) 
{

    float error = 0;
    float ref = 0;

    for( unsigned int i = 0; i < len; ++i) {

        float diff = reference[i] - data[i];
        error += diff * diff;
        ref += reference[i] * reference[i];
    }

    float normRef = sqrtf(ref);
    if (fabs(ref) < 1e-7) {
#ifdef _DEBUG
        std::cerr << "ERROR, reference l2-norm is 0\n";
#endif
        return false;
    }
    float normError = sqrtf(error);
    error = normError / normRef;
    bool result = error < epsilon;
#ifdef _DEBUG
    if( ! result) 
    {
        std::cerr << "ERROR, l2-norm error " 
            << error << " is greater than epsilon " << epsilon << "\n";
    }
#endif

    return result;
}

////////////////////////////////////////////////////////////////////////////////
//! Process the result, that is obtain result from device and do simple sanity
//! checking
//! @param  input   handles to input data
//! @param  result  handles to result data
//! @param  mat_size  matrix size
//! @param  filename  output filename
////////////////////////////////////////////////////////////////////////////////
bool
processResultDataLargeMatrix(const InputData &input, const ResultDataLarge &result,
                             const unsigned int mat_size,
                             const char *filename,
                             const unsigned int user_defined, char *exec_path)
{
    bool bCompareResult = false;
    const unsigned int mat_size_ui = sizeof(unsigned int) * mat_size;
    const unsigned int mat_size_f  = sizeof(float) * mat_size;

    // copy data from intervals that contained more than one eigenvalue after
    // the first processing step
    float *lambda_mult = (float *) malloc(sizeof(float) * mat_size);
    cudaMemcpy(lambda_mult, result.g_lambda_mult,
                               sizeof(float) * mat_size,
                               cudaMemcpyDeviceToHost);
    unsigned int *pos_mult =
        (unsigned int *) malloc(sizeof(unsigned int) * mat_size);
    cudaMemcpy(pos_mult, result.g_pos_mult,
                               sizeof(unsigned int) * mat_size,
                               cudaMemcpyDeviceToHost);

    unsigned int *blocks_mult_sum =
        (unsigned int *) malloc(sizeof(unsigned int) * mat_size);
    cudaMemcpy(blocks_mult_sum, result.g_blocks_mult_sum,
                               sizeof(unsigned int) * mat_size,
                               cudaMemcpyDeviceToHost);

    unsigned int num_one_intervals;
    cudaMemcpy(&num_one_intervals, result.g_num_one,
                               sizeof(unsigned int),
                               cudaMemcpyDeviceToHost);

    unsigned int sum_blocks_mult = mat_size - num_one_intervals;


    // copy data for intervals that contained one eigenvalue after the first
    // processing step
    float *left_one = (float *) malloc(mat_size_f);
    float *right_one = (float *) malloc(mat_size_f);
    unsigned int *pos_one = (unsigned int *) malloc(mat_size_ui);
    cudaMemcpy(left_one, result.g_left_one, mat_size_f,
                               cudaMemcpyDeviceToHost);
    cudaMemcpy(right_one, result.g_right_one, mat_size_f,
                               cudaMemcpyDeviceToHost);
    cudaMemcpy(pos_one, result.g_pos_one, mat_size_ui,
                               cudaMemcpyDeviceToHost);

    // extract eigenvalues
    float *eigenvals = (float *) malloc(mat_size_f);

    // singleton intervals generated in the second step
    for (unsigned int i = 0; i < sum_blocks_mult; ++i)
    {

        eigenvals[pos_mult[i] - 1] = lambda_mult[i];
    }

    // singleton intervals generated in the first step
    unsigned int index = 0;

    for (unsigned int i = 0; i < num_one_intervals; ++i, ++index)
    {

        eigenvals[pos_one[i] - 1] = left_one[i];
    }

        // compare with reference solution

        unsigned int input_data_size = 0;

/*         char *ref_path = sdkFindFilePath("reference.dat", exec_path);
        assert(NULL != ref_path);
        sdkReadFile(ref_path, &reference, &input_data_size, false);
        assert(input_data_size == mat_size);
 */
        std::vector<float> buffer;
        buffer.reserve(input_data_size);
        std::fstream file("./data/reference.dat");
        float *reference = new float[input_data_size];
        for(size_t i = 0, temp = 0; i < mat_size; i++){
            file >> temp;
            buffer.push_back(temp);
        }
        std::copy(reference, reference + input_data_size, buffer.begin());

        // there's an imprecision of Sturm count computation which makes an
        // additional offset necessary
        float tolerance = 1.0e-5f + 5.0e-6f;

        if (sdkCompareL2fe(reference, eigenvals, mat_size, tolerance) == true)
        {
            bCompareResult = true;
        }
        else
        {
            bCompareResult = false;
        }

        free(reference);

    freePtr(eigenvals);
    freePtr(lambda_mult);
    freePtr(pos_mult);
    freePtr(blocks_mult_sum);
    freePtr(left_one);
    freePtr(right_one);
    freePtr(pos_one);

    return bCompareResult;
}
