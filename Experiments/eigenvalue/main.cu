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

/* Computation of eigenvalues of symmetric, tridiagonal matrix using
 * bisection.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <string>
#include <algorithm>
#include <fstream>
#include <vector>

// includes, project
#include "config.h"
#include "structs.h"
#include "util.h"

#include "bisect_large.cuh"
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(int argc, char **argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    bool bQAResults = false;

    printf("Starting eigenvalues\n");

    bQAResults = runTest(argc, argv);
    printf("Test %s\n", bQAResults ? "Succeeded!" : "Failed!");

    exit(bQAResults ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize the input data to the algorithm
//! @param input  handles to the input data
//! @param exec_path  path where executable is run (argv[0])
//! @param mat_size  size of the matrix
//! @param user_defined  1 if the matrix size has been requested by the user,
//!                      0 if the default size
////////////////////////////////////////////////////////////////////////////////
void
initInputData(InputData &input, char *exec_path,
              const unsigned int mat_size, const unsigned int user_defined)
{
    // allocate memory
    input.a = (float *) malloc(sizeof(float) * mat_size);
    input.b = (float *) malloc(sizeof(float) * mat_size);

    if (user_defined == 1)
    {

        // initialize diagonal and superdiagonal entries with random values
        srand(278217421);

        // srand( clock());
        for (unsigned int i = 0; i < mat_size; ++i)
        {
            input.a[i] = (float)(2.0 * (((double)rand()
                                         / (double) RAND_MAX) - 0.5));
            input.b[i] = (float)(2.0 * (((double)rand()
                                         / (double) RAND_MAX) - 0.5));
        }

        // the first element of s is used as padding on the device (thus the
        // whole vector is copied to the device but the kernels are launched
        // with (s+1) as start address
        input.b[0] = 0.0f;
    }
    else
    {

        // read default matrix
        unsigned int input_data_size = mat_size;
        std::vector<float> buffer;
        buffer.reserve(input_data_size);
        std::fstream file("./data/diagonal.dat");
        for(size_t i = 0, temp = 0; i < mat_size; i++){
            file >> temp;
            buffer.push_back(temp);
        }
        input.a = new float[input_data_size];
        std::copy(input.a, input.a + input_data_size, buffer.begin());
        file.close();

        file.open("./data/superdiagonal.dat");
        for(size_t i = 0, temp = 0; i < mat_size; i++){
            file >> temp;
            buffer[i] = temp;
        }
        input.b = new float[input_data_size];
        std::copy(input.b, input.b + input_data_size, buffer.begin());
    }

    // allocate device memory for input
    cudaMalloc((void **) &(input.g_b_raw), sizeof(float) * mat_size);
    cudaMalloc((void **) &(input.g_a)    , sizeof(float) * mat_size);

    // copy data to device
    cudaMemcpy(input.g_b_raw, input.b, sizeof(float) * mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(input.g_a    , input.a, sizeof(float) * mat_size, cudaMemcpyHostToDevice);

    input.g_b = input.g_b_raw + 1;
}

////////////////////////////////////////////////////////////////////////////////
//! Clean up input data, in particular allocated memory
//! @param input  handles to the input data
////////////////////////////////////////////////////////////////////////////////
void
cleanupInputData(InputData &input)
{

    freePtr(input.a);
    freePtr(input.b);

    cudaFree(input.g_a);
    input.g_a = NULL;
    cudaFree(input.g_b_raw);
    input.g_b_raw = NULL;
    input.g_b = NULL;
}

////////////////////////////////////////////////////////////////////////////////
//! Check if a specific matrix size has to be used
//! @param argc  number of command line arguments (from main(argc, argv)
//! @param argv  pointers to command line arguments (from main(argc, argv)
//! @param matrix_size  size of matrix, updated if specific size specified on
//!                     command line
////////////////////////////////////////////////////////////////////////////////
void
getMatrixSize(int argc, char **argv,
              unsigned int &mat_size, unsigned int &user_defined)
{
/*     int temp = -1;

    if (checkCmdLineFlag(argc, (const char **)argv, "matrix-size"))
    {
        temp = getCmdLineArgumentInt(argc, (const char **) argv, "matrix-size");
    }

    if (temp > 0)
    {

        mat_size = (unsigned int) temp;
        // data type short is used in the kernel
        assert(mat_size < (1 << 16));

        user_defined = 1;
    } */

    printf("Matrix size: %i x %i\n", mat_size, mat_size);
}

////////////////////////////////////////////////////////////////////////////////
//! Check if a specific precision of the eigenvalue has to be obtained
//! @param argc  number of command line arguments (from main(argc, argv)
//! @param argv  pointers to command line arguments (from main(argc, argv)
//! @param iters_timing  numbers of iterations for timing, updated if a
//!                      specific number is specified on the command line
////////////////////////////////////////////////////////////////////////////////
void
getPrecision(int argc, char **argv, float &precision)
{

/*     float temp = -1.0f;

    if (checkCmdLineFlag(argc, (const char **)argv, "precision"))
    {
        temp = getCmdLineArgumentFloat(argc, (const char **) argv, "precision");
    }

    if (temp > 0.0f)
    {
        precision = temp;
    } */

    printf("Precision: %f\n", precision);
}

////////////////////////////////////////////////////////////////////////////////
//! Check if a particular number of iterations for timings has to be used
//! @param argc  number of command line arguments (from main(argc, argv)
//! @param argv  pointers to command line arguments (from main(argc, argv)
//! @param  iters_timing  number of timing iterations, updated if user
//!                       specific value
////////////////////////////////////////////////////////////////////////////////
void
getItersTiming(int argc, char **argv, unsigned int &iters_timing)
{

    /* int temp = -1;

    if (checkCmdLineFlag(argc, (const char **)argv, "iters-timing"))
    {
        temp = getCmdLineArgumentInt(argc, (const char **) argv, "iters-timing");
    }

    if (temp > 0)
    {
        iters_timing = temp;
    } */

    printf("Iterations to be timed: %i\n", iters_timing);
}

////////////////////////////////////////////////////////////////////////////////
//! Check if a particular filename has to be used for the file where the result
//! is stored
//! @param argc  number of command line arguments (from main(argc, argv)
//! @param argv  pointers to command line arguments (from main(argc, argv)
//! @param  filename  filename of result file, updated if user specified
//!                   filename
////////////////////////////////////////////////////////////////////////////////
void
getResultFilename(int argc, char **argv, const char *&filename)
{

    /* char *temp = NULL;
    getCmdLineArgumentString(argc, (const char **) argv, "filename-result",
                             &temp);

    if (NULL != temp)
    {

        filename = (char *) malloc(sizeof(char) * strlen(temp));
        strcpy(filename, temp);

        free(temp);
    } */

    printf("Result filename: '%s'\n", filename);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
bool
runTest(int argc, char **argv)
{
    bool bCompareResult = false;

    // default
    unsigned int mat_size = 2048;
    // flag if the matrix size is due to explicit user request
    unsigned int user_defined = 0;
    // desired precision of eigenvalues
    float  precision = 0.00001f;
    unsigned int iters_timing = 100;
    const char* result_file = "eigenvalues.dat";

    // check if there is a command line request for the matrix size
    getMatrixSize(argc, argv, mat_size, user_defined);

    // check if user requested specific precision
    getPrecision(argc, argv, precision);

    // check if user requested specific number of iterations for timing
    getItersTiming(argc, argv, iters_timing);

    // file name for result file
    getResultFilename(argc, argv, result_file);

    // set up input
    InputData input;
    initInputData(input, argv[0], mat_size, user_defined);

    // compute Gerschgorin interval
    float lg = FLT_MAX;
    float ug = -FLT_MAX;
        ResultDataLarge  result;
        initResultDataLargeMatrix(result, mat_size);

        // run the kernel
        computeEigenvaluesLargeMatrix(input, result, mat_size,
                                      precision, lg, ug,
                                      iters_timing);

        // get the result from the device and do some sanity checks
        // save the result if user specified matrix size
        bCompareResult = processResultDataLargeMatrix(input, result, mat_size, result_file,
                                                      user_defined, argv[0]);

        // cleanup
        cleanupResultDataLargeMatrix(result);
    // }

    cleanupInputData(input);
    // cudaDeviceReset();

    return bCompareResult;
}
