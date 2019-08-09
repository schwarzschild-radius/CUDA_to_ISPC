#include <cassert>
#include <ctime>
#include <iostream>
#include <vector>

#include "bitonic_sort.cuh"
#include "bitonic_sort.h"
#include "cuda_utils.cuh"

#define SHARED_SIZE_LIMIT 1024U

unsigned int factorRadix2(unsigned int *log2L, unsigned int L) {
    if (!L) {
        *log2L = 0;
        return 0;
    } else {
        for (*log2L = 0; (L & 1) == 0; L >>= 1, *log2L++)
            ;

        return L;
    }
}

unsigned int bitonicSort(unsigned int *d_DstKey, unsigned int *d_DstVal,
                         unsigned int *d_SrcKey, unsigned int *d_SrcVal,
                         unsigned int batchSize, unsigned int arrayLength,
                         unsigned int dir) {
    if (arrayLength < 2)
        return 0;

    unsigned int log2L;
    unsigned int factorizationRemainder = factorRadix2(&log2L, arrayLength);
    assert(factorizationRemainder == 1);

    dir = (dir != 0);

    unsigned int blockCount = batchSize * arrayLength / SHARED_SIZE_LIMIT;
    unsigned int threadCount = SHARED_SIZE_LIMIT / 2;

    if (arrayLength <= SHARED_SIZE_LIMIT) {
        assert((batchSize * arrayLength) % SHARED_SIZE_LIMIT == 0);
        bitonicSortShared<<<blockCount, threadCount>>>(
            d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, arrayLength, dir);
    } else {
        bitonicSortShared1<<<blockCount, threadCount>>>(d_DstKey, d_DstVal,
                                                        d_SrcKey, d_SrcVal);

        for (unsigned int size = 2 * SHARED_SIZE_LIMIT; size <= arrayLength;
             size <<= 1)
            for (unsigned stride = size / 2; stride > 0; stride >>= 1)
                if (stride >= SHARED_SIZE_LIMIT) {
                    bitonicMergeGlobal<<<(batchSize * arrayLength) / 512,
                                         256>>>(d_DstKey, d_DstVal, d_DstKey,
                                                d_DstVal, arrayLength, size,
                                                stride, dir);
                } else {
                    bitonicMergeShared<<<blockCount, threadCount>>>(
                        d_DstKey, d_DstVal, d_DstKey, d_DstVal, arrayLength,
                        size, dir);
                    break;
                }
    }

    return threadCount;
}

void executeCUDA(unsigned int N, unsigned int numIterations, unsigned int DIR,
                 unsigned int *h_InputKey, unsigned int *h_InputVal,
                 unsigned int *h_OutputKeyGPU, unsigned int *h_OutputValGPU) {
    unsigned int *d_InputKey, *d_InputVal, *d_OutputKey, *d_OutputVal;
    cudaCheck(cudaMalloc((void **)&d_InputKey, N * sizeof(unsigned int)));
    cudaCheck(cudaMalloc((void **)&d_InputVal, N * sizeof(unsigned int)));
    cudaCheck(cudaMalloc((void **)&d_OutputKey, N * sizeof(unsigned int)));
    cudaCheck(cudaMalloc((void **)&d_OutputVal, N * sizeof(unsigned int)));
    cudaCheck(cudaMemcpy(d_InputKey, h_InputKey, N * sizeof(unsigned int),
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_InputVal, h_InputVal, N * sizeof(unsigned int),
                         cudaMemcpyHostToDevice));

    for (unsigned int arrayLength = 64; arrayLength <= N; arrayLength *= 2) {
        printf("Testing array length %u (%u arrays per batch)...\n",
               arrayLength, N / arrayLength);
        unsigned int threadCount = 0;
        for (unsigned int i = 0; i < numIterations; i++)
            threadCount =
                bitonicSort(d_OutputKey, d_OutputVal, d_InputKey, d_InputVal,
                            N / arrayLength, arrayLength, DIR);

        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaMemcpy(h_OutputKeyGPU, d_OutputKey,
                             N * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(h_OutputValGPU, d_OutputVal,
                             N * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }
}

unsigned int bitonicSortISPC(unsigned int *d_DstKey, unsigned int *d_DstVal,
                             unsigned int *d_SrcKey, unsigned int *d_SrcVal,
                             unsigned int batchSize, unsigned int arrayLength,
                             unsigned int dir) {
    if (arrayLength < 2)
        return 0;

    unsigned int log2L;
    unsigned int factorizationRemainder = factorRadix2(&log2L, arrayLength);
    assert(factorizationRemainder == 1);

    dir = (dir != 0);

    unsigned int blockCount = batchSize * arrayLength / SHARED_SIZE_LIMIT;
    unsigned int threadCount = SHARED_SIZE_LIMIT / 2;

    if (arrayLength <= SHARED_SIZE_LIMIT) {
        assert((batchSize * arrayLength) % SHARED_SIZE_LIMIT == 0);
        ispc::bitonicSortShared({(int32_t)blockCount, 1, 1},
                                {(int32_t)threadCount, 1, 1}, 0, d_DstKey,
                                d_DstVal, d_SrcKey, d_SrcVal, arrayLength, dir);
    } else {
        ispc::bitonicSortShared1({(int32_t)blockCount, 1, 1},
                                 {(int32_t)threadCount, 1, 1}, 0, d_DstKey,
                                 d_DstVal, d_SrcKey, d_SrcVal);

        for (unsigned int size = 2 * SHARED_SIZE_LIMIT; size <= arrayLength;
             size <<= 1)
            for (unsigned stride = size / 2; stride > 0; stride >>= 1)
                if (stride >= SHARED_SIZE_LIMIT) {
                    ispc::bitonicMergeGlobal(
                        {(int32_t)(batchSize * arrayLength) / 512, 1, 1},
                        {(int32_t)256, 1, 1}, 0, d_DstKey, d_DstVal, d_DstKey,
                        d_DstVal, arrayLength, size, stride, dir);
                } else {
                    ispc::bitonicMergeShared({(int32_t)blockCount, 1, 1},
                                             {(int32_t)threadCount, 1, 1}, 0,
                                             d_DstKey, d_DstVal, d_DstKey,
                                             d_DstVal, arrayLength, size, dir);
                    break;
                }
    }

    return threadCount;
}

void executeISPC(unsigned int N, unsigned int numIterations, unsigned int DIR,
                 unsigned int *h_InputKey, unsigned int *h_InputVal,
                 unsigned int *h_OutputKeyGPU, unsigned int *h_OutputValGPU) {
    for (unsigned int arrayLength = 64; arrayLength <= N; arrayLength *= 2) {
        printf("Testing array length %u (%u arrays per batch)...\n",
               arrayLength, N / arrayLength);
        unsigned int threadCount = 0;
        for (unsigned int i = 0; i < numIterations; i++)
            threadCount =
                bitonicSortISPC(h_OutputKeyGPU, h_OutputValGPU, h_InputKey,
                                h_InputVal, N / arrayLength, arrayLength, DIR);
    }
}

unsigned int validateSortedKeys(unsigned int *resKey, unsigned int *srcKey,
                                unsigned int batchSize,
                                unsigned int arrayLength,
                                unsigned int numValues, unsigned int dir) {
    unsigned int *srcHist;
    unsigned int *resHist;

    if (arrayLength < 2) {
        printf("validateSortedKeys(): arrayLength too short, exiting...\n");
        return 1;
    }

    printf("...inspecting keys array: ");

    srcHist = (unsigned int *)malloc(numValues * sizeof(unsigned int));
    resHist = (unsigned int *)malloc(numValues * sizeof(unsigned int));

    int flag = 1;

    for (unsigned int j = 0; j < batchSize;
         j++, srcKey += arrayLength, resKey += arrayLength) {
        // Build histograms for keys arrays
        memset(srcHist, 0, numValues * sizeof(unsigned int));
        memset(resHist, 0, numValues * sizeof(unsigned int));

        for (unsigned int i = 0; i < arrayLength; i++) {
            if (srcKey[i] < numValues && resKey[i] < numValues) {
                srcHist[srcKey[i]]++;
                resHist[resKey[i]]++;
            } else {
                flag = 0;
                break;
            }
        }

        if (!flag) {
            printf("***Set %u source/result key arrays are not limited "
                   "properly***\n",
                   j);
            goto brk;
        }

        // Compare the histograms
        for (unsigned int i = 0; i < numValues; i++)
            if (srcHist[i] != resHist[i]) {
                flag = 0;
                break;
            }

        if (!flag) {
            printf("***Set %u source/result keys histograms do not match***\n",
                   j);
            goto brk;
        }

        if (dir) {
            // Ascending order
            for (unsigned int i = 0; i < arrayLength - 1; i++)
                if (resKey[i + 1] < resKey[i]) {
                    flag = 0;
                    break;
                }
        } else {
            // Descending order
            for (unsigned int i = 0; i < arrayLength - 1; i++)
                if (resKey[i + 1] > resKey[i]) {
                    flag = 0;
                    break;
                }
        }

        if (!flag) {
            printf("***Set %u result key array is not ordered properly***\n",
                   j);
            goto brk;
        }
    }

brk:
    free(resHist);
    free(srcHist);

    if (flag)
        printf("OK\n");

    return flag;
}

int validateValues(unsigned int *resKey, unsigned int *resVal,
                   unsigned int *srcKey, unsigned int batchSize,
                   unsigned int arrayLength) {
    int correctFlag = 1, stableFlag = 1;

    printf("...inspecting keys and values array: ");

    for (unsigned int i = 0; i < batchSize;
         i++, resKey += arrayLength, resVal += arrayLength) {
        for (unsigned int j = 0; j < arrayLength; j++) {
            if (resKey[j] != srcKey[resVal[j]])
                correctFlag = 0;

            if ((j < arrayLength - 1) && (resKey[j] == resKey[j + 1]) &&
                (resVal[j] > resVal[j + 1]))
                stableFlag = 0;
        }
    }

    printf(correctFlag ? "OK\n" : "***corrupted!!!***\n");
    printf(stableFlag ? "...stability property: stable!\n"
                      : "...stability property: NOT stable\n");

    return correctFlag;
}

bool checkResults(unsigned int N, unsigned int numValues, unsigned int DIR,
                  std::vector<unsigned int> ref_keys,
                  std::vector<unsigned int> cuda_keys,
                  std::vector<unsigned int> ispc_keys,
                  std::vector<unsigned int> ref_values,
                  std::vector<unsigned int> cuda_values,
                  std::vector<unsigned int> ispc_values) {
    int keysFlagCUDA = validateSortedKeys(cuda_keys.data(), ref_keys.data(), 1,
                                          N, numValues, DIR);
    int valuesFlagCUDA = validateValues(cuda_keys.data(), cuda_values.data(),
                                        ref_keys.data(), 1, N);

    int keysFlagISPC = validateSortedKeys(ispc_keys.data(), ref_keys.data(), 1,
                                          N, numValues, DIR);
    int valuesFlagISPC = validateValues(ispc_keys.data(), ispc_values.data(),
                                        ref_keys.data(), 1, N);
    return keysFlagCUDA && valuesFlagCUDA;
}

int main() {
    unsigned int *h_InputKey, *h_InputVal, *h_OutputKeyGPU, *h_OutputValGPU;
    const unsigned int N = 1048576;
    const unsigned int DIR = 0;
    const unsigned int numValues = 65536;
    const unsigned int numIterations = 1;

    std::vector<unsigned int> ref_keys(N), cuda_keys(N), ispc_keys(N);
    std::vector<unsigned int> ref_values(N), cuda_values(N), ispc_values(N);
    srand(time(NULL));

    for (unsigned int i = 0; i < N; i++) {
        ref_keys[i] = rand() % numValues;
        ref_values[i] = i;
    }

    executeCUDA(N, numIterations, DIR, ref_keys.data(), ref_values.data(),
                cuda_keys.data(), cuda_values.data());
    executeISPC(N, numIterations, DIR, ref_keys.data(), ref_values.data(),
                ispc_keys.data(), ispc_values.data());

    if (checkResults(N, numValues, DIR, ref_keys, cuda_keys, ispc_keys,
                     ref_values, cuda_values, ispc_values)) {
        return 1;
    }

    return 0;
}
