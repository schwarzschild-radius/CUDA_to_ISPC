#include "bitonic_sort.cuh"

#define SHARED_SIZE_LIMIT 1024U
#define    UMUL(a, b) __umul24((a), (b))
#define UMAD(a, b, c) ( UMUL((a), (b)) + (c) )

__device__ inline void Comparator(
    unsigned int &keyA,
    unsigned int &valA,
    unsigned int &keyB,
    unsigned int &valB,
    unsigned int dir
)
{
    unsigned int t;

    if ((keyA > keyB) == dir)
    {
        t = keyA;
        keyA = keyB;
        keyB = t;
        t = valA;
        valA = valB;
        valB = t;
    }
}

__global__ void bitonicSortShared(unsigned int *d_DstKey,
                                  unsigned int *d_DstVal,
                                  unsigned int *d_SrcKey,
                                  unsigned int *d_SrcVal,
                                  unsigned int arrayLength, unsigned int dir) {
    // Shared memory storage for one or more short vectors
    __shared__ unsigned int s_key[SHARED_SIZE_LIMIT];
    __shared__ unsigned int s_val[SHARED_SIZE_LIMIT];

    // Offset to the beginning of subbatch and load data
    d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x + 0] = d_SrcKey[0];
    s_val[threadIdx.x + 0] = d_SrcVal[0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] =
        d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
    s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] =
        d_SrcVal[(SHARED_SIZE_LIMIT / 2)];

    for (unsigned int size = 2; size < arrayLength; size <<= 1) {
        // Bitonic merge
        unsigned int ddd = dir ^ ((threadIdx.x & (size / 2)) != 0);

        for (unsigned int stride = size / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride],
                       s_val[pos + stride], ddd);
        }
    }

    // ddd == dir for the last bitonic merge step
    {
        for (unsigned int stride = arrayLength / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride],
                       s_val[pos + stride], dir);
        }
    }

    __syncthreads();
    d_DstKey[0] = s_key[threadIdx.x + 0];
    d_DstVal[0] = s_val[threadIdx.x + 0];
    d_DstKey[(SHARED_SIZE_LIMIT / 2)] =
        s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
    d_DstVal[(SHARED_SIZE_LIMIT / 2)] =
        s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

__global__ void bitonicSortShared1(unsigned int *d_DstKey,
                                   unsigned int *d_DstVal,
                                   unsigned int *d_SrcKey,
                                   unsigned int *d_SrcVal) {
    // Shared memory storage for current subarray
    __shared__ unsigned int s_key[SHARED_SIZE_LIMIT];
    __shared__ unsigned int s_val[SHARED_SIZE_LIMIT];

    // Offset to the beginning of subarray and load data
    d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x + 0] = d_SrcKey[0];
    s_val[threadIdx.x + 0] = d_SrcVal[0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] =
        d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
    s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] =
        d_SrcVal[(SHARED_SIZE_LIMIT / 2)];

    for (unsigned int size = 2; size < SHARED_SIZE_LIMIT; size <<= 1) {
        // Bitonic merge
        unsigned int ddd = (threadIdx.x & (size / 2)) != 0;

        for (unsigned int stride = size / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride],
                       s_val[pos + stride], ddd);
        }
    }

    // Odd / even arrays of SHARED_SIZE_LIMIT elements
    // sorted in opposite directions
    unsigned int ddd = blockIdx.x & 1;
    {
        for (unsigned int stride = SHARED_SIZE_LIMIT / 2; stride > 0;
             stride >>= 1) {
            __syncthreads();
            unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride],
                       s_val[pos + stride], ddd);
        }
    }

    __syncthreads();
    d_DstKey[0] = s_key[threadIdx.x + 0];
    d_DstVal[0] = s_val[threadIdx.x + 0];
    d_DstKey[(SHARED_SIZE_LIMIT / 2)] =
        s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
    d_DstVal[(SHARED_SIZE_LIMIT / 2)] =
        s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

__global__ void bitonicMergeGlobal(unsigned int *d_DstKey,
                                   unsigned int *d_DstVal,
                                   unsigned int *d_SrcKey,
                                   unsigned int *d_SrcVal,
                                   unsigned int arrayLength, unsigned int size,
                                   unsigned int stride, unsigned int dir) {
    unsigned int global_comparatorI = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int comparatorI = global_comparatorI & (arrayLength / 2 - 1);

    // Bitonic merge
    unsigned int ddd = dir ^ ((comparatorI & (size / 2)) != 0);
    unsigned int pos =
        2 * global_comparatorI - (global_comparatorI & (stride - 1));

    unsigned int keyA = d_SrcKey[pos + 0];
    unsigned int valA = d_SrcVal[pos + 0];
    unsigned int keyB = d_SrcKey[pos + stride];
    unsigned int valB = d_SrcVal[pos + stride];

    Comparator(keyA, valA, keyB, valB, ddd);

    d_DstKey[pos + 0] = keyA;
    d_DstVal[pos + 0] = valA;
    d_DstKey[pos + stride] = keyB;
    d_DstVal[pos + stride] = valB;
}

// Combined bitonic merge steps for
// size > SHARED_SIZE_LIMIT and stride = [1 .. SHARED_SIZE_LIMIT / 2]
__global__ void bitonicMergeShared(unsigned int *d_DstKey,
                                   unsigned int *d_DstVal,
                                   unsigned int *d_SrcKey,
                                   unsigned int *d_SrcVal,
                                   unsigned int arrayLength, unsigned int size,
                                   unsigned int dir) {
    // Shared memory storage for current subarray
    __shared__ unsigned int s_key[SHARED_SIZE_LIMIT];
    __shared__ unsigned int s_val[SHARED_SIZE_LIMIT];

    d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x + 0] = d_SrcKey[0];
    s_val[threadIdx.x + 0] = d_SrcVal[0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] =
        d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
    s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] =
        d_SrcVal[(SHARED_SIZE_LIMIT / 2)];

    // Bitonic merge
    unsigned int comparatorI =
        UMAD(blockIdx.x, blockDim.x, threadIdx.x) & ((arrayLength / 2) - 1);
    unsigned int ddd = dir ^ ((comparatorI & (size / 2)) != 0);

    for (unsigned int stride = SHARED_SIZE_LIMIT / 2; stride > 0;
         stride >>= 1) {
        __syncthreads();
        unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride],
                   s_val[pos + stride], ddd);
    }

    __syncthreads();
    d_DstKey[0] = s_key[threadIdx.x + 0];
    d_DstVal[0] = s_val[threadIdx.x + 0];
    d_DstKey[(SHARED_SIZE_LIMIT / 2)] =
        s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
    d_DstVal[(SHARED_SIZE_LIMIT / 2)] =
        s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}