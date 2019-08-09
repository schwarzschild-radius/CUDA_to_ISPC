#include <ISPCMacros.ispc.h>

#define SHARED_SIZE_LIMIT 1024U
#define UMUL(a, b) ((a) * (b))
#define UMAD(a, b, c) (UMUL((a), (b)) + (c))

ISPC_DEVICE_FUNCTION(void, Comparator, unsigned int *keyA, unsigned int *valA,
                     unsigned int *keyB, unsigned int *valB, unsigned int dir) {
    unsigned int t;

    if ((*keyA > *keyB) == dir) {
        t = *keyA;
        *keyA = *keyB;
        *keyB = t;
        t = *valA;
        *valA = *valB;
        *valB = t;
    }
}

ISPC_KERNEL(bitonicSortShared, uniform unsigned int DstKey[],
            uniform unsigned int DstVal[], uniform unsigned int SrcKey[],
            uniform unsigned int SrcVal[], uniform unsigned int arrayLength,
            uniform unsigned int dir) {
    uniform unsigned int s_key[SHARED_SIZE_LIMIT];
    uniform unsigned int s_val[SHARED_SIZE_LIMIT];

ISPC_GRID_START
ISPC_BLOCK_START
    unsigned int *d_SrcKey =
        SrcKey + blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    unsigned int *d_SrcVal =
        SrcVal + blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    unsigned int *d_DstKey =
        DstKey + blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    unsigned int *d_DstVal =
        DstVal + blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x + 0] = d_SrcKey[0];
    s_val[threadIdx.x + 0] = d_SrcVal[0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] =
        d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
    s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] =
        d_SrcVal[(SHARED_SIZE_LIMIT / 2)];
    ISPC_BLOCK_END
    for (unsigned int size = 2; size < arrayLength; size <<= 1) {
        for (unsigned int stride = size / 2; stride > 0; stride >>= 1) {
            ISPC_BLOCK_START
            unsigned int ddd = dir ^ ((threadIdx.x & (size / 2)) != 0);
            unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            ISPC_DEVICE_CALL(Comparator, &s_key[pos + 0], &s_val[pos + 0],
                             &s_key[pos + stride], &s_val[pos + stride], ddd);
            ISPC_BLOCK_END
        }
    }

    // ddd == dir for the last bitonic merge step
    {
        for (unsigned int stride = arrayLength / 2; stride > 0; stride >>= 1) {
            ISPC_BLOCK_START
            unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            ISPC_DEVICE_CALL(Comparator, &s_key[pos + 0], &s_val[pos + 0],
                             &s_key[pos + stride], &s_val[pos + stride], dir);
            ISPC_BLOCK_END
        }
    }

    ISPC_BLOCK_START
    unsigned int *d_SrcKey =
        SrcKey + blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    unsigned int *d_SrcVal =
        SrcVal + blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    unsigned int *d_DstKey =
        DstKey + blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    unsigned int *d_DstVal =
        DstVal + blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstKey[0] = s_key[threadIdx.x + 0];
    d_DstVal[0] = s_val[threadIdx.x + 0];
    d_DstKey[(SHARED_SIZE_LIMIT / 2)] =
        s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
    d_DstVal[(SHARED_SIZE_LIMIT / 2)] =
        s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
    ISPC_BLOCK_END
    ISPC_GRID_END
}

ISPC_KERNEL(bitonicSortShared1, uniform unsigned int DstKey[],
            uniform unsigned int DstVal[], uniform unsigned int SrcKey[],
            uniform unsigned int SrcVal[]) {
    uniform unsigned int s_key[SHARED_SIZE_LIMIT];
    uniform unsigned int s_val[SHARED_SIZE_LIMIT];

    ISPC_GRID_START
    ISPC_BLOCK_START
    // Offset to the beginning of subarray and load data
    unsigned int *d_SrcKey =
        SrcKey + blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    unsigned int *d_SrcVal =
        SrcVal + blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    unsigned int *d_DstKey =
        DstKey + blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    unsigned int *d_DstVal =
        DstVal + blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x + 0] = d_SrcKey[0];
    s_val[threadIdx.x + 0] = d_SrcVal[0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] =
        d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
    s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] =
        d_SrcVal[(SHARED_SIZE_LIMIT / 2)];
    ISPC_BLOCK_END

    for (unsigned int size = 2; size < SHARED_SIZE_LIMIT; size <<= 1) {
        // Bitonic merge

        for (unsigned int stride = size / 2; stride > 0; stride >>= 1) {
            ISPC_BLOCK_START
            unsigned int ddd = (threadIdx.x & (size / 2)) != 0;
            unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            ISPC_DEVICE_CALL(Comparator, &s_key[pos + 0], &s_val[pos + 0],
                             &s_key[pos + stride], &s_val[pos + stride], ddd);
            ISPC_BLOCK_END
        }
    }

    // Odd / even arrays of SHARED_SIZE_LIMIT elements
    // sorted in opposite directions
    unsigned int ddd = blockIdx.x & 1;
    {
        for (unsigned int stride = SHARED_SIZE_LIMIT / 2; stride > 0;
             stride >>= 1) {
            ISPC_BLOCK_START
            unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            ISPC_DEVICE_CALL(Comparator, &s_key[pos + 0], &s_val[pos + 0],
                             &s_key[pos + stride], &s_val[pos + stride], ddd);
            ISPC_BLOCK_END
        }
    }

    ISPC_BLOCK_START
    unsigned int *d_SrcKey =
        SrcKey + blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    unsigned int *d_SrcVal =
        SrcVal + blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    unsigned int *d_DstKey =
        DstKey + blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    unsigned int *d_DstVal =
        DstVal + blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstKey[0] = s_key[threadIdx.x + 0];
    d_DstVal[0] = s_val[threadIdx.x + 0];
    d_DstKey[(SHARED_SIZE_LIMIT / 2)] =
        s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
    d_DstVal[(SHARED_SIZE_LIMIT / 2)] =
        s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
    ISPC_BLOCK_END
    ISPC_GRID_END
}

ISPC_KERNEL(bitonicMergeGlobal, uniform unsigned int d_DstKey[],
            uniform unsigned int d_DstVal[], uniform unsigned int d_SrcKey[],
            uniform unsigned int d_SrcVal[], uniform unsigned int arrayLength,
            uniform unsigned int size, uniform unsigned int stride,
            uniform unsigned int dir) {
    ISPC_GRID_START
    ISPC_BLOCK_START
    unsigned int global_comparatorI = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int comparatorI = global_comparatorI & (arrayLength / 2 - 1);

    // Bitonic merge
    unsigned int ddd = dir ^ ((comparatorI & (size / 2)) != 0);
    unsigned int pos =
        2 * global_comparatorI - (global_comparatorI & (stride - 1));

    unsigned int *keyA = &d_SrcKey[pos + 0];
    unsigned int *valA = &d_SrcVal[pos + 0];
    unsigned int *keyB = &d_SrcKey[pos + stride];
    unsigned int *valB = &d_SrcVal[pos + stride];

    Comparator(gridDim, blockDim, blockIdx, threadIdx, keyA, valA, keyB, valB,
               ddd);

    d_DstKey[pos + 0] = *keyA;
    d_DstVal[pos + 0] = *valA;
    d_DstKey[pos + stride] = *keyB;
    d_DstVal[pos + stride] = *valB;
    ISPC_BLOCK_END
    ISPC_GRID_END
}

// Combined bitonic merge steps for
// size > SHARED_SIZE_LIMIT and stride = [1 .. SHARED_SIZE_LIMIT / 2]
ISPC_KERNEL(bitonicMergeShared, uniform unsigned int DstKey[],
            uniform unsigned int DstVal[], uniform unsigned int SrcKey[],
            uniform unsigned int SrcVal[], uniform unsigned int arrayLength,
            uniform unsigned int size, uniform unsigned int dir) {
    // Shared memory storage for current subarray
    uniform unsigned int s_key[SHARED_SIZE_LIMIT];
    uniform unsigned int s_val[SHARED_SIZE_LIMIT];

    ISPC_GRID_START

    ISPC_BLOCK_START
    unsigned int *d_SrcKey =
        SrcKey + blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    unsigned int *d_SrcVal =
        SrcVal + blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    unsigned int *d_DstKey =
        DstKey + blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    unsigned int *d_DstVal =
        DstVal + blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x + 0] = d_SrcKey[0];
    s_val[threadIdx.x + 0] = d_SrcVal[0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] =
        d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
    s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] =
        d_SrcVal[(SHARED_SIZE_LIMIT / 2)];
    ISPC_BLOCK_END

    for (unsigned int stride = SHARED_SIZE_LIMIT / 2; stride > 0;
         stride >>= 1) {
        ISPC_BLOCK_START
        unsigned int comparatorI =
            UMAD(blockIdx.x, blockDim.x, threadIdx.x) & ((arrayLength / 2) - 1);
        unsigned int ddd = dir ^ ((comparatorI & (size / 2)) != 0);
        unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        ISPC_DEVICE_CALL(Comparator, &s_key[pos + 0], &s_val[pos + 0],
                         &s_key[pos + stride], &s_val[pos + stride], ddd);
        ISPC_BLOCK_END
    }

    ISPC_BLOCK_START
    unsigned int *d_SrcKey =
        SrcKey + blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    unsigned int *d_SrcVal =
        SrcVal + blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    unsigned int *d_DstKey =
        DstKey + blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    unsigned int *d_DstVal =
        DstVal + blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    unsigned int comparatorI =
        UMAD(blockIdx.x, blockDim.x, threadIdx.x) & ((arrayLength / 2) - 1);
    unsigned int ddd = dir ^ ((comparatorI & (size / 2)) != 0);
    d_DstKey[0] = s_key[threadIdx.x + 0];
    d_DstVal[0] = s_val[threadIdx.x + 0];
    d_DstKey[(SHARED_SIZE_LIMIT / 2)] =
        s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
    d_DstVal[(SHARED_SIZE_LIMIT / 2)] =
        s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
    ISPC_BLOCK_END
    ISPC_GRID_END
}