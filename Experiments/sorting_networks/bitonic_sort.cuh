#include <cuda_runtime.h>

__global__ void bitonicSortShared(unsigned int *d_DstKey,
                                  unsigned int *d_DstVal,
                                  unsigned int *d_SrcKey,
                                  unsigned int *d_SrcVal,
                                  unsigned int arrayLength, unsigned int dir);

__global__ void bitonicSortShared1(unsigned int *d_DstKey,
                                   unsigned int *d_DstVal,
                                   unsigned int *d_SrcKey,
                                   unsigned int *d_SrcVal);

__global__ void bitonicMergeGlobal(unsigned int *d_DstKey,
                                   unsigned int *d_DstVal,
                                   unsigned int *d_SrcKey,
                                   unsigned int *d_SrcVal,
                                   unsigned int arrayLength, unsigned int size,
                                   unsigned int stride, unsigned int dir);

__global__ void bitonicMergeShared(unsigned int *d_DstKey,
                                   unsigned int *d_DstVal,
                                   unsigned int *d_SrcKey,
                                   unsigned int *d_SrcVal,
                                   unsigned int arrayLength, unsigned int size,
                                   unsigned int dir);