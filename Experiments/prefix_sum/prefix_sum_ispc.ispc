struct Dim3 {
    uniform int x, y, z;
};

export void prefix_sum_ispc(uniform Dim3 &gridDim, uniform Dim3 &blockDim,
                            uniform int a[], uniform size_t N) {
    int<3> blockIdx, threadIdx;

    for (blockIdx.z = 0; blockIdx.z < gridDim.z; blockIdx.z++) {
        for (blockIdx.y = 0; blockIdx.y < gridDim.y; blockIdx.y++) {
            for (blockIdx.x = 0; blockIdx.x < gridDim.x; blockIdx.x++) {
                int tid = threadIdx.x;
                int i = 0;
                for (i = 2; i <= N; i *= 2) {
                    for (threadIdx.z = 0; threadIdx.z < blockDim.z;
                         threadIdx.z++) {
                        for (threadIdx.y = 0; threadIdx.y < blockDim.y;
                             threadIdx.y++) {
                            for (threadIdx.x = programIndex;
                                 threadIdx.x < blockDim.x;
                                 threadIdx.x += programCount) {
                                if (((i - tid % i) == 1) && tid != 0) {
                                    a[tid] = a[tid] + a[tid - i / 2];
                                }
                            }
                        }
                    }
                }

                for (threadIdx.z = 0; threadIdx.z < blockDim.z; threadIdx.z++) {
                    for (threadIdx.y = 0; threadIdx.y < blockDim.y;
                         threadIdx.y++) {
                        for (threadIdx.x = programIndex;
                             threadIdx.x < blockDim.x;
                             threadIdx.x += programCount) {
                            tid = threadIdx.x;
                            if (tid == N - 1) {
                                a[tid] = 0;
                            }
                        }
                    }
                }

                for (; i > 1; i /= 2) {
                    for (threadIdx.z = 0; threadIdx.z < blockDim.z;
                         threadIdx.z++) {
                        for (threadIdx.y = 0; threadIdx.y < blockDim.y;
                             threadIdx.y++) {
                            for (threadIdx.x = programIndex;
                                 threadIdx.x < blockDim.x;
                                 threadIdx.x += programCount) {
                                tid = threadIdx.x;
                                if (((i - tid % i) == 1) && tid != 0) {
                                    int temp = a[tid - i / 2];
                                    a[tid - i / 2] = a[tid];
                                    a[tid] = a[tid] + temp;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}