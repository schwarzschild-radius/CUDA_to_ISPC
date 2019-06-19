struct Dim3 {
    uniform int x, y, z;
};

export void reduce(uniform Dim3 &gridDim, uniform Dim3 &blockDim,
                   uniform int arr[], uniform int partial_sum[],
                   uniform unsigned int n) {
    size_t<3> blockIdx, threadIdx;
    /* Gridwise in ISPC */
    for (blockIdx.z = 0; blockIdx.z < gridDim.z; blockIdx.z++) {
        for (blockIdx.y = 0; blockIdx.y < gridDim.y; blockIdx.y++) {
            for (blockIdx.x = 0; blockIdx.x < gridDim.x; blockIdx.x++) {
                /* Blockwise in ISPC */
                for (int s = n / 2; s > 0; s >>= 1) {
                    for (threadIdx.z = 0; threadIdx.z < blockDim.z;
                         threadIdx.z++) {
                        for (threadIdx.y = 0; threadIdx.y < blockDim.y;
                             threadIdx.y++) {
                            for (threadIdx.x = programIndex;
                                 threadIdx.x < blockDim.x;
                                 threadIdx.x += programCount) {
                                size_t gid =
                                    threadIdx.x + blockDim.x * blockIdx.x;
                                if (threadIdx.x < s) {
                                    arr[gid] += arr[gid + s];
                                }
                            }
                        }
                    }
                }
                partial_sum[blockIdx.x] = arr[blockIdx.x * blockDim.x];
            }
        }
    }
}