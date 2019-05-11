struct gridDim {
    uniform unsigned int16 gridDim_z;
    uniform unsigned int16 gridDim_y;
    uniform unsigned int16 gridDim_x;
};

struct blockDim {
    uniform unsigned int16 blockDim_z;
    uniform unsigned int16 blockDim_y;
    uniform unsigned int16 blockDim_x;
};

task void transpose_parallel_per_element_ispc_task_(
    uniform gridDim &grid_dim, uniform blockDim &block_dim, uniform int a[],
    uniform int b[], uniform unsigned int32 N, uniform unsigned int32 K) {

    size_t blockIdx_z = taskIndex2;
    size_t blockIdx_y = taskIndex1;
    size_t blockIdx_x = taskIndex0;
    for (size_t threadIdx_z = 0; threadIdx_z < block_dim.blockDim_z;
         threadIdx_z++) {
        for (size_t threadIdx_y = 0; threadIdx_y < block_dim.blockDim_y;
             threadIdx_y++) {
            for (size_t threadIdx_x = programIndex;
                 threadIdx_x < block_dim.blockDim_x;
                 threadIdx_x += programCount) {
                int i = blockIdx_x * K + threadIdx_x;
                int j = blockIdx_y * K + threadIdx_y;
                b[j + i * N] = a[i + j * N];
            }
        }
    }
}

export void transpose_parallel_per_element_ispc_task(
    uniform gridDim &grid_dim, uniform blockDim &block_dim, uniform int a[],
    uniform int b[], uniform unsigned int32 N, uniform unsigned int32 K) {
    launch[grid_dim.gridDim_x, grid_dim.gridDim_y,
           grid_dim
               .gridDim_z] transpose_parallel_per_element_ispc_task_(grid_dim,
                                                                     block_dim,
                                                                     a, b, N,
                                                                     K);
}

export void transpose_parallel_per_element_ispc(
    uniform gridDim &grid_dim, uniform blockDim &block_dim, uniform int a[],
    uniform int b[], uniform unsigned int32 N, uniform unsigned int32 K) {
    for (size_t blockIdx_z = 0; blockIdx_z < grid_dim.gridDim_z; blockIdx_z++) {
        for (size_t blockIdx_y = 0; blockIdx_y < grid_dim.gridDim_y;
             blockIdx_y++) {
            for (size_t blockIdx_x = 0; blockIdx_x < grid_dim.gridDim_x;
                 blockIdx_x++) {
                for (size_t threadIdx_z = 0; threadIdx_z < block_dim.blockDim_z;
                     threadIdx_z++) {
                    for (size_t threadIdx_y = 0;
                         threadIdx_y < block_dim.blockDim_y; threadIdx_y++) {
                        for (size_t threadIdx_x = programIndex;
                             threadIdx_x < block_dim.blockDim_x;
                             threadIdx_x += programCount) {
                            int i = blockIdx_x * K + threadIdx_x;
                            int j = blockIdx_y * K + threadIdx_y;
                            b[j + i * N] = a[i + j * N];
                        }
                    }
                }
            }
        }
    }
}