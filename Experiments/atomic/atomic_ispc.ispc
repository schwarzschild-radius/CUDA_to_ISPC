struct gridDim {
    uniform unsigned int32 gridDim_x;
    uniform unsigned int32 gridDim_y;
    uniform unsigned int32 gridDim_z;
};

struct blockDim {
    uniform unsigned int32 blockDim_x;
    uniform unsigned int32 blockDim_y;
    uniform unsigned int32 blockDim_z;
};

export void atomic_ispc(uniform gridDim& grid_dim, uniform blockDim& block_dim,
                        uniform int d_bins[], uniform const int d_in[],
                        uniform const int BIN_COUNT) {
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
                            int myId = threadIdx_x + block_dim.blockDim_x * blockIdx_x;
                            int myItem = d_in[myId];
                            int myBin = myItem % BIN_COUNT;
                            atomic_add_local(&(d_bins[myBin]), 1);
                        }
                    }
                }
            }
        }
    }
}