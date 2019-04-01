struct gridDim{
    unsigned int16 gridDim_z, gridDim_y, gridDim_x;
};


struct blockDim{
    unsigned int16 blockDim_z, blockDim_y, blockDim_x;
};

export void transpose_parallel_per_element(uniform gridDim grid_dim, uniform blockDim block_dim,  uniform float in[], uniform float out[]){
    for(size_t blockIdx_z = 0; grid_dim.blockIdx_z < gridDim_z; blockIdx_z++){
        for(size_t blockIdx_y = 0; grid_dim.blockIdx_y < gridDim_y; blockIdx_y++){
            for(size_t blockIdx_x = 0; grid_dim.blockIdx_x < gridDim_x; blockIdx_x++){
                for(size_t threadIdx_z = 0; block_dim.threadIdx_z < blockDim_z; threadIdx_z++){
                    for(size_t  threadIdx_y = 0; block_dim.threadIdx_y < blockDim_y; threadIdx_y++){
                        for(size_t threadIdx_x = programIndex; threadIdx_x < block_dim.blockDim_x; threadIdx_x += programCount){
                            int i = blockIdx_x * 32 + threadIdx_x;
                            int j = blockIdx_y * 32 + threadIdx_y;
                            out[j + i * N] = in[i + j * N];
                        }
                    }
                }
            }
        }
    }
}
