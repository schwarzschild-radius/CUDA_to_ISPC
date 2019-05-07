struct gridDim{
    uniform unsigned int16 gridDim_z;
    uniform unsigned int16 gridDim_y;
    uniform unsigned int16 gridDim_x;
};


struct blockDim{
    uniform unsigned int16 blockDim_z;
    uniform unsigned int16 blockDim_y; 
    uniform unsigned int16 blockDim_x;
};

task void transpose_actual(uniform gridDim& grid_dim, uniform blockDim& block_dim, uniform float a[], uniform float b[]){

    size_t blockIdx_z = taskIndex2;
    size_t blockIdx_y = taskIndex1;
    size_t blockIdx_x = taskIndex0;
	for(size_t threadIdx_z = 0; threadIdx_z < block_dim.blockDim_z; threadIdx_z++){
		for(size_t  threadIdx_y = 0; threadIdx_y < block_dim.blockDim_y; threadIdx_y++){
			for(size_t threadIdx_x = programIndex; threadIdx_x < block_dim.blockDim_x; threadIdx_x += programCount){
                            int i = blockIdx_x * 32 + threadIdx_x;
                            int j = blockIdx_y * 32 + threadIdx_y;
                            b[j + i * 1024] = a[i + j * 1024];
                        }
                    }
                }
}

export void transpose_parallel_per_element_ispc_task(uniform gridDim& grid_dim, 
                                           uniform blockDim& block_dim,  
                                           uniform float a[], 
                                           uniform float b[])
{
	launch [grid_dim.gridDim_x, grid_dim.gridDim_y, grid_dim.gridDim_z] transpose_actual(grid_dim, block_dim, a, b);
}

export void transpose_parallel_per_element_ispc(uniform gridDim& grid_dim, 
                                           uniform blockDim& block_dim,  
                                           uniform float a[], 
                                           uniform float b[])
{
    for(size_t blockIdx_z = 0; blockIdx_z < grid_dim.gridDim_z; blockIdx_z++){
        for(size_t blockIdx_y = 0; blockIdx_y < grid_dim.gridDim_y; blockIdx_y++){
            for(size_t blockIdx_x = 0; blockIdx_x < grid_dim.gridDim_x; blockIdx_x++){
                for(size_t threadIdx_z = 0; threadIdx_z < block_dim.blockDim_z; threadIdx_z++){
                    for(size_t  threadIdx_y = 0; threadIdx_y < block_dim.blockDim_y; threadIdx_y++){
                        for(size_t threadIdx_x = programIndex; threadIdx_x < block_dim.blockDim_x; threadIdx_x += programCount){
                            int i = blockIdx_x * 32 + threadIdx_x;
                            int j = blockIdx_y * 32 + threadIdx_y;
                            b[j + i * 1024] = a[i + j * 1024];
                        }
                    }
                }
            }
        }
    }
}


