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

struct ConstantMemory {
    const uniform float c_ax, c_bx, c_cx, c_dx, c_ay, c_by, c_cy, c_dy, c_az,
        c_bz, c_cz, c_dz;
};

export void derivative_x_ispc(uniform const ConstantMemory &cm,
                              uniform float s_f[], uniform gridDim &grid_dim,
                              uniform blockDim &block_dim, uniform float f[],
                              uniform float df[], uniform const int sPencils,
                              uniform const int mx, uniform const int my) {
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
                            int i = threadIdx_x;
                            int j =
                                blockIdx_x * block_dim.blockDim_y + threadIdx_y;
                            int k = blockIdx_y;
                            int si = i + 4; // local i for shared memory access
                                            // + halo offset
                            int sj =
                                threadIdx_y; // local j for shared memory access

                            int globalIdx = k * mx * my + j * mx + i;

                            s_f[sj * (mx + 8) + si] = f[globalIdx];

                            // syncthreads
                        }
                        for (size_t threadIdx_x = programIndex;
                             threadIdx_x < block_dim.blockDim_x;
                             threadIdx_x += programCount) {

                            int i = threadIdx_x;
                            int j =
                                blockIdx_x * block_dim.blockDim_y + threadIdx_y;
                            int k = blockIdx_y;
                            int si = i + 4; // local i for shared memory access
                                            // + halo offset
                            int sj =
                                threadIdx_y; // local j for shared memory access

                            int globalIdx = k * mx * my + j * mx + i;
                            // fill in periodic images in shared memory array
                            if (i < 4) {
                                s_f[sj * (mx + 8) + si - 4] =
                                    s_f[sj * (mx + 8) + si + mx - 5];
                                s_f[sj * (mx + 8) + si + mx] =
                                    s_f[sj * (mx + 8) + si + 1];
                            }

                            // syncthreads
                        }
                        for (size_t threadIdx_x = programIndex;
                             threadIdx_x < block_dim.blockDim_x;
                             threadIdx_x += programCount) {
                            int i = threadIdx_x;
                            int j =
                                blockIdx_x * block_dim.blockDim_y + threadIdx_y;
                            int k = blockIdx_y;
                            int si = i + 4; // local i for shared memory access
                                            // + halo offset
                            int sj =
                                threadIdx_y; // local j for shared memory access

                            int globalIdx = k * mx * my + j * mx + i;

                            df[globalIdx] =
                                (cm.c_ax * (s_f[sj * (mx + 8) + si + 1] -
                                            s_f[sj * (mx + 8) + si - 1]) +
                                 cm.c_bx * (s_f[sj * (mx + 8) + si + 2] -
                                            s_f[sj * (mx + 8) + si - 2]) +
                                 cm.c_cx * (s_f[sj * (mx + 8) + si + 3] -
                                            s_f[sj * (mx + 8) + si - 3]) +
                                 cm.c_dx * (s_f[sj * (mx + 8) + si + 4] -
                                            s_f[sj * (mx + 8) + si - 4]));
                        }
                    }
                }
            }
        }
    }
}
export void derivative_x_lPencils_ispc(uniform float f[], uniform float df[]) {}
export void derivative_y_ispc(uniform float f[], uniform float df[]) {}
export void derivative_y_lPencils_ispc(uniform float f[], uniform float df[]) {}
export void derivative_z_ispc(uniform float f[], uniform float df[]) {}
export void derivative_z_lPencils_ispc(uniform float f[], uniform float df[]) {}