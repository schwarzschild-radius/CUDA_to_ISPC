#define ISPC_GRID                                                              \
    for (blockIdx.z = 0; blockIdx.z < gridDim.z; blockIdx.z++) {               \
        for (blockIdx.y = 0; blockIdx.y < gridDim.y; blockIdx.y++) {           \
            for (blockIdx.x = 0; blockIdx.x < gridDim.x; blockIdx.x++) {

#define ISPC_BLOCK                                                             \
    for (threadIdx.z = 0; threadIdx.z < blockDim.z; threadIdx.z++) {           \
        for (threadIdx.y = 0; threadIdx.y < blockDim.y; threadIdx.y++) {       \
            for (threadIdx.x = programIndex; threadIdx.x < blockDim.x;         \
                 threadIdx.x += programCount) {

#define ISPC_START                                                             \
    ISPC_GRID                                                                  \
    ISPC_BLOCK

#define ISPC_GRID_END                                                          \
    }                                                                          \
    }                                                                          \
    }

#define ISPC_BLOCK_END                                                         \
    }                                                                          \
    }                                                                          \
    }

#define ISPC_END                                                               \
    ISPC_GRID_END                                                              \
    ISPC_BLOCK_END

#define SYNCTHREADS()                                                           \
    ISPC_BLOCK_END                                                                   \
    ISPC_BLOCK



struct Dim3{
    unsigned int x, y, z;
};

struct ConstantMemory {
    const uniform float c_ax, c_bx, c_cx, c_dx, c_ay, c_by, c_cy, c_dy, c_az,
        c_bz, c_cz, c_dz;
};

export void derivative_x_ispc(uniform const ConstantMemory &cm,
                              uniform float s_f[],const uniform Dim3 &gridDim,
                              const uniform Dim3 &blockDim, uniform float f[],
                              uniform float df[], uniform const int sPencils,
                              uniform const int mx, uniform const int my) {
                            Dim3 blockIdx, threadIdx;
ISPC_START
                            int i = threadIdx.x;
                            int j =
                                blockIdx.x * blockDim.y + threadIdx.y;
                            int k = blockIdx.y;
                            int si = i + 4; // local i for shared memory access
                                            // + halo offset
                            int sj =
                                threadIdx.y; // local j for shared memory access

                            int globalIdx = k * mx * my + j * mx + i;

                            s_f[sj * (mx + 8) + si] = f[globalIdx];
SYNCTHREADS()
                            // syncthreads

                            int i = threadIdx.x;
                            int j =
                                blockIdx.x * blockDim.y + threadIdx.y;
                            int k = blockIdx.y;
                            int si = i + 4; // local i for shared memory access
                                            // + halo offset
                            int sj =
                                threadIdx.y; // local j for shared memory access

                            int globalIdx = k * mx * my + j * mx + i;
                            // fill in periodic images in shared memory array
                            if (i < 4) {
                                s_f[sj * (mx + 8) + si - 4] =
                                    s_f[sj * (mx + 8) + si + mx - 5];
                                s_f[sj * (mx + 8) + si + mx] =
                                    s_f[sj * (mx + 8) + si + 1];
                            }

                            // syncthreadz
SYNCTHREADS()
                            int i = threadIdx.x;
                            int j =
                                blockIdx.x * blockDim.y + threadIdx.y;
                            int k = blockIdx.y;
                            int si = i + 4; // local i for shared memory access
                                            // + halo offset
                            int sj =
                                threadIdx.y; // local j for shared memory access

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
ISPC_END                        
}
export void
derivative_x_lPencils_ispc(uniform const ConstantMemory &cm,
                           uniform float s_f[], const uniform Dim3 &gridDim,
                           const uniform Dim3 &blockDim, uniform float f[],
                           uniform float df[], uniform const int lPencils,
                           uniform const int mx, uniform const int my) {
                               Dim3 blockIdx, threadIdx;
ISPC_START
                            int i = threadIdx.x;
                            int jBase = blockIdx.x * lPencils;
                            int k = blockIdx.y;
                            int si = i + 4; // local i for shared memory access
                                            // + halo offset

                            for (int sj = threadIdx.y; sj < lPencils;
                                 sj += blockDim.y) {
                                int globalIdx =
                                    k * mx * my + (jBase + sj) * mx + i;
                                s_f[sj * (mx + 8) + si] = f[globalIdx];
                            }
SYNCTHREADS()
                            int i = threadIdx.x;
                            int jBase = blockIdx.x * lPencils;
                            int k = blockIdx.y;
                            int si = i + 4;
                            // fill in periodic images in shared memory array
                            if (i < 4) {
                                for (int sj = threadIdx.y; sj < lPencils;
                                     sj += blockDim.y) {
                                    s_f[sj * (mx + 8) + si - 4] =
                                        s_f[sj * (mx + 8) + si + mx - 5];
                                    s_f[sj * (mx + 8) + si + mx] =
                                        s_f[sj * (mx + 8) + si + 1];
                                }
                            }
SYNCTHREADS()                        
                            int i = threadIdx.x;
                            int jBase = blockIdx.x * lPencils;
                            int k = blockIdx.y;
                            int si = i + 4;

                            for (int sj = threadIdx.y; sj < lPencils;
                                 sj += blockDim.y) {
                                int globalIdx =
                                    k * mx * my + (jBase + sj) * mx + i;
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
ISPC_END                    
}
export void derivative_y_ispc(uniform const ConstantMemory &cm,
                              uniform float s_f[], const uniform Dim3 &gridDim,
                              const uniform Dim3 &blockDim, uniform float f[],
                              uniform float df[], uniform const int sPencils,
                              uniform const int mx, uniform const int my) {
                                  Dim3 blockIdx, threadIdx;
ISPC_START
                            int i =
                                blockIdx.x * blockDim.x + threadIdx.x;
                            int j = threadIdx.y;
                            int k = blockIdx.y;
                            int si = threadIdx.x;
                            int sj = j + 4;

                            int globalIdx = k * mx * my + j * mx + i;

                            s_f[sj * sPencils + si] = f[globalIdx];
SYNCTHREADS()
                            int i =
                                blockIdx.x * blockDim.x + threadIdx.x;
                            int j = threadIdx.y;
                            int k = blockIdx.y;
                            int si = threadIdx.x;
                            int sj = j + 4;
                            int globalIdx = k * mx * my + j * mx + i;

                            if (j < 4) {
                                s_f[(sj - 4) * sPencils + si] =
                                    s_f[(sj + my - 5) * sPencils + si];
                                s_f[(sj + my) * sPencils + si] =
                                    s_f[(sj + 1) * sPencils + si];
                            }
SYNCTHREADS()                    
                            int i =
                                blockIdx.x * blockDim.x + threadIdx.x;
                            int j = threadIdx.y;
                            int k = blockIdx.y;
                            int si = threadIdx.x;
                            int sj = j + 4;
                            int globalIdx = k * mx * my + j * mx + i;

                            df[globalIdx] =
                                (cm.c_ay * (s_f[(sj + 1) * sPencils + si] -
                                            s_f[(sj - 1) * sPencils + si]) +
                                 cm.c_by * (s_f[(sj + 2) * sPencils + si] -
                                            s_f[(sj - 2) * sPencils + si]) +
                                 cm.c_cy * (s_f[(sj + 3) * sPencils + si] -
                                            s_f[(sj - 3) * sPencils + si]) +
                                 cm.c_dy * (s_f[(sj + 4) * sPencils + si] -
                                            s_f[(sj - 4) * sPencils + si]));
ISPC_END
}
export void
derivative_y_lPencils_ispc(uniform const ConstantMemory &cm,
                           uniform float s_f[], const uniform Dim3 &gridDim,
                           const uniform Dim3 &blockDim, uniform float f[],
                           uniform float df[], uniform const int lPencils,
                           uniform const int mx, uniform const int my) {
                               Dim3 blockIdx, threadIdx;
ISPC_START
                            int i =
                                blockIdx.x * blockDim.x + threadIdx.x;
                            int k = blockIdx.y;
                            int si = threadIdx.x;

                            for (int j = threadIdx.y; j < my;
                                 j += blockDim.y) {
                                int globalIdx = k * mx * my + j * mx + i;
                                int sj = j + 4;
                                s_f[sj * lPencils + si] = f[globalIdx];
                            }
SYNCTHREADS()                        

                            int i =
                                blockIdx.x * blockDim.x + threadIdx.x;
                            int k = blockIdx.y;
                            int si = threadIdx.x;
                            int sj = threadIdx.y + 4;
                            if (sj < 8) {
                                s_f[(sj - 4) * lPencils + si] =
                                    s_f[(sj + my - 5) * lPencils + si];
                                s_f[(sj + my) * lPencils + si] =
                                    s_f[(sj + 1) * lPencils + si];
                            }
SYNCTHREADS()                    
                            int i =
                                blockIdx.x * blockDim.x + threadIdx.x;
                            int k = blockIdx.y;
                            int si = threadIdx.x;
                            for (int j = threadIdx.y; j < my;
                                 j += blockDim.y) {
                                int globalIdx = k * mx * my + j * mx + i;
                                int sj = j + 4;
                                df[globalIdx] =
                                    (cm.c_ay * (s_f[(sj + 1) * lPencils + si] -
                                                s_f[(sj - 1) * lPencils + si]) +
                                     cm.c_by * (s_f[(sj + 2) * lPencils + si] -
                                                s_f[(sj - 2) * lPencils + si]) +
                                     cm.c_cy * (s_f[(sj + 3) * lPencils + si] -
                                                s_f[(sj - 3) * lPencils + si]) +
                                     cm.c_dy * (s_f[(sj + 4) * lPencils + si] -
                                                s_f[(sj - 4) * lPencils + si]));
                            }
ISPC_END                        
}
export void derivative_z_ispc(uniform const ConstantMemory &cm,
                              uniform float s_f[], const uniform Dim3 &gridDim,
                              const uniform Dim3 &blockDim, uniform float f[],
                              uniform float df[], uniform const int sPencils,
                              uniform const int mx, uniform const int my) {}
export void
derivative_z_lPencils_ispc(uniform const ConstantMemory &cm,
                           uniform float s_f[], const uniform Dim3 &gridDim,
                           const uniform Dim3 &blockDim, uniform float f[],
                           uniform float df[], uniform const int lPencils,
                           uniform const int mx, uniform const int my) {}