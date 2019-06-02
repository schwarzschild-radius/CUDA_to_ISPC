#ifndef FINITE_DIFFERENCE_CUDA_CUH
#define FINITE_DIFFERENCE_CUDA_CUH

#include <cstdio>

extern float fx, fy, fz;
extern const int mx, my, mz;

extern const int sPencils;
extern const int lPencils;

__constant__ float c_ax, c_bx, c_cx, c_dx;
__constant__ float c_ay, c_by, c_cy, c_dy;
__constant__ float c_az, c_bz, c_cz, c_dz;

extern __global__ void derivative_x(float *f, float *df);
extern __global__ void derivative_x_lPencils(float *f, float *df);
extern __global__ void derivative_y(float *f, float *df);
extern __global__ void derivative_y_lPencils(float *f, float *df);
extern __global__ void derivative_z(float *f, float *df);
extern __global__ void derivative_z_lPencils(float *f, float *df);

#endif