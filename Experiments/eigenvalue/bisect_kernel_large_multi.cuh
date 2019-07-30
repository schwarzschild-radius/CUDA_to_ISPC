#include "config.h"
#include "util.h"
#include "bisect_util.cuh"

__global__ void bisectKernelLarge_MultIntervals(
    float *g_d, float *g_s, const unsigned int n, unsigned int *blocks_mult,
    unsigned int *blocks_mult_sum, float *g_left, float *g_right,
    unsigned int *g_left_count, unsigned int *g_right_count, float *g_lambda,
    unsigned int *g_pos, float precision);