#include "config.h"
#include "util.h"
#include "bisect_util.cuh"


__global__ void bisectKernelLarge_OneIntervals(
    float *g_d, float *g_s, const unsigned int n, unsigned int num_intervals,
    float *g_left, float *g_right, unsigned int *g_pos, float precision);