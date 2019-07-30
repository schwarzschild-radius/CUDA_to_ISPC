#include "ISPCMacros.ispc"

ISPC_KERNEL(bisectKernelLarge_OneIntervals, 
                uniform float g_d[], 
                uniform float g_s[],
                uniform const unsigned int n, 
                uniform unsigned int num_intervals, 
                uniform float g_left[],
                uniform float g_right[], 
                uniform unsigned int g_pos[], 
                uniform float precision) {}