#include "ISPCMacros.ispc"

ISPC_KERNEL(bisectKernelLarge, uniform float g_d[], uniform float g_s[],
            uniform const unsigned int n, uniform const float lg,
            uniform const float ug, uniform const unsigned int lg_eig_count,
            uniform const unsigned int ug_eig_count, uniform float epsilon,
            uniform unsigned int g_num_one[],
            uniform unsigned int g_num_blocks_mult[],
            uniform float g_left_one[], uniform float g_right_one[],
            uniform unsigned int g_pos_one[], uniform float g_left_mult[],
            uniform float g_right_mult[],
            uniform unsigned int g_left_count_mult[],
            uniform unsigned int g_right_count_mult[],
            uniform unsigned int g_blocks_mult[],
            uniform unsigned int g_blocks_mult_sum[]) {

}