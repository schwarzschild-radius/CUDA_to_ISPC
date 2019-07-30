#include "config.h"
#include "util.h"
#include "bisect_util.cuh"

__device__
void writeToGmem(const unsigned int tid, const unsigned int tid_2,
                 const unsigned int num_threads_active,
                 const unsigned int num_blocks_mult,
                 float *g_left_one, float *g_right_one,
                 unsigned int *g_pos_one,
                 float *g_left_mult, float *g_right_mult,
                 unsigned int *g_left_count_mult,
                 unsigned int *g_right_count_mult,
                 float *s_left, float *s_right,
                 unsigned short *s_left_count, unsigned short *s_right_count,
                 unsigned int *g_blocks_mult,
                 unsigned int *g_blocks_mult_sum,
                 unsigned short *s_compaction_list,
                 unsigned short *s_cl_helper,
                 unsigned int offset_mult_lambda
                );
__device__
void
compactStreamsFinal(const unsigned int tid, const unsigned int tid_2,
                    const unsigned int num_threads_active,
                    unsigned int &offset_mult_lambda,
                    float *s_left, float *s_right,
                    unsigned short *s_left_count, unsigned short *s_right_count,
                    unsigned short *s_cl_one, unsigned short *s_cl_mult,
                    unsigned short *s_cl_blocking, unsigned short *s_cl_helper,
                    unsigned int is_one_lambda, unsigned int is_one_lambda_2,
                    float &left, float &right, float &left_2, float &right_2,
                    unsigned int &left_count, unsigned int &right_count,
                    unsigned int &left_count_2, unsigned int &right_count_2,
                    unsigned int c_block_iend, unsigned int c_sum_block,
                    unsigned int c_block_iend_2, unsigned int c_sum_block_2
                   );

__device__
void
scanCompactBlocksStartAddress(const unsigned int tid, const unsigned int tid_2,
                              const unsigned int num_threads_compaction,
                              unsigned short *s_cl_blocking,
                              unsigned short *s_cl_helper
                             );

__device__
void
scanSumBlocks(const unsigned int tid, const unsigned int tid_2,
              const unsigned int num_threads_active,
              const unsigned int num_threads_compaction,
              unsigned short *s_cl_blocking,
              unsigned short *s_cl_helper
             );

__device__
void
scanInitial(const unsigned int tid, const unsigned int tid_2,
            const unsigned int num_threads_active,
            const unsigned int num_threads_compaction,
            unsigned short *s_cl_one, unsigned short *s_cl_mult,
            unsigned short *s_cl_blocking, unsigned short *s_cl_helper
           );

__device__
void
storeNonEmptyIntervalsLarge(unsigned int addr,
                            const unsigned int num_threads_active,
                            float  *s_left, float *s_right,
                            unsigned short  *s_left_count,
                            unsigned short *s_right_count,
                            float left, float mid, float right,
                            const unsigned short left_count,
                            const unsigned short mid_count,
                            const unsigned short right_count,
                            float epsilon,
                            unsigned int &compact_second_chunk,
                            unsigned short *s_compaction_list,
                            unsigned int &is_active_second
                           );

__global__
void
bisectKernelLarge(float *g_d, float *g_s, const unsigned int n,
                  const float lg, const float ug,
                  const unsigned int lg_eig_count,
                  const unsigned int ug_eig_count,
                  float epsilon,
                  unsigned int *g_num_one,
                  unsigned int *g_num_blocks_mult,
                  float *g_left_one, float *g_right_one,
                  unsigned int *g_pos_one,
                  float *g_left_mult, float *g_right_mult,
                  unsigned int *g_left_count_mult,
                  unsigned int *g_right_count_mult,
                  unsigned int *g_blocks_mult,
                  unsigned int *g_blocks_mult_sum
                 );