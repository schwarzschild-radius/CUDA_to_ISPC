#include "ISPCMacros.ispc"
#include "bisect_util.ispc"
#include "config.ispc"

ISPC_KERNEL(bisectKernelLarge_MultIntervals, uniform float g_d[],
            uniform float g_s[], uniform const unsigned int n,
            uniform unsigned int blocks_mult[],
            uniform unsigned int blocks_mult_sum[], uniform float g_left[],
            uniform float g_right[], uniform unsigned int g_left_count[],
            uniform unsigned int g_right_count[], uniform float g_lambda[],
            uniform unsigned int g_pos[], uniform float precision) {

    // shared memory area
    uniform float s_left[2 * MAX_THREADS_BLOCK];
    uniform float s_right[2 * MAX_THREADS_BLOCK];
    uniform unsigned int s_left_count[2 * MAX_THREADS_BLOCK];
    uniform unsigned int s_right_count[2 * MAX_THREADS_BLOCK];
    uniform unsigned int s_compaction_list[2 * MAX_THREADS_BLOCK + 1];
    uniform unsigned int all_threads_converged;
    uniform unsigned int num_threads_active;
    uniform unsigned int num_threads_compaction;
    uniform unsigned int compact_second_chunk;
    uniform unsigned int c_block_start;
    uniform unsigned int c_block_end;
    uniform unsigned int c_block_offset_output;

    // context
    float mid = 0.0f;
    unsigned int mid_count = 0;
    float left;
    float right;
    unsigned int left_count;
    unsigned int right_count;
    unsigned int is_active_second = 0;
    ISPC_GRID_START

    ISPC_BLOCK_START
    const unsigned int tid = threadIdx.x;
    if (tid == 0) {

        c_block_start = blocks_mult[blockIdx.x];
        c_block_end = blocks_mult[blockIdx.x + 1];
        c_block_offset_output = blocks_mult_sum[blockIdx.x];

        num_threads_active = c_block_end - c_block_start;
        s_compaction_list[0] = 0;
        num_threads_compaction = ceilPow2(num_threads_active);

        all_threads_converged = 1;
        compact_second_chunk = 0;
    }

    SYNCTHREADS()
    const unsigned int tid = threadIdx.x;
    if (tid < num_threads_active) {
        s_left[tid] = g_left[c_block_start + tid];
        s_right[tid] = g_right[c_block_start + tid];
        s_left_count[tid] = g_left_count[c_block_start + tid];
        s_right_count[tid] = g_right_count[c_block_start + tid];
    }

    ISPC_BLOCK_END

    while (true) {
ISPC_BLOCK_START
        const unsigned int tid = threadIdx.x;
        ISPC_DEVICE_CALL(subdivideActiveInterval(unsigned_int), tid, s_left,
                         s_right, s_left_count, s_right_count,
                         num_threads_active, left, right, left_count,
                         right_count, mid, all_threads_converged);
ISPC_BLOCK_END

        if (all_thread_converged == 1) {
            break;
        }

        /* mid_count = computeNumSmallerEigenvalsLarge(g_d, g_s, n, mid, tid,
                                                    num_threads_active, s_left,
                                                    s_right, (left == right)); */
        // inline
        float delta = 1.0f;
        unsigned int count = 0;

        unsigned int rem = n;

        // do until whole diagonal and superdiagonal has been loaded and
        // processed
        for (unsigned int i = 0; i < n; i += blockDim.x) {
ISPC_BLOCK_START

            // read new chunk of data into shared memory
            if ((i + threadIdx.x) < n) {

                s_left[threadIdx.x] = *(g_d + i + threadIdx.x);
                s_right[threadIdx.x] = *(g_s + i + threadIdx.x - 1);
            }

SYNCTHREADS()

            if (tid < num_threads_active) {

                // perform (optimized) Gaussian elimination to determine the
                // number of eigenvalues that are smaller than n
                for (unsigned int k = 0; k < min(rem, blockDim.x); ++k) {
                    delta = s_left[k] - mid - (s_right[k] * s_right[k]) / delta;
                    count += (delta < 0) ? 1 : 0;
                }

            } // end if thread currently processing an interval

            rem -= blockDim.x;
ISPC_BLOCK_END            
        }

        mid = count;

        // inline here

        SYNCTHREADS()
        const unsigned int tid = threadIdx.x;
        if (tid < num_threads_active) {

            // store intervals
            if (left != right) {

                storeNonEmptyIntervals(
                    tid, num_threads_active, s_left, s_right, s_left_count,
                    s_right_count, left, mid, right, left_count, mid_count,
                    right_count, precision, compact_second_chunk,
                    s_compaction_list_exc, is_active_second);
            } else {

                storeIntervalConverged(
                    s_left, s_right, s_left_count, s_right_count, left, mid,
                    right, left_count, mid_count, right_count,
                    s_compaction_list_exc, compact_second_chunk,
                    num_threads_active, is_active_second);
            }
        }

        SYNCTHREADS()
        const unsigned int tid = threadIdx.x;
        if (compact_second_chunk == 1) {

            createIndicesCompaction(s_compaction_list_exc,
                                    num_threads_compaction);

            compactIntervals(s_left, s_right, s_left_count, s_right_count, mid,
                             right, mid_count, right_count, s_compaction_list,
                             num_threads_active, is_active_second);
        }

        if (0 == tid) {
            num_threads_active += s_compaction_list[num_threads_active];
            num_threads_compaction = ceilPow2(num_threads_active);

            compact_second_chunk = 0;
            all_threads_converged = 1;
        }

        SYNCTHREADS()
        const unsigned int tid = threadIdx.x;
        s_compaction_list_exc[threadIdx.x] = 0;
        s_compaction_list_exc[threadIdx.x + blockDim.x] = 0;

        ISPC_BLOCK_END
    }

    ISPC_BLOCK_START
    const unsigned int tid = threadIdx.x;
    if (tid < num_threads_active) {

        unsigned int addr = c_block_offset_output + tid;

        g_lambda[addr] = s_left[tid];
        g_pos[addr] = s_right_count[tid];
    }

    ISPC_BLOCK_END
    ISPC_GRID_END
}