#ifndef _BISECT_UTIL_CUH
#define _BISECT_UTIL_CUH

#include "config.h"
#include "util.h"

__device__  int floorPow2(int n);

__device__  int ceilPow2(int n);

__device__  float computeMidpoint(const float left, const float right);

template <class S, class T>
__device__ void storeInterval(unsigned int addr, float *s_left, float *s_right,
                              T *s_left_count, T *s_right_count, float left,
                              float right, S left_count, S right_count,
                              float precision) {
    s_left_count[addr] = left_count;
    s_right_count[addr] = right_count;

    // check if interval converged
    float t0 = abs(right - left);
    float t1 = max(abs(left), abs(right)) * precision;

    if (t0 <= max(MIN_ABS_INTERVAL, t1)) {

        // compute mid point
        float lambda = computeMidpoint(left, right);

        // mark as converged
        s_left[addr] = lambda;
        s_right[addr] = lambda;
    } else {

        // store current limits
        s_left[addr] = left;
        s_right[addr] = right;
    }
}

__device__  unsigned int
computeNumSmallerEigenvals(float *g_d, float *g_s, const unsigned int n,
                           const float x, const unsigned int tid,
                           const unsigned int num_intervals_active, float *s_d,
                           float *s_s, unsigned int converged);

__device__  unsigned int
computeNumSmallerEigenvalsLarge(float *g_d, float *g_s, const unsigned int n,
                                const float x, const unsigned int tid,
                                const unsigned int num_intervals_active,
                                float *s_d, float *s_s, unsigned int converged);

template <class S, class T>
__device__ void storeNonEmptyIntervals(
    unsigned int addr, const unsigned int num_threads_active, float *s_left,
    float *s_right, T *s_left_count, T *s_right_count, float left, float mid,
    float right, const S left_count, const S mid_count, const S right_count,
    float precision, unsigned int &compact_second_chunk,
    T *s_compaction_list_exc, unsigned int &is_active_second) {
    // check if both child intervals are valid
    if ((left_count != mid_count) && (mid_count != right_count)) {

        // store the left interval
        storeInterval(addr, s_left, s_right, s_left_count, s_right_count, left,
                      mid, left_count, mid_count, precision);

        // mark that a second interval has been generated, only stored after
        // stream compaction of second chunk
        is_active_second = 1;
        s_compaction_list_exc[threadIdx.x] = 1;
        compact_second_chunk = 1;
    } else {

        // only one non-empty child interval

        // mark that no second child
        is_active_second = 0;
        s_compaction_list_exc[threadIdx.x] = 0;

        // store the one valid child interval
        if (left_count != mid_count) {
            storeInterval(addr, s_left, s_right, s_left_count, s_right_count,
                          left, mid, left_count, mid_count, precision);
        } else {
            storeInterval(addr, s_left, s_right, s_left_count, s_right_count,
                          mid, right, mid_count, right_count, precision);
        }
    }
}

template <class T>
__device__ void createIndicesCompaction(T *s_compaction_list_exc,
                                        unsigned int num_threads_compaction) {

    unsigned int offset = 1;
    const unsigned int tid = threadIdx.x;

    // higher levels of scan tree
    for (int d = (num_threads_compaction >> 1); d > 0; d >>= 1) {

        __syncthreads();

        if (tid < d) {

            unsigned int ai = offset * (2 * tid + 1) - 1;
            unsigned int bi = offset * (2 * tid + 2) - 1;

            s_compaction_list_exc[bi] =
                s_compaction_list_exc[bi] + s_compaction_list_exc[ai];
        }

        offset <<= 1;
    }

    // traverse down tree: first down to level 2 across
    for (int d = 2; d < num_threads_compaction; d <<= 1) {

        offset >>= 1;
        __syncthreads();

        if (tid < (d - 1)) {

            unsigned int ai = offset * (tid + 1) - 1;
            unsigned int bi = ai + (offset >> 1);

            s_compaction_list_exc[bi] =
                s_compaction_list_exc[bi] + s_compaction_list_exc[ai];
        }
    }

    __syncthreads();
}

template <class T>
__device__ void compactIntervals(float *s_left, float *s_right, T *s_left_count,
                                 T *s_right_count, float mid, float right,
                                 unsigned int mid_count,
                                 unsigned int right_count, T *s_compaction_list,
                                 unsigned int num_threads_active,
                                 unsigned int is_active_second) {
    const unsigned int tid = threadIdx.x;

    // perform compaction / copy data for all threads where the second
    // child is not dead
    if ((tid < num_threads_active) && (1 == is_active_second)) {
        unsigned int addr_w = num_threads_active + s_compaction_list[tid];

        s_left[addr_w] = mid;
        s_right[addr_w] = right;
        s_left_count[addr_w] = mid_count;
        s_right_count[addr_w] = right_count;
    }
}

template <class T, class S>
__device__ void storeIntervalConverged(float *s_left, float *s_right,
                                       T *s_left_count, T *s_right_count,
                                       float &left, float &mid, float &right,
                                       S &left_count, S &mid_count,
                                       S &right_count, T *s_compaction_list_exc,
                                       unsigned int &compact_second_chunk,
                                       const unsigned int num_threads_active) {
    const unsigned int tid = threadIdx.x;
    const unsigned int multiplicity = right_count - left_count;

    // check multiplicity of eigenvalue
    if (1 == multiplicity) {

        // just re-store intervals, simple eigenvalue
        s_left[tid] = left;
        s_right[tid] = right;
        s_left_count[tid] = left_count;
        s_right_count[tid] = right_count;

        // mark that no second child / clear
        s_right_count[tid + num_threads_active] = 0;
        s_compaction_list_exc[tid] = 0;
    } else {

        // number of eigenvalues after the split less than mid
        mid_count = left_count + (multiplicity >> 1);

        // store left interval
        s_left[tid] = left;
        s_right[tid] = right;
        s_left_count[tid] = left_count;
        s_right_count[tid] = mid_count;

        mid = left;

        // mark that second child interval exists
        s_right_count[tid + num_threads_active] = right_count;
        s_compaction_list_exc[tid] = 1;
        compact_second_chunk = 1;
    }
}

template <class T, class S>
__device__ void storeIntervalConverged(float *s_left, float *s_right,
                                       T *s_left_count, T *s_right_count,
                                       float &left, float &mid, float &right,
                                       S &left_count, S &mid_count,
                                       S &right_count, T *s_compaction_list_exc,
                                       unsigned int &compact_second_chunk,
                                       const unsigned int num_threads_active,
                                       unsigned int &is_active_second) {
    const unsigned int tid = threadIdx.x;
    const unsigned int multiplicity = right_count - left_count;

    // check multiplicity of eigenvalue
    if (1 == multiplicity) {

        // just re-store intervals, simple eigenvalue
        s_left[tid] = left;
        s_right[tid] = right;
        s_left_count[tid] = left_count;
        s_right_count[tid] = right_count;

        // mark that no second child / clear
        is_active_second = 0;
        s_compaction_list_exc[tid] = 0;
    } else {

        // number of eigenvalues after the split less than mid
        mid_count = left_count + (multiplicity >> 1);

        // store left interval
        s_left[tid] = left;
        s_right[tid] = right;
        s_left_count[tid] = left_count;
        s_right_count[tid] = mid_count;

        mid = left;

        // mark that second child interval exists
        is_active_second = 1;
        s_compaction_list_exc[tid] = 1;
        compact_second_chunk = 1;
    }
}

template <class T>
__device__ void subdivideActiveInterval(
    const unsigned int tid, float *s_left, float *s_right, T *s_left_count,
    T *s_right_count, const unsigned int num_threads_active, float &left,
    float &right, unsigned int &left_count, unsigned int &right_count,
    float &mid, unsigned int &all_threads_converged) {
    // for all active threads
    if (tid < num_threads_active) {

        left = s_left[tid];
        right = s_right[tid];
        left_count = s_left_count[tid];
        right_count = s_right_count[tid];

        // check if thread already converged
        if (left != right) {

            mid = computeMidpoint(left, right);
            all_threads_converged = 0;
        } else if ((right_count - left_count) > 1) {
            // mark as not converged if multiple eigenvalues enclosed
            // duplicate interval in storeIntervalsConverged()
            all_threads_converged = 0;
        }

    } // end for all active threads
}

#endif