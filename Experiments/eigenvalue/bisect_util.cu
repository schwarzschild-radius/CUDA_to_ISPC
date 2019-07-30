#include "bisect_util.cuh"

// includes, project


////////////////////////////////////////////////////////////////////////////////
//! Compute the next lower power of two of n
//! @param  n  number for which next higher power of two is seeked
////////////////////////////////////////////////////////////////////////////////
__device__
 int
floorPow2(int n)
{

    // early out if already power of two
    if (0 == (n & (n-1)))
    {
        return n;
    }

    int exp;
    frexp((float)n, &exp);
    return (1 << (exp - 1));
}

////////////////////////////////////////////////////////////////////////////////
//! Compute the next higher power of two of n
//! @param  n  number for which next higher power of two is seeked
////////////////////////////////////////////////////////////////////////////////
__device__
 int
ceilPow2(int n)
{

    // early out if already power of two
    if (0 == (n & (n-1)))
    {
        return n;
    }

    int exp;
    frexp((float)n, &exp);
    return (1 << exp);
}

////////////////////////////////////////////////////////////////////////////////
//! Compute midpoint of interval [\a left, \a right] avoiding overflow if
//! possible
//! @param left   left / lower limit of interval
//! @param right  right / upper limit of interval
////////////////////////////////////////////////////////////////////////////////
__device__
 float
computeMidpoint(const float left, const float right)
{

    float mid;

    if (sign_f(left) == sign_f(right))
    {
        mid = left + (right - left) * 0.5f;
    }
    else
    {
        mid = (left + right) * 0.5f;
    }

    return mid;
}

////////////////////////////////////////////////////////////////////////////////
//! Check if interval converged and store appropriately
//! @param  addr    address where to store the information of the interval
//! @param  s_left  shared memory storage for left interval limits
//! @param  s_right  shared memory storage for right interval limits
//! @param  s_left_count  shared memory storage for number of eigenvalues less
//!                       than left interval limits
//! @param  s_right_count  shared memory storage for number of eigenvalues less
//!                       than right interval limits
//! @param  left   lower limit of interval
//! @param  right  upper limit of interval
//! @param  left_count  eigenvalues less than \a left
//! @param  right_count  eigenvalues less than \a right
//! @param  precision  desired precision for eigenvalues
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//! Compute number of eigenvalues that are smaller than x given a symmetric,
//! real, and tridiagonal matrix
//! @param  g_d  diagonal elements stored in global memory
//! @param  g_s  superdiagonal elements stored in global memory
//! @param  n    size of matrix
//! @param  x    value for which the number of eigenvalues that are smaller is
//!              seeked
//! @param  tid  thread identified (e.g. threadIdx.x or gtid)
//! @param  num_intervals_active  number of active intervals / threads that
//!                               currently process an interval
//! @param  s_d  scratch space to store diagonal entries of the tridiagonal
//!              matrix in shared memory
//! @param  s_s  scratch space to store superdiagonal entries of the tridiagonal
//!              matrix in shared memory
//! @param  converged  flag if the current thread is already converged (that
//!         is count does not have to be computed)
////////////////////////////////////////////////////////////////////////////////
__device__
 unsigned int
computeNumSmallerEigenvals(float *g_d, float *g_s, const unsigned int n,
                           const float x,
                           const unsigned int tid,
                           const unsigned int num_intervals_active,
                           float *s_d, float *s_s,
                           unsigned int converged
                          )
{

    float  delta = 1.0f;
    unsigned int count = 0;

    __syncthreads();

    // read data into shared memory
    if (threadIdx.x < n)
    {
        s_d[threadIdx.x] = *(g_d + threadIdx.x);
        s_s[threadIdx.x] = *(g_s + threadIdx.x - 1);
    }

    __syncthreads();

    // perform loop only for active threads
    if ((tid < num_intervals_active) && (0 == converged))
    {

        // perform (optimized) Gaussian elimination to determine the number
        // of eigenvalues that are smaller than n
        for (unsigned int k = 0; k < n; ++k)
        {
            delta = s_d[k] - x - (s_s[k] * s_s[k]) / delta;
            count += (delta < 0) ? 1 : 0;
        }

    }  // end if thread currently processing an interval

    return count;
}

////////////////////////////////////////////////////////////////////////////////
//! Compute number of eigenvalues that are smaller than x given a symmetric,
//! real, and tridiagonal matrix
//! @param  g_d  diagonal elements stored in global memory
//! @param  g_s  superdiagonal elements stored in global memory
//! @param  n    size of matrix
//! @param  x    value for which the number of eigenvalues that are smaller is
//!              seeked
//! @param  tid  thread identified (e.g. threadIdx.x or gtid)
//! @param  num_intervals_active  number of active intervals / threads that
//!                               currently process an interval
//! @param  s_d  scratch space to store diagonal entries of the tridiagonal
//!              matrix in shared memory
//! @param  s_s  scratch space to store superdiagonal entries of the tridiagonal
//!              matrix in shared memory
//! @param  converged  flag if the current thread is already converged (that
//!         is count does not have to be computed)
////////////////////////////////////////////////////////////////////////////////
__device__
 unsigned int
computeNumSmallerEigenvalsLarge(float *g_d, float *g_s, const unsigned int n,
                                const float x,
                                const unsigned int tid,
                                const unsigned int num_intervals_active,
                                float *s_d, float *s_s,
                                unsigned int converged
                               )
{
    float  delta = 1.0f;
    unsigned int count = 0;

    unsigned int rem = n;

    // do until whole diagonal and superdiagonal has been loaded and processed
    for (unsigned int i = 0; i < n; i += blockDim.x)
    {

        __syncthreads();

        // read new chunk of data into shared memory
        if ((i + threadIdx.x) < n)
        {

            s_d[threadIdx.x] = *(g_d + i + threadIdx.x);
            s_s[threadIdx.x] = *(g_s + i + threadIdx.x - 1);
        }

        __syncthreads();


        if (tid < num_intervals_active)
        {

            // perform (optimized) Gaussian elimination to determine the number
            // of eigenvalues that are smaller than n
            for (unsigned int k = 0; k < min(rem,blockDim.x); ++k)
            {
                delta = s_d[k] - x - (s_s[k] * s_s[k]) / delta;
                // delta = (abs( delta) < (1.0e-10)) ? -(1.0e-10) : delta;
                count += (delta < 0) ? 1 : 0;
            }

        }  // end if thread currently processing an interval

        rem -= blockDim.x;
    }

    return count;
}