#include <cuda_runtime.h>

#define  TIME_STEPS 16
#define  CACHE_DELTA (2 * TIME_STEPS)
#define  CACHE_SIZE (256)
#define  CACHE_STEP (CACHE_SIZE - CACHE_DELTA)
// Number of time steps
#define NUM_STEPS 2048
// Max option batch size
#define MAX_OPTIONS 1024


#if NUM_STEPS % CACHE_DELTA
#error Bad constants
#endif

//Preprocessed input option data
typedef struct
{
    double S;
    double X;
    double T, R, V;
    double vDt;
    double puByDf;
    double pdByDf;
} __TOptionData;

__global__ void binomialOptionsKernel(__TOptionData *d_OptionData, double *d_CallBuffer, float *d_CallValue);