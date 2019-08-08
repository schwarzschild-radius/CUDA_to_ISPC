#include <cuda_runtime.h>

#define n_directions 32

__global__ void sobol(unsigned n_vectors, unsigned n_dimensions,
                                unsigned *d_directions, float *d_output);