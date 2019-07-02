#ifndef _BISECT_LARGE_CUH_
#define _BISECT_LARGE_CUH_
#include <cuda_runtime.h>
#include "structs.h"

void computeEigenvaluesLargeMatrix(const InputData &input,
                                   const ResultDataLarge &result,
                                   const unsigned int mat_size,
                                   const float precision, const float lg,
                                   const float ug,
                                   const unsigned int iterations);
void initResultDataLargeMatrix(ResultDataLarge &result,
                               const unsigned int mat_size);

void cleanupResultDataLargeMatrix(ResultDataLarge &result);

bool processResultDataLargeMatrix(const InputData &input,
                                  const ResultDataLarge &result,
                                  const unsigned int mat_size,
                                  const char *filename);
#endif