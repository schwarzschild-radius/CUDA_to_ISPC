#include "gputimer.h"

#define cudaCheck(stmt)                                                        \
  {                                                                            \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "Failed to run " << #stmt << std::endl;                     \
      std::cerr << cudaGetErrorString(err) << std::endl;                       \
    }                                                                          \
  }

#define cudaCheckLaunch(stmt)                                                  \
  {                                                                            \
    GpuTimer timer;                                                            \
    timer.start();                                                             \
    stmt;                                                                      \
    timer.stop();                                                              \
    std::cout << "Time elapsed: " << timer.elapsed();                          \
    cudaDeviceSynchronize();                                                   \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "Failed to run " << #stmt << std::endl;                     \
      std::cerr << cudaGetErrorString(err) << std::endl;                       \
    }                                                                          \
  }