#include "gputimer.h"

#define cudaCheck(stmt)                                                        \
  {                                                                            \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "Failed to run " << #stmt << std::endl;                     \
      std::cerr << cudaGetErrorString(err) << std::endl;                       \
    }                                                                          \
  }

#define cudaCheckLaunch(kernel, grid, block, ...)                              \
  {                                                                            \
    GpuTimer timer;                                                            \
    timer.Start();                                                             \
    kernel<<<grid, block>>>(__VA_ARGS__);                                      \
    timer.Stop();                                                              \
    std::cout << #kernel << ":\nTime elapsed: " << timer.Elapsed() << "\n";    \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "Failed to run " << #kernel << std::endl;                   \
      std::cerr << cudaGetErrorString(err) << std::endl;                       \
    }                                                                          \
  }