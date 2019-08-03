#define cudaCheck(stmt)                                                        \
    {                                                                          \
        cudaError_t err = stmt;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << __LINE__ << ":Failed to run " << #stmt << std::endl;               \
            std::cerr << cudaGetErrorString(err) << std::endl;                 \
        }                                                                      \
    }

#define cudaCheckLaunch(kernel, grid, block, shared_memory, ...)                 \
    {                                                                          \
        kernel<<<grid, block, shared_memory>>>(__VA_ARGS__);                   \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            std::cerr << "Failed to run " << #kernel << std::endl;             \
            std::cerr << cudaGetErrorString(err) << std::endl;                 \
        }                                                                      \
    }