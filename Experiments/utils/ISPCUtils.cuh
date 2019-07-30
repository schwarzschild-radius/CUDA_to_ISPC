#define ISPC_LAUNCH_KERNEL(kernel, grid_param, block_param,                    \
                           dynamic_shmem_size, ...)                            \
    ispc::kernel(grid_param, block_param, dynamic_shmem_size, __VA_ARGS__)

struct A{};