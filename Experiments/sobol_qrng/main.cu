#include <iostream>
#include <math.h>
#include <stdexcept>
#include <vector>

#include "cuda_utils.cuh"
#include "sobol_data.h"
#include "sobol_qrng.cuh"
#include "sobol_qrng.h"

#define L1ERROR_TOLERANCE (1e-6)
#define k_2powneg32 2.3283064E-10F

void executeCUDA(int n_dimensions, int n_vectors, unsigned int *h_directions,
                 float *h_outputGPU) {
    unsigned int *d_directions;
    float *d_output;
    cudaCheck(cudaMalloc((void **)&d_directions,
                         n_dimensions * n_directions * sizeof(unsigned int)));
    cudaCheck(cudaMalloc((void **)&d_output,
                         n_vectors * n_dimensions * sizeof(float)));

    cudaCheck(cudaMemcpy(d_directions, h_directions,
                         n_dimensions * n_directions * sizeof(unsigned int),
                         cudaMemcpyHostToDevice));
    const int threadsperblock = 64;
    int device = 0;
    cudaDeviceProp prop;
    cudaCheck(cudaGetDevice(&device));
    cudaCheck(cudaGetDeviceProperties(&prop, device));
    dim3 dimGrid;
    dim3 dimBlock;
    dimGrid.y = n_dimensions;
    if (n_dimensions < (4 * prop.multiProcessorCount)) {
        dimGrid.x = 4 * prop.multiProcessorCount;
    } else {
        dimGrid.x = 1;
    }
    if (dimGrid.x > (unsigned int)(n_vectors / threadsperblock)) {
        dimGrid.x = (n_vectors + threadsperblock - 1) / threadsperblock;
    }
    unsigned int targetDimGridX = dimGrid.x;
    for (dimGrid.x = 1; dimGrid.x < targetDimGridX; dimGrid.x *= 2)
        ;
    dimBlock.x = threadsperblock;
    std::cout << dimGrid.x << " " << dimBlock.x << '\n';
    sobol<<<dimGrid, dimBlock>>>(n_vectors, n_dimensions, d_directions,
                                 d_output);

    cudaDeviceSynchronize();
    cudaCheck(cudaMemcpy(h_outputGPU, d_output,
                         n_vectors * n_dimensions * sizeof(float),
                         cudaMemcpyDeviceToHost));
}

void executeISPC(int n_dimensions, int n_vectors, unsigned int *h_directions,
                 float *h_output) {
    const int threadsperblock = 64;
    int device = 0;
    cudaDeviceProp prop;
    cudaCheck(cudaGetDevice(&device));
    cudaCheck(cudaGetDeviceProperties(&prop, device));
    dim3 dimGrid;
    dim3 dimBlock;
    dimGrid.y = n_dimensions;
    if (n_dimensions < (4 * prop.multiProcessorCount)) {
        dimGrid.x = 4 * prop.multiProcessorCount;
    } else {
        dimGrid.x = 1;
    }
    if (dimGrid.x > (unsigned int)(n_vectors / threadsperblock)) {
        dimGrid.x = (n_vectors + threadsperblock - 1) / threadsperblock;
    }
    unsigned int targetDimGridX = dimGrid.x;
    for (dimGrid.x = 1; dimGrid.x < targetDimGridX; dimGrid.x *= 2)
        ;
    dimBlock.x = threadsperblock;
    std::cout << dimGrid.x << " " << dimBlock.x << '\n';
    ispc::sobol(
        {static_cast<int32_t>(dimGrid.x), static_cast<int32_t>(dimGrid.y),
         static_cast<int32_t>(dimGrid.z)},
        {static_cast<int32_t>(dimBlock.x), static_cast<int32_t>(dimBlock.y),
         static_cast<int32_t>(dimBlock.z)},
        0, n_vectors, n_dimensions, h_directions, h_output);
}

void executeReference(int n_dimensions, int n_vectors, unsigned int *directions,
                      float *output) {
    unsigned int *v = directions;

    for (int d = 0; d < n_dimensions; d++) {
        unsigned int X = 0;
        output[n_vectors * d] = 0.0;

        for (int i = 1; i < n_vectors; i++) {
            X ^= v[ffs(~(i - 1)) - 1];
            output[i + n_vectors * d] = (float)X * k_2powneg32;
        }

        v += n_directions;
    }
}

bool checkResuts(int n_dimensions, int n_vectors, std::vector<float> ref,
                 std::vector<float> cuda, std::vector<float> ispc) {
    float CUDA_l1norm_diff = 0.0F;
    float CUDA_l1norm_ref = 0.0F;
    float CUDA_l1error;

    float ISPC_l1norm_diff = 0.0F;
    float ISPC_l1norm_ref = 0.0F;
    float ISPC_l1error;
    for (int d = 0; d < n_dimensions; d++) {
        for (int v = 0; v < n_vectors; v++) {
            /* std::cout << cuda[d * n_vectors + v] << " "
                      << ref[d * n_vectors + v] << " "
                      << ispc[d * n_vectors + v] << "\n"; */
            float v_ref = ref[d * n_vectors + v];
            // CUDA
            CUDA_l1norm_diff += fabs(cuda[d * n_vectors + v] - v_ref);
            CUDA_l1norm_ref += fabs(v_ref);
            // ISPC
            ISPC_l1norm_diff += fabs(ispc[d * n_vectors + v] - v_ref);
            ISPC_l1norm_ref += fabs(v_ref);
        }
    }

    // Output the L1-Error
    CUDA_l1error = CUDA_l1norm_diff / CUDA_l1norm_ref;
    ISPC_l1error = ISPC_l1norm_diff / ISPC_l1norm_ref;

    if (CUDA_l1norm_ref == 0 && ISPC_l1norm_ref == 0) {
        return true;
    } else {
        std::cout << "L1-Error: " << CUDA_l1error << " " << ISPC_l1error
                  << std::endl;
    }
    return false;
}

int main() {
    // We will generate n_vectors vectors of n_dimensions numbers
    int n_vectors = 100000;
    int n_dimensions = 100;

    std::vector<float> ref(n_vectors * n_dimensions),
        cuda(n_vectors * n_dimensions), ispc(n_vectors * n_dimensions);
    if (ref.size() == 0 || cuda.size() == 0 || ispc.size() == 0) {
        return 1;
    }
    unsigned int *h_directions = new unsigned int[n_dimensions * n_directions];
    unsigned int *v = h_directions;

    for (int dim = 0; dim < n_dimensions; dim++) {
        if (dim == 0) {
            for (int i = 0; i < n_directions; i++) {
                v[i] = 1 << (31 - i);
            }
        } else {
            int d = sobol_primitives[dim].degree;
            for (int i = 0; i < d; i++) {
                v[i] = sobol_primitives[dim].m[i] << (31 - i);
            }
            for (int i = d; i < n_directions; i++) {
                v[i] = v[i - d] ^ (v[i - d] >> d);
                for (int j = 1; j < d; j++) {
                    v[i] ^= (((sobol_primitives[dim].a >> (d - 1 - j)) & 1) *
                             v[i - j]);
                }
            }
        }
        v += n_directions;
    }

    executeReference(n_dimensions, n_vectors, h_directions, ref.data());
    executeCUDA(n_dimensions, n_vectors, h_directions, cuda.data());
    executeISPC(n_dimensions, n_vectors, h_directions, ispc.data());

    if (checkResuts(n_dimensions, n_vectors, ref, cuda, ispc))
        return 1;

    return 0;
}