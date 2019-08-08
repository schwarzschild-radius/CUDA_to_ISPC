CXXFLAGS=-std=c++17 -g
CUDA_ARCH_FLAGS=--cuda-gpu-arch=sm_50 --cuda-path=/usr/local/cuda-9.0
CUDA_LINK_FLAGS=-L /usr/local/cuda-9.0/lib64  -lcudart -I ../utils/

sobol_qrng: main.cu tasksys.o sobol_qrng_cuda.o sobol_qrng_ispc.o sobol_data.o
	clang++-9 -o sobol_qrng ${CUDA_ARCH_FLAGS} ${CUDA_LINK_FLAGS} ${CXXFLAGS} -lpthread sobol_qrng_cuda.o tasksys.o sobol_qrng_ispc.o sobol_data.o main.cu
sobol_qrng_cuda.o: sobol_qrng.cu sobol_qrng.cuh
	clang++-9 -c ${CUDA_ARCH_FLAGS} -o sobol_qrng_cuda.o ${CXXFLAGS} sobol_qrng.cu
sobol_data.o : sobol_data.cpp sobol_data.h
	clang++-9 -c -o sobol_data.o ${CXXFLAGS} sobol_data.cpp
sobol_qrng_ispc.o: sobol_qrng.ispc
	ispc -g --target avx2-i32x8 -h sobol_qrng.h -o sobol_qrng_ispc.o sobol_qrng.ispc
tasksys.o:
	clang++-9 -c -o tasksys.o ${CXXFLAGS} ../utils/tasksys.cpp 

clean:
	rm sobol_qrng_ispc.o tasksys.o sobol_qrng_cuda.o sobol_qrng sobol_qrng.h