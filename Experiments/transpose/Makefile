CXXFLAGS=-std=c++17
CLANGCXX=clang++-9
CUDA_PATH=/usr/local/cuda-9.0
CUDA_ARCH_FLAGS=--cuda-gpu-arch=sm_50 --cuda-path=${CUDA_PATH}
CUDA_LINK_FLAGS=-L ${CUDA_PATH}/lib64  -lcudart -I ../utils/

transpose: main.cu transpose_ispc.o tasksys.o transpose_cuda.cuh transpose_cuda.o
	${CLANGCXX} ${CUDA_ARCH_FLAGS} ${CUDA_LINK_FLAGS} ${CXXFLAGS} -lpthread transpose_cuda.o transpose_ispc.o tasksys.o main.cu -o transpose
transpose_cuda.o: transpose_cuda.cu transpose_cuda.cuh
	${CLANGCXX} -c ${CUDA_ARCH_FLAGS} -o transpose_cuda.o ${CXXFLAGS} transpose_cuda.cu
transpose_ispc.o: transpose_ispc.ispc
	ispc --target avx2-i32x8 -h transpose_ispc.h -o transpose_ispc.o transpose_ispc.ispc

tasksys.o:
	${CLANGCXX} -c -o tasksys.o -std=c++11 ../utils/tasksys.cpp 

clean:
	rm *.o transpose *_ispc.h