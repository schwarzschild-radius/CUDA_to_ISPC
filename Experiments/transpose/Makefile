CXXFLAGS=-std=c++11
CUDA_ARCH_FLAGS=--cuda-gpu-arch=sm_50
CUDA_LINK_FLAGS=-L /usr/local/cuda/lib64  -lcudart -I ../utils/

transpose: transpose_ispc.o tasksys.o transpose_cuda.cuh transpose_cuda.o
	clang++ ${CUDA_ARCH_FLAGS} ${CUDA_LINK_FLAGS} ${CXXFLAGS} -lpthread transpose_cuda.o transpose_ispc.o tasksys.o main.cu
transpose_cuda.o: transpose_cuda.cu transpose_cuda.cuh
	clang++ -c ${CUDA_ARCH_FLAGS} -o transpose_cuda.o ${CXXFLAGS} transpose_cuda.cu
transpose_ispc.o: transpose_ispc.ispc
	ispc --target avx2-i32x8 -h transpose_ispc.h -o transpose_ispc.o transpose_ispc.ispc

tasksys.o:
	clang++ -c -o tasksys.o -std=c++11 ../utils/tasksys.cpp 

clean:
	rm *.o a.out *_ispc.h