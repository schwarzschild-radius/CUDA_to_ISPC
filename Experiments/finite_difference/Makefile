CXXFLAGS=-std=c++11 -I ../utils/
CUDACC=clang++
CUDA_ARCH_FLAGS=
CUDA_LINK_FLAGS=
ifeq ($(CUDACC), clang++)
	CUDA_ARCH_FLAGS=--cuda-gpu-arch=sm_50
	CUDA_LINK_FLAGS=-L /usr/local/cuda/lib64  -lcudart
endif

finite_difference: finite_difference_ispc.o tasksys.o 
	${CUDACC} ${CUDA_ARCH_FLAGS} ${CUDA_LINK_FLAGS} ${CXXFLAGS} -lpthread finite_difference_ispc.o tasksys.o main.cu
finite_difference_cuda.o: finite_difference_cuda.cu finite_difference_cuda.cuh
	${CUDACC} -c ${CUDA_ARCH_FLAGS} -o finite_difference_cuda.o ${CXXFLAGS} finite_difference_cuda.cu
finite_difference_ispc.o: finite_difference_ispc.ispc
	ispc --target avx2-i32x8 -h finite_difference_ispc.h -o finite_difference_ispc.o finite_difference_ispc.ispc

tasksys.o:
	${CUDACC} -c -o tasksys.o -std=c++11 ../utils/tasksys.cpp 

clean:
	rm *.o a.out *_ispc.h