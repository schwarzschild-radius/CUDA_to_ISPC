CXXFLAGS=-std=c++11 -I ../utils/
CUDACC=clang++-9
CUDA_ARCH_FLAGS=
CUDA_LINK_FLAGS=
CUDA_PATH=/usr/local/cuda-9.0/
ifeq ($(CUDACC), clang++-9)
	CUDA_ARCH_FLAGS=--cuda-gpu-arch=sm_50 --cuda-path=${CUDA_PATH}
	CUDA_LINK_FLAGS=-L ${CUDA_PATH}lib64  -lcudart
endif

finite_difference: finite_difference_ispc.o tasksys.o 
	${CUDACC} ${CUDA_ARCH_FLAGS} ${CUDA_LINK_FLAGS} ${CXXFLAGS} -lpthread finite_difference_ispc.o tasksys.o main.cu -o finite_difference
finite_difference_cuda.o: finite_difference_cuda.cu finite_difference_cuda.cuh
	${CUDACC} -c ${CUDA_ARCH_FLAGS} -o finite_difference_cuda.o ${CXXFLAGS} finite_difference_cuda.cu
finite_difference_ispc.o: finite_difference_ispc.ispc
	ispc --target avx2-i32x8 -h finite_difference_ispc.h -o finite_difference_ispc.o finite_difference_ispc.ispc

tasksys.o:
	${CUDACC} -c -o tasksys.o -std=c++11 ../utils/tasksys.cpp 

clean:
	rm *.o finite_difference *_ispc.h