CXXFLAGS=-std=c++17
CLANGCXX=clang++-9
CUDA_ARCH_FLAGS=sm_50
CUDA_PATH=/usr/local/cuda-9.0
CUDA_LINK_FLAGS=-L${CUDA_PATH}/lib64  -lcudart
INCLUDES=-I ../utils/
ISPC_TARGET=avx2-i32x8
DEBUG=-g

CUDA_CLANG=${CLANGCXX} --cuda-gpu-arch=${CUDA_ARCH_FLAGS} --cuda-path=${CUDA_PATH} ${CUDA_LINK_FLAGS} ${CXXFLAGS}

main: main.cu bisect_large.o
	${CUDA_CLANG} -g -o main main.cu bisect_large.o

bisect_large.o: bisect_large.cu
	${CUDA_CLANG} -o bisect_large.o -c bisect_large.cu

# prefix_sum: main.cu prefix_sum_ispc.o tasksys.o prefix_sum_cuda.cuh prefix_sum_cuda.o prefix_sum_cuda.cu
 	# clang++ -o prefix_sum ${CUDA_ARCH_FLAGS} ${CUDA_LINK_FLAGS} ${CXXFLAGS} -lpthread prefix_sum_cuda.o prefix_sum_ispc.o tasksys.o main.cu
# prefix_sum_cuda.o: prefix_sum_cuda.cu prefix_sum_cuda.cuh
	# clang++ -c ${CUDA_ARCH_FLAGS} -o prefix_sum_cuda.o ${CXXFLAGS} prefix_sum_cuda.cu
# prefix_sum_ispc.o: prefix_sum_ispc.ispc
# 	ispc --target avx2-i32x8 -h prefix_sum_ispc.h -o prefix_sum_ispc.o prefix_sum_ispc.ispc

# tasksys.o:
# 	clang++ -c -o tasksys.o -std=c++11 ../utils/tasksys.cpp 

clean:
	rm *.o main *_ispc.h