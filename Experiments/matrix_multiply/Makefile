CXXFLAGS=-std=c++17
CUDA_PATH=/usr/local/cuda-9.0/
CLANGCXX=clang++-9
CUDA_ARCH_FLAGS=--cuda-gpu-arch=sm_50 --cuda-path=${CUDA_PATH}
CUDA_LINK_FLAGS=-L ${CUDA_PATH}/lib64  -lcudart -I ../utils/

matrix_multiply: main.cu matrix_multiply_ispc.o tasksys.o matrix_multiply_cuda.cuh matrix_multiply_cuda.o
	${CLANGCXX} ${CUDA_ARCH_FLAGS} ${CUDA_LINK_FLAGS} ${CXXFLAGS} -lpthread matrix_multiply_cuda.o matrix_multiply_ispc.o tasksys.o main.cu -o matrix_multiply
matrix_multiply_cuda.o: matrix_multiply_cuda.cu matrix_multiply_cuda.cuh
	${CLANGCXX} -c ${CUDA_ARCH_FLAGS} -o matrix_multiply_cuda.o ${CXXFLAGS} matrix_multiply_cuda.cu
matrix_multiply_ispc.o: matrix_multiply_ispc.ispc
	ispc --target avx2-i32x8 -h matrix_multiply_ispc.h -o matrix_multiply_ispc.o matrix_multiply_ispc.ispc

tasksys.o:
	${CLANGCXX} -c -o tasksys.o -std=c++11 ../utils/tasksys.cpp 

clean:
	rm *.o matrix_multiply *_ispc.h