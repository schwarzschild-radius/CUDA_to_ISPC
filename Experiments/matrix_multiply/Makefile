CXXFLAGS=-std=c++11
CUDA_ARCH_FLAGS=--cuda-gpu-arch=sm_50
CUDA_LINK_FLAGS=-L /usr/local/cuda-9.0/lib64  -lcudart -I ../utils/

matrix_multiply: matrix_multiply_ispc.o tasksys.o matrix_multiply_cuda.cuh matrix_multiply_cuda.o
	clang++-9 ${CUDA_ARCH_FLAGS} ${CUDA_LINK_FLAGS} ${CXXFLAGS} -lpthread matrix_multiply_cuda.o matrix_multiply_ispc.o tasksys.o main.cu
matrix_multiply_cuda.o: matrix_multiply_cuda.cu matrix_multiply_cuda.cuh
	clang++-9 -c ${CUDA_ARCH_FLAGS} -o matrix_multiply_cuda.o ${CXXFLAGS} matrix_multiply_cuda.cu
matrix_multiply_ispc.o: matrix_multiply_ispc.ispc
	ispc --target avx2-i32x8 -h matrix_multiply_ispc.h -o matrix_multiply_ispc.o matrix_multiply_ispc.ispc

tasksys.o:
	clang++ -c -o tasksys.o -std=c++11 ../utils/tasksys.cpp 

clean:
	rm *.o a.out *_ispc.h