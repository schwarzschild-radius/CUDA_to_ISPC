CXXFLAGS=-std=c++17
CUDA_ARCH_FLAGS=--cuda-gpu-arch=sm_50 --cuda-path=/usr/local/cuda-9.0
CUDA_LINK_FLAGS=-L /usr/local/cuda-9.0/lib64  -lcudart -I ../utils/

prefix_sum: main.cu prefix_sum_ispc.o tasksys.o prefix_sum_cuda.cuh prefix_sum_cuda.o prefix_sum_cuda.cu
	clang++ -o prefix_sum ${CUDA_ARCH_FLAGS} ${CUDA_LINK_FLAGS} ${CXXFLAGS} -lpthread prefix_sum_cuda.o prefix_sum_ispc.o tasksys.o main.cu
prefix_sum_cuda.o: prefix_sum_cuda.cu prefix_sum_cuda.cuh
	clang++ -c ${CUDA_ARCH_FLAGS} -o prefix_sum_cuda.o ${CXXFLAGS} prefix_sum_cuda.cu
prefix_sum_ispc.o: prefix_sum_ispc.ispc
	ispc --target avx2-i32x8 -h prefix_sum_ispc.h -o prefix_sum_ispc.o prefix_sum_ispc.ispc

tasksys.o:
	clang++ -c -o tasksys.o -std=c++11 ../utils/tasksys.cpp 

clean:
	rm *.o a.out *_ispc.h