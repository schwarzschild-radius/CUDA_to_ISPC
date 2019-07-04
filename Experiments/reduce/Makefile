CXXFLAGS=-std=c++11
CUDA_ARCH_FLAGS=--cuda-gpu-arch=sm_50 --cuda-path=/usr/local/cuda-9.0
CUDA_LINK_FLAGS=-L /usr/local/cuda-9.0/lib64  -lcudart -I ../utils/

reduce: main.cu reduce_ispc.o tasksys.o reduce_cuda.cuh reduce_cuda.o
	clang++ ${CUDA_ARCH_FLAGS} ${CUDA_LINK_FLAGS} ${CXXFLAGS} -lpthread reduce_cuda.o reduce_ispc.o tasksys.o main.cu -o reduce
reduce_cuda.o: reduce_cuda.cu reduce_cuda.cuh
	clang++ -c ${CUDA_ARCH_FLAGS} -o reduce_cuda.o ${CXXFLAGS} reduce_cuda.cu
reduce_ispc.o: reduce_ispc.ispc
	ispc --target avx2-i32x8 -h reduce_ispc.h -o reduce_ispc.o reduce_ispc.ispc

tasksys.o:
	clang++ -c -o tasksys.o -std=c++11 ../utils/tasksys.cpp 

clean:
	rm *.o reduce *_ispc.h