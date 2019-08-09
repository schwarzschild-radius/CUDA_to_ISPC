CXXFLAGS=-std=c++17 -g
CUDA_ARCH_FLAGS=--cuda-gpu-arch=sm_50 --cuda-path=/usr/local/cuda-9.0
CUDA_LINK_FLAGS=-L /usr/local/cuda-9.0/lib64  -lcudart -I ../utils/

bitonic_sort: main.cu tasksys.o bitonic_sort_cuda.o bitonic_sort_ispc.o
	clang++-9 -o bitonic_sort ${CUDA_ARCH_FLAGS} ${CUDA_LINK_FLAGS} ${CXXFLAGS} -lpthread bitonic_sort_cuda.o tasksys.o main.cu bitonic_sort_ispc.o
bitonic_sort_cuda.o: bitonic_sort.cu bitonic_sort.cuh
	clang++-9 -c ${CUDA_ARCH_FLAGS} -o bitonic_sort_cuda.o ${CXXFLAGS} bitonic_sort.cu
bitonic_sort_ispc.o: bitonic_sort.ispc
	ispc -g --target avx2-i32x8 -h bitonic_sort.h -o bitonic_sort_ispc.o bitonic_sort.ispc -I ../utils
tasksys.o:
	clang++-9 -c -o tasksys.o ${CXXFLAGS} ../utils/tasksys.cpp 

clean:
	rm bitonic_sort_ispc.o tasksys.o bitonic_sort_cuda.o bitonic_sort bitonic_sort.h