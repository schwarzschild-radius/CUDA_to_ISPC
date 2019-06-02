CXXFLAGS=-std=c++11
CUDA_ARCH_FLAGS=--cuda-gpu-arch=sm_50
CUDA_LINK_FLAGS=-L /usr/local/cuda/lib64  -lcudart -I ../utils/

atomic: atomic_ispc.o tasksys.o atomic_cuda.cuh atomic_cuda.o
	clang++ ${CUDA_ARCH_FLAGS} ${CUDA_LINK_FLAGS} ${CXXFLAGS} -lpthread atomic_cuda.o atomic_ispc.o tasksys.o main.cu
atomic_cuda.o: atomic_cuda.cu atomic_cuda.cuh
	clang++ -c ${CUDA_ARCH_FLAGS} -o atomic_cuda.o ${CXXFLAGS} atomic_cuda.cu
atomic_ispc.o: atomic_ispc.ispc
	ispc --target avx2-i32x8 -h atomic_ispc.h -o atomic_ispc.o atomic_ispc.ispc

tasksys.o:
	clang++ -c -o tasksys.o -std=c++11 ../utils/tasksys.cpp 

clean:
	rm *.o a.out *_ispc.h