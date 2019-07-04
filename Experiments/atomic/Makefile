CXXFLAGS=-std=c++17
CUDA_PATH=/usr/local/cuda-9.0/
CUDA_ARCH_FLAGS=--cuda-gpu-arch=sm_50 --cuda-path=${CUDA_PATH}
CUDA_LINK_FLAGS=-L ${CUDA_PATH}/lib64  -lcudart -I ../utils/

atomic: main.cu atomic_ispc.o tasksys.o atomic_cuda.cuh atomic_cuda.o
	clang++-9 ${CUDA_ARCH_FLAGS} ${CUDA_LINK_FLAGS} ${CXXFLAGS} -lpthread atomic_cuda.o atomic_ispc.o tasksys.o main.cu -o atomic
atomic_cuda.o: atomic_cuda.cu atomic_cuda.cuh
	clang++-9 -c ${CUDA_ARCH_FLAGS} -o atomic_cuda.o ${CXXFLAGS} atomic_cuda.cu
atomic_ispc.o: atomic_ispc.ispc
	ispc --target avx2-i32x8 -h atomic_ispc.h -o atomic_ispc.o atomic_ispc.ispc

tasksys.o:
	clang++-9 -c -o tasksys.o -std=c++11 ../utils/tasksys.cpp 

clean:
	rm *.o atomic *_ispc.h