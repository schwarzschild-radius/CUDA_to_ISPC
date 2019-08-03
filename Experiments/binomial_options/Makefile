CXXFLAGS=-std=c++17 -g
CUDA_ARCH_FLAGS=--cuda-gpu-arch=sm_50 --cuda-path=/usr/local/cuda-9.0
CUDA_LINK_FLAGS=-L /usr/local/cuda-9.0/lib64  -lcudart -I ../utils/

binomial_options: main.cu tasksys.o binomial_options_cuda.o binomial_options_ispc.o
	clang++-9 -o binomial_options ${CUDA_ARCH_FLAGS} ${CUDA_LINK_FLAGS} ${CXXFLAGS} -lpthread binomial_options_cuda.o tasksys.o binomial_options_ispc.o main.cu
binomial_options_cuda.o: binomial_options.cu binomial_options.cuh
	clang++-9 -c ${CUDA_ARCH_FLAGS} -o binomial_options_cuda.o ${CXXFLAGS} binomial_options.cu
binomial_options_ispc.o: binomial_options.ispc
	ispc --target avx2-i32x8 -h binomial_options.h -o binomial_options_ispc.o binomial_options.ispc
tasksys.o:
	clang++-9 -c -o tasksys.o ${CXXFLAGS} ../utils/tasksys.cpp 

clean:
	rm binomial_options_ispc.o tasksys.o binomial_options_cuda.o binomial_options binomial_options.h