CXXFLAGS=-std=c++17 -g
CUDA_ARCH_FLAGS=--cuda-gpu-arch=sm_50 --cuda-path=/usr/local/cuda-9.0
CUDA_LINK_FLAGS=-L /usr/local/cuda-9.0/lib64  -lcudart -I ../utils/

black_scholes: main.cu tasksys.o black_scholes_cuda.o black_scholes_ispc.o
	clang++-9 -o black_scholes ${CUDA_ARCH_FLAGS} ${CUDA_LINK_FLAGS} ${CXXFLAGS} -lpthread black_scholes_cuda.o tasksys.o main.cu black_scholes_ispc.o
black_scholes_cuda.o: black_scholes.cu black_scholes.cuh
	clang++-9 -c ${CUDA_ARCH_FLAGS} -o black_scholes_cuda.o ${CXXFLAGS} black_scholes.cu
black_scholes_ispc.o: black_scholes.ispc
	ispc --target avx2-i32x8 -h black_scholes.h -o black_scholes_ispc.o black_scholes.ispc
tasksys.o:
	clang++-9 -c -o tasksys.o ${CXXFLAGS} ../utils/tasksys.cpp 

clean:
	rm black_scholes_ispc.o tasksys.o black_scholes_cuda.o black_scholes black_scholes.h