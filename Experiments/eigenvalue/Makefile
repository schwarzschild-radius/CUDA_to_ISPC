CXXFLAGS=-std=c++17
CLANGCXX=clang++-9
CUDA_ARCH_FLAGS=sm_50
CUDA_PATH=/usr/local/cuda-9.0
CUDA_LINK_FLAGS=-L${CUDA_PATH}/lib64  -lcudart
INCLUDES=-I ../utils/
ISPC_TARGET=avx2-i32x8
DEBUG=-g
CUDA_CLANG=${CLANGCXX} --cuda-gpu-arch=${CUDA_ARCH_FLAGS} --cuda-path=${CUDA_PATH}
CUDA_CLANG_SEPARATE_COMPILER_FLAG=''
ISPC=ispc

NVCC=/usr/local/cuda-10.1/bin/nvcc
NVCC_SEPARATE_COMPILER_FLAG=-dc
NVCC_CXXFLAGS=-std=c++14
NVCC_LINK_FLAGS=

eigenvalue: main.cu bisect_kernel_large_multi.o bisect_kernel_large.o bisect_kernel_large_onei.o bisect_kernel_large_ispc.o bisect_kernel_large_multi_ispc.o bisect_kernel_large_onei_ispc.o bisect_util.o
	${NVCC} ${NVCC_LINK_FLAGS} ${NVCC_CXXFLAGS} -I ../utils -g -o eigenvalue bisect_kernel_large_multi.o bisect_kernel_large.o bisect_kernel_large_onei.o bisect_util.o bisect_kernel_large_ispc.o bisect_kernel_large_multi_ispc.o bisect_kernel_large_onei_ispc.o main.cu

bisect_kernel_large.o: bisect_kernel_large.cu bisect_util.o
	${NVCC} ${NVCC_CXXFLAGS} ${NVCC_SEPARATE_COMPILER_FLAG} -o bisect_kernel_large.o -c bisect_kernel_large.cu

bisect_kernel_large_multi.o: bisect_kernel_large_multi.cu bisect_util.o
	${NVCC} ${NVCC_CXXFLAGS} ${NVCC_SEPARATE_COMPILER_FLAG} -o bisect_kernel_large_multi.o -c bisect_kernel_large_multi.cu

bisect_kernel_large_onei.o: bisect_kernel_large_onei.cu bisect_util.o
	${NVCC} ${NVCC_CXXFLAGS} ${NVCC_SEPARATE_COMPILER_FLAG} -o bisect_kernel_large_onei.o -c bisect_kernel_large_onei.cu

bisect_util.o: bisect_util.cu
	echo ${NVCC}
	${NVCC} ${NVCC_CXXFLAGS} ${NVCC_SEPARATE_COMPILER_FLAG} -o bisect_util.o -c bisect_util.cu

bisect_kernel_large_ispc.o: bisect_kernel_large.ispc
	${ISPC} --target=${ISPC_TARGET} -I ../utils -o bisect_kernel_large_ispc.o -h bisect_kernel_large.h bisect_kernel_large.ispc

bisect_kernel_large_multi_ispc.o: bisect_kernel_large_multi.ispc
	${ISPC} --target=${ISPC_TARGET} -I ../utils -o bisect_kernel_large_multi_ispc.o -h bisect_kernel_large_multi.h bisect_kernel_large_multi.ispc

bisect_kernel_large_onei_ispc.o: bisect_kernel_large_onei.ispc
	${ISPC} --target=${ISPC_TARGET} -I ../utils -o bisect_kernel_large_onei_ispc.o -h bisect_kernel_large_onei.h bisect_kernel_large_onei.ispc

clean:
	rm eigenvalue \
	bisect_kernel_large_multi.o bisect_kernel_large.o bisect_kernel_large_onei.o bisect_util.o \
	bisect_kernel_large_ispc.o bisect_kernel_large_multi_ispc.o bisect_kernel_large_onei_ispc.o \
	bisect_kernel_large.h bisect_kernel_large_multi.h bisect_kernel_large_onei.h