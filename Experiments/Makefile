ISPC:=ispc
NVCC:=nvcc
transpose: transpose_ispc.o tasksys.o transpose.cu
	$(NVCC) transpose.cu transpose_ispc.o tasksys.o -gencode arch=compute_50,code=sm_50 -std=c++14

transpose.o: transpose.ispc
	$(ISPC) -o transpose_ispc.o -h transpose_ispc.h transpose.ispc --target avx2-i32x8

tasksys.o: tasksys.cpp
	$(NVCC) -c tasksys.cpp -o tasksys.o -lpthread

.PHONY : clean

clean:
	*.o *_ispc.h