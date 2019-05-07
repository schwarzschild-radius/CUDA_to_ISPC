ISPC:=ispc
NVCC:=nvcc
transpose: transpose.o tranpose.cu
	$(NVCC) tranpose.cu -gencode arch=compute_50 code=sm_50 -std=c++14

transpose.o: transpose.ispc
	$(ISPC) -o tranpose_ispc.o -h tranpose_ispc.h tranpose.ispc --target avx2-i32x8

.PHONY : clean

clean:
	*.o *_ispc.h