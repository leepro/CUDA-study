CUDA_FILES = 1_vector_add.cu 2_vector_add_atomic.cu 3_square.cu 4_reduce_sum.cu 5_dot_product.cu 6_transpose.cu 7_matmul.cu 8_ptx_example.cu circulate.cu
EXECUTABLES = $(CUDA_FILES:.cu=)

NVCC = nvcc
NVCC_FLAGS = -O2 -arch=sm_60

.PHONY: all clean clear

all: $(EXECUTABLES)

%: %.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean clear:
	rm -f $(EXECUTABLES)