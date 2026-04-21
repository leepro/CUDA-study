# CUDA Learning Examples

A collection of simple CUDA programs to learn GPU programming.

## Programs

| # | File | Concept |
|---|------|---------|
| 1 | `1_vector_add.cu` | Basic vector addition with grid/block indexing |
| 2 | `2_vector_add_atomic.cu` | Histogram using atomic operations |
| 3 | `3_square.cu` | Element-wise squaring |
| 4 | `4_reduce_sum.cu` | Parallel reduction for sum |
| 5 | `5_dot_product.cu` | Dot product with shared memory |
| 6 | `6_transpose.cu` | Matrix transpose |
| - | `matmul.cu` | Matrix multiplication (full kernel) |

## Compile

```bash
nvcc <file>.cu -o <output>
```

Or use the provided script:

```bash
./compile.sh
```

## Run

```bash
./<program_name>
```

## Concepts Covered

- **Grid/Block indexing**: `blockIdx.x * blockDim.x + threadIdx.x`
- **Memory management**: `cudaMalloc`, `cudaMemcpy`, `cudaFree`
- **Kernel execution**: `<<<blocks, threads>>>`
- **Shared memory**: `__shared__` for thread-block local storage
- **Atomic operations**: `atomicAdd` for race-condition-free updates
- **Parallel reduction**: Tree-based summation pattern