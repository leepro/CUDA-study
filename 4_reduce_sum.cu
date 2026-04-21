#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024
#define THREADS 256

__global__ void reduceSumKernel(int *data, int *result, int n) {
    __shared__ int sdata[THREADS];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += data[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

int main() {
    int *h_data = (int *)malloc(N * sizeof(int));
    int h_result = 0;

    for (int i = 0; i < N; i++) {
        h_data[i] = i + 1;
    }

    int *d_data, *d_result;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));

    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice);

    reduceSumKernel<<<4, THREADS>>>(d_data, d_result, N);

    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sum 1..%d = %d (expected: %d)\n", N, h_result, N * (N + 1) / 2);

    cudaFree(d_data);
    cudaFree(d_result);
    free(h_data);

    return 0;
}