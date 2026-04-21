#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024*2
#define BLOCK_SIZE 16

__global__ void matMulKernel(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

void matMul(float *h_A, float *h_B, float *h_C, int n) {
    int size = n * n * sizeof(float);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    matMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    float *h_A = (float *)malloc(N * N * sizeof(float));
    float *h_B = (float *)malloc(N * N * sizeof(float));
    float *h_C = (float *)malloc(N * N * sizeof(float));
    
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    
    matMul(h_A, h_B, h_C, N);
    
    printf("C[0] = %f (expected: %f)\n", h_C[0], (float)N * 2.0f);
    printf("C[1023] = %f\n", h_C[1023]);
    
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
