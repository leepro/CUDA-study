#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024*1024

__global__ void vectorAdd(float *A, float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    float *h_A = (float *)malloc(N * sizeof(float));
    float *h_B = (float *)malloc(N * sizeof(float));
    float *h_C = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

//    printf("%f %f\n", h_A[100], h_B[100]);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (N + 255) / 256;
    vectorAdd<<<blocks, 256>>>(d_A, d_B, d_C, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("C[0] = %f (expected: %f)\n", h_C[0], h_A[0] + h_B[0]);
    printf("C[100] = %f (expected: %f)\n", h_C[100], h_A[100] + h_B[100]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
