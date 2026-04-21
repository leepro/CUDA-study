#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define N 1024

__global__ void vectorAddAsm(float *A, float *B, float *C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float a = A[idx];
        float b = B[idx];
        float c;
        asm("add.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
        C[idx] = c;
    }
}

__global__ void madAsm(float *A, float *B, float *C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float a = A[idx];
        float b = B[idx];
        float c = C[idx];
        float result;
        asm("mad.rn.f32 %0, %1, %2, %3;" : "=f"(result) : "f"(a), "f"(b), "f"(c));
        C[idx] = result;
    }
}

__global__ void maxAsm(float *A, float *B, float *C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float a = A[idx];
        float b = B[idx];
        float result;
        asm("max.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
        C[idx] = result;
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

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (N + 255) / 256;

    printf("Testing PTX inline assembly:\n");
    vectorAddAsm<<<blocks, 256>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("add.f32: C[0] = %f, C[1] = %f, C[100] = %f\n", 
           h_C[0], h_C[1], h_C[100]);
    printf("expected: %f, %f, %f\n", 
           h_A[0] + h_B[0], h_A[1] + h_B[1], h_A[100] + h_B[100]);

    cudaMemcpy(d_C, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    madAsm<<<blocks, 256>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("mad.f32: C[0] = %f, C[1] = %f\n", h_C[0], h_C[1]);
    printf("expected: %f, %f\n", h_A[0]*h_B[0] + h_A[0], h_A[1]*h_B[1] + h_A[1]);

    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);
    maxAsm<<<blocks, 256>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("max.f32: C[0] = %f, C[1] = %f\n", h_C[0], h_C[1]);
    printf("expected: %f, %f\n", fmaxf(h_A[0], h_B[0]), fmaxf(h_A[1], h_B[1]));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}