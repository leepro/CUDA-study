#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 8

__global__ void transposeKernel(float *input, float *output, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        output[col * n + row] = input[row * n + col];
    }
}

int main() {
    float *h_input = (float *)malloc(N * N * sizeof(float));
    float *h_output = (float *)malloc(N * N * sizeof(float));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_input[i * N + j] = i * N + j;
        }
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * N * sizeof(float));
    cudaMalloc(&d_output, N * N * sizeof(float));

    cudaMemcpy(d_input, h_input, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(4, 4);
    dim3 dimGrid((N + 3) / 4, (N + 3) / 4);
    transposeKernel<<<dimGrid, dimBlock>>>(d_input, d_output, N);

    cudaMemcpy(h_output, d_output, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Input[0][1] = %f -> Output[1][0] = %f\n", 
           h_input[0 * N + 1], h_output[1 * N + 0]);
    printf("Input[2][3] = %f -> Output[3][2] = %f\n", 
           h_input[2 * N + 3], h_output[3 * N + 2]);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}