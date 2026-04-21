#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024*1024

__global__ void squareKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * data[idx];
    }
}

int main() {
    float *h_data = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_data[i] = i;
    }

    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (N + 255) / 256;
    squareKernel<<<blocks, 256>>>(d_data, N);

    cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("data[0]^2 = %f\n", h_data[0]);
    printf("data[10]^2 = %f\n", h_data[10]);
    printf("data[100]^2 = %f\n", h_data[100]);

    cudaFree(d_data);
    free(h_data);

    return 0;
}