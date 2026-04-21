#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024

__global__ void histogramKernel(int *data, int *bins, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int val = data[idx] % 10;
        atomicAdd(&bins[val], 1);
    }
}

int main() {
    int *h_data = (int *)malloc(N * sizeof(int));
    int *h_bins = (int *)malloc(10 * sizeof(int));

    for (int i = 0; i < N; i++) {
        h_data[i] = i;
    }
    for (int i = 0; i < 10; i++) h_bins[i] = 0;

    int *d_data, *d_bins;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_bins, 10 * sizeof(int));

    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bins, h_bins, 10 * sizeof(int), cudaMemcpyHostToDevice);

    histogramKernel<<<1, 256>>>(d_data, d_bins, N);

    cudaMemcpy(h_bins, d_bins, 10 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Histogram (mod 10):\n");
    for (int i = 0; i < 10; i++) {
        printf("Bin %d: %d\n", i, h_bins[i]);
    }

    cudaFree(d_data);
    cudaFree(d_bins);
    free(h_data);
    free(h_bins);

    return 0;
}