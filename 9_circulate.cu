#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void circulate(int *data, int num_threads) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_threads) {
        if (tid > 0) {
            data[tid] = data[tid - 1] + 1;
        }
    }
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int cores = prop.multiProcessorCount;
    int threads_per_block = prop.maxThreadsPerBlock;
    
    printf("CUDA Device: %s\n", prop.name);
    printf("Number of multiprocessors: %d\n", cores);
    printf("Max threads per block: %d\n", threads_per_block);
    printf("Total cores in system: %d\n", cores * 128);
    
    int num_threads = cores * 128;
    if (num_threads > 1024) num_threads = 1024;
    
    printf("\nCirculating number through %d threads...\n", num_threads);
    
    int *d_data;
    cudaMalloc(&d_data, num_threads * sizeof(int));
    
    cudaMemset(d_data, 0, num_threads * sizeof(int));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    int blocks = (num_threads + 255) / 256;
    for (int i = 0; i < num_threads; i++) {
        circulate<<<blocks, 256>>>(d_data, num_threads);
        cudaDeviceSynchronize();
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    int *h_data = (int *)malloc(num_threads * sizeof(int));
    cudaMemcpy(h_data, d_data, num_threads * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Final value after circulating through all threads: %d\n", h_data[num_threads - 1]);
    printf("Elapsed time: %.6f ms\n", elapsedTime);
    
    cudaFree(d_data);
    free(h_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}