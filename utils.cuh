#ifndef UTILS_CUH
#define UTILS_CUH

#include <iostream>
#include <curand_kernel.h>

template <typename T>
__global__ void randn_kernel(T* arr, int size, double vmin, double vmax, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        double rand_val = curand_uniform(&state);
        arr[idx] = int(vmin + (vmax - vmin) * rand_val);
    }
}

template <typename T>
void randn(T* arr, int size, double vmin, double vmax) {
    T* d_arr;
    unsigned long long seed = 1234;

    cudaMalloc((void**)&d_arr, size * sizeof(T));
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    randn_kernel<T><<<numBlocks, blockSize>>>(d_arr, size, vmin, vmax, seed);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }

    cudaMemcpy(arr, d_arr, size * sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}

#endif // UTILS_CUH