// nvcc -arch=sm_80 -I$HOME/cutlass/include gemm_optimization.cu && ./a.out && rm a.out
#include <cstdlib>
#include <stdio.h>
#include <iostream>

#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#include "cute/arch/mma_sm80.hpp"
#include "utils.cuh"

__global__ void trivialGemm(double *A, double *B, double *C, int M, int N, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M && j < N)
    {
        double sum = 0.0f;
        for (int k = 0; k < K; k++)
        {
            sum += A[i + k * M] * B[k + j * K];
        }
        C[i + j * M] = sum; // Store result in C
    }
}

constexpr int BM = 16;
constexpr int BN = 16;
constexpr int BK = 8;

template <int BM, int BN, int BK, typename TiledMMA>
__global__ void cuteStyleGemm(double *pA, double *pB, double *pC, int m, int n, int k, TiledMMA tiled_mma)
{
    /*
        A, B, C are column-major matrices
        A: (m, k):(1, m)
        B: (k, n):(1, k)
        C: (m, n):(1, m)
    */
    assert(m % BM == 0);
    assert(n % BN == 0);
    assert(k % BK == 0);
    static_assert(BM % 8 == 0);
    static_assert(BK % 4 == 0);
    static_assert(BN % 4 == 0);

    using namespace cute;
    Tensor A = make_tensor(pA, make_shape(m, k), make_stride(1, m));
    Tensor B = make_tensor(pB, make_shape(n, k), make_stride(k, 1)); // B is (k, n) in CuTe, but (n, k) in this code
    Tensor C = make_tensor(pC, make_shape(m, n), make_stride(1, m));

    Tensor gA = local_tile(A, Shape<Int<BM>, Int<BK>>{}, make_coord(blockIdx.x, _));          // same as zipped_divide(A, make_shape(BM, BK))(_, make_coord(blockIdx.x, _));
    Tensor gB = local_tile(B, Shape<Int<BN>, Int<BK>>{}, make_coord(blockIdx.y, _));          // zipped_divide(B, make_shape(BK, BN))(_, make_coord(_, blockIdx.y));
    Tensor gC = local_tile(C, Shape<Int<BM>, Int<BN>>{}, make_coord(blockIdx.x, blockIdx.y)); // zipped_divide(C, make_shape(BM, BN))(make_coord(_, _), make_coord(blockIdx.x, blockIdx.y));

    auto thr_mma = tiled_mma.get_slice(threadIdx.x);

    auto tAgA = thr_mma.partition_A(gA);
    auto tBgB = thr_mma.partition_B(gB);
    auto tCgC = thr_mma.partition_C(gC);

    auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));
    auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));
    auto tCrC = thr_mma.partition_fragment_C(gC(_, _));

    clear(tCrC);
#pragma unroll
    for (int blk_k = 0; blk_k < k / BK; blk_k++)
    {
        copy(tAgA(_, _, _, blk_k), tArA);
        copy(tBgB(_, _, _, blk_k), tBrB);
        cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
    }
    copy(tCrC, tCgC);
}

__global__ void checkArrayEquality(double *array1, double *array2, bool *result, int length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length)
    {
        if (array1[idx] != array2[idx] && (*result))
        {
            atomicExch((int *)result, int(false));
        }
    }
}

void assertEqual(double *A, double *B, int size)
{
    bool h_result = true;
    bool *d_result;
    double *d_A, *d_B;

    cudaMalloc((void **)&d_result, sizeof(bool));
    cudaMalloc((void **)&d_A, size * sizeof(double));
    cudaMalloc((void **)&d_B, size * sizeof(double));
    cudaMemcpy(d_result, &h_result, sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    checkArrayEquality<<<numBlocks, blockSize>>>(A, B, d_result, size);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_result);
    cudaFree(d_A);
    cudaFree(d_B);

    assert(h_result);
}

void printMatrix(double *A, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            // A[i, j]
            std::cout << A[i + j * m] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl
              << std::endl;
}

int main(int argc, char const *argv[])
{
    constexpr int WARMUP = 5;
    constexpr int RUN = 10;
    int m = 2048;
    int n = 2048;
    int k = 2048;
    // parse args m n k if provided
    if (argc == 4)
    {
        m = std::atoi(argv[1]);
        n = std::atoi(argv[2]);
        k = std::atoi(argv[3]);
    }
    double *A, *B, *C, *C_ref;
    cudaHostAlloc(reinterpret_cast<void **>(&A), m * k * sizeof(double), cudaHostAllocDefault);
    cudaHostAlloc(reinterpret_cast<void **>(&B), k * n * sizeof(double), cudaHostAllocDefault);
    cudaHostAlloc(reinterpret_cast<void **>(&C), m * n * sizeof(double), cudaHostAllocDefault);
    cudaHostAlloc(reinterpret_cast<void **>(&C_ref), m * n * sizeof(double), cudaHostAllocDefault);
    randn(A, m * k, 0, 10);
    randn(B, k * n, 0, 10);

    double *dA, *dB, *dC;
    cudaMalloc(reinterpret_cast<void **>(&dA), m * k * sizeof(double));
    cudaMalloc(reinterpret_cast<void **>(&dB), k * n * sizeof(double));
    cudaMalloc(reinterpret_cast<void **>(&dC), m * n * sizeof(double));
    cudaMemcpy(dA, A, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, k * n * sizeof(double), cudaMemcpyHostToDevice);

    {
        // test trivialGemm
        randn(C, m * n, 0, 100);
        dim3 threads(16, 16);
        dim3 blocks(m / 16, n / 16);
        // warm up
        for (int i = 0; i < WARMUP; i++)
        {
            trivialGemm<<<blocks, threads>>>(dA, dB, dC, m, n, k);
        }
        cudaDeviceSynchronize();
        time_t start, end;
        start = clock();
        for (int i = 0; i < RUN; i++)
        {
            trivialGemm<<<blocks, threads>>>(dA, dB, dC, m, n, k);
        }
        cudaDeviceSynchronize();
        end = clock();
        printf("Runtime of trivialGemm: %f\n", (double)(end - start) / CLOCKS_PER_SEC);
        cudaMemcpy(C_ref, dC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
    }

    {
        // test cuteStyleGemm
        using namespace cute;
        randn(C, m * n, 0, 100);
        auto tiled_mma = cute::make_tiled_mma(SM80_8x8x4_F64F64F64F64_TN{}, make_layout(Shape<Int<BM / 8>, Int<BN / 8>>{}));
        dim3 threads(size(tiled_mma));
        dim3 blocks(m / BM, n / BN);
        // warm up
        for (int i = 0; i < WARMUP; i++)
        {
            cuteStyleGemm<BM, BN, BK><<<blocks, threads>>>(dA, dB, dC, m, n, k, tiled_mma);
        }
        cudaDeviceSynchronize();
        time_t start, end;
        start = clock();
        for (int i = 0; i < RUN; i++)
        {
            cuteStyleGemm<BM, BN, BK><<<blocks, threads>>>(dA, dB, dC, m, n, k, tiled_mma);
        }
        cudaDeviceSynchronize();
        end = clock();
        printf("Runtime of cuteStyleGemm: %f\n", (double)(end - start) / CLOCKS_PER_SEC);
        cudaMemcpy(C, dC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
        assertEqual(C, C_ref, m * n);
        printf("cuteStyleGemm passed\n");
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    cudaFreeHost(C_ref);
    return 0;
}