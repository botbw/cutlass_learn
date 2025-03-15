// nvcc -arch=sm_80 -I${CUTLASS_REPO_PATH}/include mma.cu && ./a.out && rm a.out

#include <cstdlib>
#include <stdio.h>
#include <iostream>

#include <cuda_bf16.h>

#include "cute/arch/mma_sm80.hpp"
#include "utils.cuh"

__global__ void gemmKernel(double *A, double *B, double *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row index of C
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column index of C

    if (row < M && col < N) {
        double sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;  // Store result in C
    }
}


template <int BM, int BN, int BK, int WARP_REP_M, int WARP_REP_N>
__global__ void simpleGemmWOCuTe(const double *A, const double *B, double *C, int m, int n, int k) {
    /*
        A, B, C are column-major matrices
        A: (m, k):(1, m)
        B: (k, n):(1, n)
        C: (m, n):(1, m)
    */
    // sanity check
    assert(m % BM == 0);
    assert(n % BN == 0);
    assert(k % BK == 0);
    static_assert(WARP_REP_M * 8 == BM);
    static_assert(WARP_REP_N * 8 == BN);
    static_assert(BM % 8 == 0);
    static_assert(BK % 4 == 0);
    static_assert(BN % 4 == 0);

    int warpId = threadIdx.x / 32;
    int warp_i = warpId % WARP_REP_N, warp_j = warpId / WARP_REP_N;
    int lane_id = threadIdx.x % 32;

    double c0_acc, c1_acc;
    c0_acc = c1_acc = 0;
    for (int blk_k = 0; blk_k < k / BK; blk_k++) {
        double a0, b0, d0, d1;
        #pragma unroll
        for (int warp_k = 0; warp_k < BK / 4; warp_k++) {  // each tensor core takes k = 4
            // from nvidia official guide
            int a_row = lane_id / 4;
            int a_col = lane_id % 4;
            // A is 8 x 4 per warp
            a0 = // A[blockIdx.x * BM + warp_i * 8 + a_row , blk_k * BK + warp_k * 4 + warp_j * 4 + a_col]
                *(A + blockIdx.x * BM + warp_i * 8 + a_row + (blk_k * BK + warp_k * 4 + warp_j * 4 + a_col) * m);

            // from nvidia official guide
            int b_row = lane_id % 4;
            int b_col = lane_id / 4;
            // B is 4 x 8 per warp
            b0 = // B[blk_k * BK + warp_k * 4 + warp_i * 4 + b_row, blockIdx.y * BN + warp_j * 8 + b_col]
                *(B + blk_k * BK + warp_k * 4 + warp_i * 4 + b_row + (blockIdx.y * BN + warp_j * 8 + b_col) * k);

            // wait for warp to load value to register and mma
            cute::SM80_8x8x4_F64F64F64F64_TN::fma(d0, d1, a0, b0, 0, 0);

            // accumulate
            c0_acc += d0;
            c1_acc += d1;
        }
    }
    // from nvidia official guide
    int c_groupID = lane_id / 4;
    int c_threadID_in_group = lane_id % 4;
    int c_row = c_groupID;
    int c0_col = c_threadID_in_group * 2;
    int c1_col = c_threadID_in_group * 2 + 1;

    // C is 8 x 8 per warp
    double *pC0 = // &C[blockIdx.x * BM + warp_i * 8 + c_row, blockIdx.y * BN + warp_j * 8 + c0_col]
        C + blockIdx.x * BM + warp_i * 8 + c_row + (blockIdx.y * BN + warp_j * 8 + c0_col) * m;
    double *pC1 = // &C[blockIdx.x * BM + warp_i * 8 + c_row, blockIdx.y * BN + warp_j * 8 + c1_col]
        C + blockIdx.x * BM + warp_i * 8 + c_row + (blockIdx.y * BN + warp_j * 8 + c1_col) * m;

    // save result
    *pC0 = c0_acc; //c0_acc;
    *pC1 = c1_acc; //c1_acc;
}

void refGemm(double *A, double *B, double *C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int l = 0; l < k; l++) {
                // A[i, l] * B[l, j]
                sum += A[i + l * m] * B[l + j * k];
            }
            // C[i, j]
            C[i + j * m] = sum;
        }
    }
}

void assertEqual(double *A, double *B, int size) {
    for (int i = 0; i < size; i++) {
        assert(A[i] == B[i]);
    }
}

void printMatrix(double *A, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            // A[i, j]
            std::cout << A[i + j * m] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;
}

int main(int argc, char const *argv[])
{
    int m = 2048;
    int n = 2048;
    int k = 2048;
    // parse args m n k
    if (argc == 4) {
        m = std::atoi(argv[1]);
        n = std::atoi(argv[2]);
        k = std::atoi(argv[3]);
    }
    double *A, *B, *C, *C_ref;
    cudaHostAlloc(reinterpret_cast<void**>(&A), m * k * sizeof(double), cudaHostAllocDefault);
    cudaHostAlloc(reinterpret_cast<void**>(&B), k * n * sizeof(double), cudaHostAllocDefault);
    cudaHostAlloc(reinterpret_cast<void**>(&C), m * n * sizeof(double), cudaHostAllocDefault);
    cudaHostAlloc(reinterpret_cast<void**>(&C_ref), m * n * sizeof(double), cudaHostAllocDefault);
    randn(A, m * k, 0, 10);
    randn(B, k * n, 0, 10);
    refGemm(A, B, C_ref, m, n, k);

    double *dA, *dB, *dC;
    cudaMalloc(reinterpret_cast<void**>(&dA), m * k * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&dB), k * n * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&dC), m * n * sizeof(double));
    cudaMemcpy(dA, A, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, k * n * sizeof(double), cudaMemcpyHostToDevice);

    constexpr int BK = 4;
    constexpr int WARP_REP_M = 1;
    constexpr int WARP_REP_N = 1;
    constexpr int BM = WARP_REP_M * 8;
    constexpr int BN = WARP_REP_N * 8;

    dim3 threads(32 * WARP_REP_M * WARP_REP_N);
    dim3 blocks(m / BM, n / BN);
    simpleGemmWOCuTe<BM, BN, BK, WARP_REP_M, WARP_REP_N><<<blocks, threads>>>(dA, dB, dC, m, n, k);
    cudaError_t err = cudaDeviceSynchronize();
    printf("CUDA error: %s\n", cudaGetErrorString(err));

    cudaMemcpy(C, dC, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    assertEqual(C, C_ref, m * n);
    printf("CUDA GEMM passed\n");

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    cudaFreeHost(C_ref);
    return 0;
}