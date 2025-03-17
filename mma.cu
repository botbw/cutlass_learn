// CUTLASS_REPO_PATH=~/cutlass nvcc -arch=sm_80 -I${CUTLASS_REPO_PATH}/include mma.cu && ./a.out && rm a.out
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

template <int BM, int BN, int BK>
__global__ void trivialMMAWithoutCuTe(const double *A, const double *B, double *C, int m, int n, int k)
{
    /*
        A, B, C are column-major matrices
        A: (m, k):(1, m)
        B: (k, n):(1, k)
        C: (m, n):(1, m)
    */
    // sanity check
    assert(m % BM == 0);
    assert(n % BN == 0);
    assert(k % BK == 0);
    static_assert(BM % 8 == 0);
    static_assert(BK % 4 == 0);
    static_assert(BN % 4 == 0);

    constexpr int WARP_REP_M = BM / 8;
    // constexpr int WARP_REP_N = BN / 8;
    constexpr int WARP_SLICE_K = BK / 4;

    int warpId = threadIdx.x / 32;
    int warp_i = warpId % WARP_REP_M;
    int warp_j = warpId / WARP_REP_M;
    int lane_id = threadIdx.x % 32;

    double c0_acc, c1_acc;
    c0_acc = c1_acc = 0;
    for (int blk_k = 0; blk_k < k / BK; blk_k++)
    {
        double a0, b0, d0, d1;
#pragma unroll
        for (int warp_k = 0; warp_k < WARP_SLICE_K; warp_k++)
        { // each tensor core takes k = 4
            // from nvidia official guide
            int a_row = lane_id / 4;
            int a_col = lane_id % 4;
            // A is 8 x 4 per warp
            a0 = // A[blockIdx.x * BM + warp_i * 8 + a_row , blk_k * BK + warp_k * 4 + warp_j * 4 + a_col]
                *(A + blockIdx.x * BM + warp_i * 8 + a_row + (blk_k * BK + warp_k * 4 + a_col) * m);

            // from nvidia official guide
            int b_row = lane_id % 4;
            int b_col = lane_id / 4;
            // B is 4 x 8 per warp
            b0 = // B[blk_k * BK + warp_k * 4 + warp_i * 4 + b_row, blockIdx.y * BN + warp_j * 8 + b_col]
                *(B + blk_k * BK + warp_k * 4 + b_row + (blockIdx.y * BN + warp_j * 8 + b_col) * k);

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
    *pC0 = c0_acc; // c0_acc;
    *pC1 = c1_acc; // c1_acc;
}

template <int BM, int BN, int BK>
__global__ void trivialMMAWithCuTe(const double *pA, const double *pB, double *pC, int m, int n, int k)
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

    constexpr int WARP_REP_M = BM / 8;
    // constexpr int WARP_REP_N = BN / 8;
    constexpr int WARP_SLICE_K = BK / 4;

    using namespace cute;
    Tensor A = make_tensor(pA, make_shape(m, k), make_stride(1, m));
    Tensor B = make_tensor(pB, make_shape(k, n), make_stride(1, k));
    Tensor C = make_tensor(pC, make_shape(m, n), make_stride(1, m));

    // tile abc and get cooresponding subtensor
    Tensor gA = zipped_divide(A, make_shape(BM, BK))(_, make_coord(blockIdx.x, _)); // (m, k) -> ((BM, BK), (m / BM, k / BK)) -> ((BM, BK), k / BK)
    Tensor gB = zipped_divide(B, make_shape(BK, BN))(_, make_coord(_, blockIdx.y));
    Tensor gC = zipped_divide(C, make_shape(BM, BN))(make_coord(_, _), make_coord(blockIdx.x, blockIdx.y));

    int warpId = threadIdx.x / 32;
    int warp_i = warpId % WARP_REP_M;
    int warp_j = warpId / WARP_REP_M;
    int lane_id = threadIdx.x % 32;

    double c0_acc, c1_acc;
    c0_acc = c1_acc = 0;
    for (int blk_k = 0; blk_k < k / BK; blk_k++)
    {
        double a0, b0, d0, d1;
        Tensor blkA = gA(make_coord(_, _), blk_k); // ((BM, BK), k / BK) -> (BM, BK)
        Tensor blkB = gB(make_coord(_, _), blk_k);
        Tensor blkA_sliced_k = zipped_divide(blkA, Shape<_8, _4>{}); // (BM, BK) -> ((8, 4), (WARP_REP_M, WARP_SLICE_K))
        Tensor blkB_sliced_k = zipped_divide(blkB, Shape<_4, _8>{}); // (BK, BN) -> ((4, 8), (WARP_SLICE_K, WARP_REP_N))
#pragma unroll
        for (int warp_k = 0; warp_k < WARP_SLICE_K; warp_k++)
        {
            auto ALayout = MMA_Traits<SM80_8x8x4_F64F64F64F64_TN>::ALayout{};
            Tensor warp_a = blkA_sliced_k(make_coord(_, _), make_coord(warp_i, warp_k)); // ((8, 4), (WARP_REP_M, WARP_SLICE_K)) -> (8, 4)
            a0 = warp_a(ALayout(lane_id, 0));

            // Note: B is (n, k) in CuTe, but (k, n) in this code
            auto BLayout = MMA_Traits<SM80_8x8x4_F64F64F64F64_TN>::BLayout{};
            Tensor warp_b = blkB_sliced_k(make_coord(_, _), make_coord(warp_k, warp_j));
            int b_nk = BLayout(lane_id, 0);
            int b_i = b_nk / 8, b_j = b_nk % 8;
            b0 = warp_b(b_i, b_j);

            cute::SM80_8x8x4_F64F64F64F64_TN::fma(d0, d1, a0, b0, 0, 0);

            c0_acc += d0;
            c1_acc += d1;
        }
    }
    Tensor warp_c = zipped_divide(gC, Shape<_8, _8>{})(make_coord(_, _), make_coord(warp_i, warp_j)); // ((BM, BN), (WARP_REP_M, WARP_REP_N)) -> ((8, 8), (WARP_REP_M, WARP_REP_N))
    auto CLayout = MMA_Traits<SM80_8x8x4_F64F64F64F64_TN>::CLayout{};

    warp_c(CLayout(lane_id, 0)) = c0_acc;
    warp_c(CLayout(lane_id, 1)) = c1_acc;
}

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


__global__ void checkArrayEquality(double* array1, double* array2, bool* result, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        if (array1[idx] != array2[idx]) {
            atomicExch((int*)result, int(false));
        }
    }
}

void assertEqual(double* A, double* B, int size) {
    bool h_result = true;
    bool* d_result;
    double *d_A, *d_B;

    cudaMalloc((void**)&d_result, sizeof(bool));
    cudaMalloc((void**)&d_A, size * sizeof(double));
    cudaMalloc((void**)&d_B, size * sizeof(double));
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
        time_t start, end;
        start = clock();
        trivialGemm<<<blocks, threads>>>(dA, dB, dC, m, n, k);
        cudaError_t err = cudaDeviceSynchronize();
        end = clock();
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        printf("Runtime of trivialGemm: %f\n", (double)(end - start) / CLOCKS_PER_SEC);
        cudaMemcpy(C_ref, dC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
    }

    {
        // test trivialMMAWithoutCuTe
        randn(C, m * n, 0, 100);
        dim3 threads(BM * BN / 2);
        dim3 blocks(m / BM, n / BN);
        time_t start, end;
        start = clock();
        trivialMMAWithoutCuTe<BM, BN, BK><<<blocks, threads>>>(dA, dB, dC, m, n, k);
        cudaError_t err = cudaDeviceSynchronize();
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        end = clock();
        printf("Runtime of trivialMMAWithoutCuTe: %f\n", (double)(end - start) / CLOCKS_PER_SEC);
        cudaMemcpy(C, dC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
        assertEqual(C, C_ref, m * n);
        printf("trivialMMAWithoutCuTe passed\n");
    }

    {
        // test trivialMMAWithCuTe
        randn(C, m * n, 0, 100);
        dim3 threads(BM * BN / 2);
        dim3 blocks(m / BM, n / BN);
        time_t start, end;
        start = clock();
        trivialMMAWithCuTe<BM, BN, BK><<<blocks, threads>>>(dA, dB, dC, m, n, k);
        cudaError_t err = cudaDeviceSynchronize();
        end = clock();
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        printf("Runtime of trivialMMAWithCuTe: %f\n", (double)(end - start) / CLOCKS_PER_SEC);
        cudaMemcpy(C, dC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
        assertEqual(C, C_ref, m * n);
        printf("trivialMMAWithCuTe passed\n");
    }

    {
        // test cuteStyleGemm
        using namespace cute;
        randn(C, m * n, 0, 100);
        auto tiled_mma = cute::make_tiled_mma(SM80_8x8x4_F64F64F64F64_TN{}, make_layout(Shape<Int<BM / 8>, Int<BN / 8>>{}));
        dim3 threads(size(tiled_mma));
        dim3 blocks(m / BM, n / BN);
        time_t start, end;
        start = clock();
        cuteStyleGemm<BM, BN, BK><<<blocks, threads>>>(dA, dB, dC, m, n, k, tiled_mma);
        cudaError_t err = cudaDeviceSynchronize();
        end = clock();
        printf("CUDA error: %s\n", cudaGetErrorString(err));
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