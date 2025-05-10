// dot_product_cuda.cu
// -----------------------------------------------------------
// Stand-alone CUDA demo: dot product of two FP32 vectors
// Shows a warp-shuffle + shared-memory tree reduction.
//
//  Build:
//      nvcc -std=c++17 -arch=sm_89 dot_product_cuda.cu -o build/dot_product_cuda
//
//  Run (10 million elements):
//      ./build/dot_product_cuda 10000000
//
// -----------------------------------------------------------

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cinttypes>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>

// ------------------------------------------------------------
// Reduction kernel
//   • 256 threads per block
//   • Each thread reads one element per stride and accumulates
//   • In-block warp-shuffle reduction → shared-mem tree
// ------------------------------------------------------------
constexpr int TPB = 256;

__global__ void dotKernel(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float*       __restrict__ blockSums,
                          size_t N)
{
    __shared__ float smem[TPB];        // one partial per thread

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // --- 1. Parallel accumulate every stride ----------------
    float partial = 0.0f;
    for (size_t i = idx; i < N; i += stride)
        partial += A[i] * B[i];

    // --- 2. Warp-level reduction (shuffle) ------------------
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);

    // --- 3. One value per warp → shared memory --------------
    int lane   = threadIdx.x & 31;            // 0-31
    int warpId = threadIdx.x >> 5;            // warp index inside block
    if (lane == 0) smem[warpId] = partial;
    __syncthreads();

    // --- 4. First warp reduces the warp sums ----------------
    if (warpId == 0) {
        partial = (threadIdx.x < (blockDim.x / warpSize)) ? smem[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
        if (lane == 0) blockSums[blockIdx.x] = partial;   // one per block
    }
}

// ------------------------------------------------------------
// Host helpers
// ------------------------------------------------------------
static std::string human(double x) {
    char b[64]; snprintf(b, sizeof(b), "%.2f", x); return {b};
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <NUM_ELEMENTS>\n";
        return 0;
    }
    size_t N = std::atoll(argv[1]);
    size_t bytes = N * sizeof(float);

    // Host allocation & init
    std::vector<float> hA(N), hB(N);
    std::mt19937 rng(42); std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < N; ++i) { hA[i] = dist(rng); hB[i] = dist(rng); }

    // --- Debug: print first 8 elements of each vector ------------------
    std::cout << "A[0:8] = ";
    for (int i = 0; i < 8 && i < static_cast<int>(N); ++i)
        std::cout << human(hA[i]) << (i == 7 ? "\n" : ", ");
    std::cout << "B[0:8] = ";
    for (int i = 0; i < 8 && i < static_cast<int>(N); ++i)
        std::cout << human(hB[i]) << (i == 7 ? "\n" : ", ");

    // Dot product of first 8 elements
    double dot8 = 0.0;
    for (int i = 0; i < 8 && i < static_cast<int>(N); ++i)
        dot8 += double(hA[i]) * double(hB[i]);
    std::cout << "Dot[0:8] = " << human(dot8) << "\n";

    // Device allocation & H→D
    float *dA, *dB, *dBlock;
    cudaMalloc(&dA, bytes);  cudaMalloc(&dB, bytes);
    int numBlocks = (N + TPB - 1) / TPB;
    cudaMalloc(&dBlock, numBlocks * sizeof(float));
    cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice);

    // Launch
    auto t0 = std::chrono::high_resolution_clock::now();
    dotKernel<<<numBlocks, TPB>>>(dA, dB, dBlock, N);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    // Copy block sums back & finalize on host
    std::vector<float> hBlock(numBlocks);
    cudaMemcpy(hBlock.data(), dBlock, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    double gpuSum = std::accumulate(hBlock.begin(), hBlock.end(), 0.0);

    // CPU reference
    double cpuSum = 0.0;
    for (size_t i = 0; i < N; ++i) cpuSum += double(hA[i]) * double(hB[i]);

    // Timing & bandwidth
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double gb = 2.0 * bytes / (1ULL << 30);         // two vectors read
    double gbps = gb / (ms / 1000.0);

    std::cout << "N = " << N << "  |  ";
    std::cout << "GPU sum = " << human(gpuSum) << "  |  CPU sum = " << human(cpuSum) << "\n";
    std::cout << "Diff = " << human(std::abs(gpuSum - cpuSum)) << "\n";
    std::cout << "Kernel time : " << human(ms) << " ms   ("
              << human(gbps) << " GB/s)\n";

    cudaFree(dA); cudaFree(dB); cudaFree(dBlock);
    return 0;
}
