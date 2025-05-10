// vector_add_cuda.cu
// -----------------------------------------------------------
// Self-diagnosing CUDA vector-addition demo (C++17)
// Adds two identical vectors constructed from an integer range.
// Prints capacity info (no args) or runs the add, verifies, and
// now also prints the **sum of all result elements**.
//
//  Build (Linux / WSL / macOS):
//      mkdir -p build
//      nvcc -std=c++17 -arch=sm_89 vector_add_cuda.cu -o build/vector_add_cuda
//
//  Run:
//      ./build/vector_add_cuda             # capacity report only
//      ./build/vector_add_cuda 1 1024      # add integers 1..1024 element-wise
// -----------------------------------------------------------

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cinttypes>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <numeric>   // std::accumulate fallback if needed
#include <cmath>     // fabsf

// -----------------------------------------------------------------------------
// (5) Define kernel — K = "Kernel" (element-wise C = A + B)
__global__ void vectorAdd(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float*       __restrict__ C,
                          size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

// -----------------------------------------------------------------------------
static std::string humanBytes(size_t bytes)
{
    char buf[64];
    double gib = static_cast<double>(bytes) / (1ULL << 30);
    snprintf(buf, sizeof(buf), "%.2f GiB", gib);
    return {buf};
}

// -----------------------------------------------------------------------------
static int run_capacity_report()
{
    int dev = 0; cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    int driverVer = 0, runtimeVer = 0;
    cudaDriverGetVersion(&driverVer); cudaRuntimeGetVersion(&runtimeVer);

    size_t freeB = 0, totalB = 0; cudaMemGetInfo(&freeB, &totalB);

    constexpr int kBuffers = 3; constexpr size_t kEltBytes = sizeof(float);
    size_t safeBytes = static_cast<size_t>(freeB * 0.90);
    size_t Nmax = safeBytes / (kBuffers * kEltBytes);

    const int threadsPerBlock = 256;
    size_t blocks = (Nmax + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "\n=== GPU Capacity Report ===\n"
              << "GPU model              : " << prop.name << "\n"
              << "CUDA driver/runtime    : " << driverVer/1000 << '.' << (driverVer%100)
              << " / " << runtimeVer/1000 << '.' << (runtimeVer%100) << "\n"
              << "Total VRAM             : " << humanBytes(totalB) << "\n"
              << "Free VRAM (runtime)    : " << humanBytes(freeB)  << "\n"
              << "Element type           : float32 (4 bytes)\n"
              << "Resident buffers       : " << kBuffers << "\n"
              << "Safe usable bytes      : " << humanBytes(safeBytes) << " (90 % of free)\n"
              << "Max vector length (N)  : " << Nmax << " elements\n"
              << "Suggested launch shape : " << blocks << " blocks × " << threadsPerBlock << " threads\n"
              << "===========================\n\n";
    return 0;
}

// -----------------------------------------------------------------------------
static int run_vector_add(long long start, long long end)
{
    if (end < start) { std::cerr << "Error: END must be ≥ START\n"; return 1; }
    size_t N = static_cast<size_t>(end - start + 1);
    size_t bytes = N * sizeof(float);
    std::cout << "Creating vector with " << N << " elements (" << humanBytes(bytes) << " per buffer)\n";

    // (1) Allocate host memory — A
    float *h_A = static_cast<float*>(malloc(bytes));
    float *h_B = static_cast<float*>(malloc(bytes));
    float *h_C = static_cast<float*>(malloc(bytes));
    if (!h_A || !h_B || !h_C) { std::cerr << "Host malloc failed\n"; return 1; }

    // (2) Initialize vectors — I
    for (size_t i = 0; i < N; ++i) {
        float v = static_cast<float>(start + i);
        h_A[i] = v; h_B[i] = v;
    }

    // (3) Allocate device memory — O (first half)
    float *d_A=nullptr,*d_B=nullptr,*d_C=nullptr;
    cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);

    // (4) Copy Host → Device — O (second half)
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // (6) Launch the kernel — L
    const int threadsPerBlock = 256;
    int blocks = static_cast<int>((N + threadsPerBlock - 1) / threadsPerBlock);
    auto t0 = std::chrono::high_resolution_clock::now();
    vectorAdd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    // (7) Copy Device → Host — L (device→Local)
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // (8) Validate results — O (Observe / Verify)
    constexpr float kEps = 1e-4f;                    // tolerance for float32
    bool ok = true; double sum = 0.0;
    for (size_t i = 0; i < N; ++i) {
        float expected = 2.0f * h_A[i];
        if (fabsf(h_C[i] - expected) > kEps) {       // tolerant compare
            ok = false; break;
        }
        sum += h_C[i];
    }

    // Expected sum formula: 2 * (start+end) * N / 2 = (start+end)*N
    double expectedSum = static_cast<double>(start + end) * static_cast<double>(N);

    // Timing & bandwidth
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double gbMoved = (3.0 * bytes) / (1ULL << 30);
    double gbps = gbMoved / (ms / 1000.0);

    std::cout << (ok ? "Verification passed ✓" : "Verification FAILED ✗") << '\n';
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Kernel time            : " << ms  << " ms\n";
    std::cout << "Effective bandwidth    : " << gbps << " GiB/s\n";
    std::cout << std::setprecision(0);
    std::cout << "Sum of result elements : " << sum << " (expected " << expectedSum << ")\n";

    // (9) Garbage-collect both sides — G
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return ok ? 0 : 1;
}

// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    if (argc == 1) return run_capacity_report();
    if (argc == 3) return run_vector_add(std::atoll(argv[1]), std::atoll(argv[2]));
    std::cout << "Usage: " << argv[0] << " [<START> <END>]\n       (no args → capacity report)\n";
    return 0;
}
