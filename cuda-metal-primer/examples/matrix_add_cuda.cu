// matrix_add_cuda.cu  (uniform report format)
// -----------------------------------------------------------
// CUDA demo: element-wise addition C = A + B (row-major FP32).
// Prints first 8 elements of A[0,*], B[0,*], C[0,*] exactly
// like the vector demo prints its first 8 values.
//
//  Build:
//      nvcc -std=c++17 -arch=sm_89 matrix_add_cuda.cu -o build/matrix_add_cuda
//
//  Run (rows cols):
//      ./build/matrix_add_cuda 4096 4096
// -----------------------------------------------------------

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <iomanip>

constexpr int TPB_X = 16;
constexpr int TPB_Y = 16;

// -----------------------------------------------------------------------------
// Row-major C = A + B
// -----------------------------------------------------------------------------
__global__ void matAddKernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float*       __restrict__ C,
                             int rows, int cols, int ld)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols)
        C[row * ld + col] = A[row * ld + col] + B[row * ld + col];
}

// -----------------------------------------------------------------------------
// Helper
// -----------------------------------------------------------------------------
static inline std::string human(double v) {
    char b[32]; std::snprintf(b, sizeof(b), "%.2f", v); return b;
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <ROWS> <COLS>\n"; return 0;
    }
    const int rows = std::atoi(argv[1]);
    const int cols = std::atoi(argv[2]);
    const int ld   = cols;                         // row-major
    const size_t elems  = size_t(rows) * cols;
    const size_t bytes  = elems * sizeof(float);

    std::vector<float> hA(elems), hB(elems), hC(elems);
    std::mt19937 rng(42); std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (size_t i = 0; i < elems; ++i) { hA[i] = dist(rng); hB[i] = dist(rng); }

    // --- Preview first 8 elements of the first row --------------------------
    std::cout << "A[0,0:8] = ";
    for (int i = 0; i < 8 && i < cols; ++i)
        std::cout << human(hA[i]) << (i == 7 ? "\n" : ", ");
    std::cout << "B[0,0:8] = ";
    for (int i = 0; i < 8 && i < cols; ++i)
        std::cout << human(hB[i]) << (i == 7 ? "\n" : ", ");

    // ------------------------------------------------------------------------
    // Device work
    // ------------------------------------------------------------------------
    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes); cudaMalloc(&dB, bytes); cudaMalloc(&dC, bytes);
    cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice);

    dim3 blk(TPB_X, TPB_Y);
    dim3 grd((cols + TPB_X - 1) / TPB_X,
             (rows + TPB_Y - 1) / TPB_Y);

    auto t0 = std::chrono::high_resolution_clock::now();
    matAddKernel<<<grd, blk>>>(dA, dB, dC, rows, cols, ld);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost);

    // --- Preview C first 8 ---------------------------------------------------
    std::cout << "C[0,0:8] = ";
    for (int i = 0; i < 8 && i < cols; ++i)
        std::cout << human(hC[i]) << (i == 7 ? "\n" : ", ");

    // Verify & checksum
    bool ok = true; double checksum = 0.0;
    for (size_t i = 0; i < elems; ++i) {
        if (hC[i] != hA[i] + hB[i]) { ok = false; break; }
        checksum += hC[i];
    }

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double gbMoved = 3.0 * bytes / (1ULL << 30);          // read A,B + write C

    std::cout << (ok ? "Verification ✓" : "Verification ✗") << '\n'
              << "Checksum  = " << human(checksum) << '\n'
              << "Kernel    = " << human(ms) << " ms   ("
              << human(gbMoved / (ms / 1000.0)) << " GB/s)" << '\n';

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return ok ? 0 : 1;
}
