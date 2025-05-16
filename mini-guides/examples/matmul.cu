// matmul.cu - Simple matrix multiplication for CUDA
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Kernel function executed on the GPU
__global__ void matrixMul(float* A, float* B, float* C, int M, int N, int K) {
    // Calculate which output element this thread computes
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only do work if we're within the matrix dimensions
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Each thread computes one element of C by accumulating the product
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        C[row * N + col] = sum;
    }
}

// Helper function to initialize a matrix with random values
void randomInit(float* data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = rand() / (float)RAND_MAX;
    }
}

// Helper function to verify results match
bool verifyResults(float* gpuResults, float* cpuResults, int size) {
    const float epsilon = 1e-5f;
    int mismatchCount = 0;
    
    for (int i = 0; i < size; ++i) {
        if (fabs(gpuResults[i] - cpuResults[i]) > epsilon) {
            mismatchCount++;
            if (mismatchCount <= 10) {
                printf("Mismatch at %d: GPU = %f, CPU = %f\n", 
                       i, gpuResults[i], cpuResults[i]);
            }
        }
    }
    
    if (mismatchCount == 0) {
        printf("Results verified: GPU and CPU outputs match!\n");
        return true;
    } else {
        printf("Found %d mismatches between GPU and CPU results.\n", mismatchCount);
        return false;
    }
}

// CPU implementation for comparison
void matrixMulCPU(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    // Matrix dimensions
    const int M = 1024;  // Output matrix height
    const int N = 1024;  // Output matrix width
    const int K = 1024;  // Inner dimension
    
    printf("Matrix multiplication: (%dx%d) x (%dx%d)\n", M, K, K, N);
    
    // Memory allocation sizes
    size_t A_size = M * K * sizeof(float);
    size_t B_size = K * N * sizeof(float);
    size_t C_size = M * N * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float*)malloc(A_size);
    float *h_B = (float*)malloc(B_size);
    float *h_C = (float*)malloc(C_size);
    float *h_C_CPU = (float*)malloc(C_size);
    
    // Initialize matrices with random data
    srand(time(NULL));
    randomInit(h_A, M * K);
    randomInit(h_B, K * N);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, A_size);
    cudaMalloc(&d_B, B_size);
    cudaMalloc(&d_C, C_size);
    
    // Copy data from host to device
    cudaMemcpy(d_A, h_A, A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, B_size, cudaMemcpyHostToDevice);
    
    // Define the thread block size
    dim3 blockDim(16, 16);
    
    // Calculate grid dimensions
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, 
                 (M + blockDim.y - 1) / blockDim.y);
    
    printf("CUDA Grid: %d x %d blocks, each %d x %d threads\n", 
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start time
    cudaEventRecord(start);
    
    // Launch kernel on the GPU
    matrixMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    
    // Wait for GPU to finish
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, start, stop);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, C_size, cudaMemcpyDeviceToHost);
    
    printf("GPU execution time: %.2f ms\n", gpuTime);
    printf("First element of result: %f\n", h_C[0]);
    
    // Run on CPU for comparison
    printf("\nRunning on CPU...\n");
    clock_t cpu_start = clock();
    
    matrixMulCPU(h_A, h_B, h_C_CPU, M, N, K);
    
    clock_t cpu_end = clock();
    float cpuTime = 1000.0 * (float)(cpu_end - cpu_start) / CLOCKS_PER_SEC;
    
    printf("CPU execution time: %.2f ms\n", cpuTime);
    printf("First element of result: %f\n", h_C_CPU[0]);
    printf("Speedup: %.2fx\n", cpuTime / gpuTime);
    
    // Verify results match
    verifyResults(h_C, h_C_CPU, M * N);
    
    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_CPU);
    
    return 0;
} 