#include <metal_stdlib>
using namespace metal;

kernel void matrixMul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint2 position [[thread_position_in_grid]]
) {
    // Get the output position (same as CUDA's row, col)
    int row = position.y;
    int col = position.x;
    
    // Only do work if we're within the matrix dimensions
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Each thread computes one output element
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        C[row * N + col] = sum;
    }
} 