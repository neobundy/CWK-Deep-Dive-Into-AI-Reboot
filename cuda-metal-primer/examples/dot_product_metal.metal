#include <metal_stdlib>
using namespace metal;

kernel void dot_product(
    device const float *A        [[buffer(0)]],
    device const float *B        [[buffer(1)]],
    device float *blockResults   [[buffer(2)]],
    device uint  &N              [[buffer(3)]],
    
    uint  thIdx  [[thread_index_in_threadgroup]],
    uint  gIdx   [[thread_position_in_grid]],
    uint  bIdx   [[threadgroup_position_in_grid]],
    uint  gridDim[[threadgroups_per_grid]],
    uint  blockDim[[threads_per_threadgroup]])
{
    // Use threadgroup memory for reduction
    threadgroup float temp[256];
    
    // Calculate starting position and stride
    uint stride = gridDim * blockDim;
    
    // Each thread calculates its partial sum
    float threadSum = 0.0f;
    for (uint i = gIdx; i < N; i += stride) {
        threadSum += A[i] * B[i];
    }
    
    // First pass: store each thread's sum
    temp[thIdx] = threadSum;
    
    // Make sure all threads have stored their sum
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction in shared memory
    for (uint s = blockDim/2; s > 0; s >>= 1) {
        if (thIdx < s) {
            temp[thIdx] += temp[thIdx + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result for this block
    if (thIdx == 0) {
        // Store in output buffer
        blockResults[bIdx] = temp[0];
    }
}
