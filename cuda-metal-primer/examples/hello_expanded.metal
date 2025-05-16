#include <metal_stdlib>
using namespace metal;

// NOTE: This kernel is demo-oriented.  We can't `printf` from Metal, so each thread stores
// its IDs into a results buffer that the CPU prints later.  Real-world kernels would perform
// math here rather than IO.

kernel void ant_battalion_report(device uint3* results [[buffer(0)]],
                                 constant uint& threadsPerThreadgroup [[buffer(1)]],
                                 uint   tid_in_tg [[thread_index_in_threadgroup]],
                                 uint   tg_id     [[threadgroup_position_in_grid]])
{
    // Calculate the global ant ID (similar to CUDA's blockIdx.x * blockDim.x + threadIdx.x)
    uint ant_id = tg_id * threadsPerThreadgroup + tid_in_tg;
    
    // Store the results to be read back on CPU
    // We can't directly print from the GPU in Metal, so we store values to display later
    results[ant_id] = uint3(ant_id, tg_id, tid_in_tg);
}