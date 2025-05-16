#include <cstdio>
#include <cuda_runtime.h>
#include <cstdlib>

// ── GPU kernel ───────────────────────────────────────────
// Our ant workers report for duty with military precision
// NOTE: GPU printf is convenient for demos **only**—it stalls warps and should be avoided in perf-critical kernels.
__global__ void ant_battalion_report()
{
    int ant_id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Ten-hut! Private Ant %d reporting from Block %d, Position %d, Sir!\n", 
           ant_id, blockIdx.x, threadIdx.x);
}

int main(int argc, char **argv)
{
    // Flush host stdout immediately so each GPU printf line appears as soon as the kernel flushes it.
    // This makes the out-of-order arrival pattern more obvious across OSes.
    setbuf(stdout, NULL);
    // Check for command line arguments
    if (argc != 3) {
        printf("Usage: %s <num_blocks> <threads_per_block>\n", argv[0]);
        printf("Example: %s 2 4 (launches 2 blocks with 4 threads each)\n", argv[0]);
        return 1;
    }
    
    // Parse command line arguments
    int num_blocks = atoi(argv[1]);
    int threads_per_block = atoi(argv[2]);
    
    // Typical Ampere/Ada GPUs cap at 1024 threads per block; exceeding that triggers a launch error.
    // We keep it simple here but be mindful when you scale this demo.
    
    // Validate inputs
    if (num_blocks <= 0 || threads_per_block <= 0) {
        printf("Error: Both arguments must be positive integers\n");
        return 1;
    }
    
    // Display battalion formation details
    printf("\n==== ASSIGNMENT OF ANT BATTALION ====\n");
    printf("Drill Sergeant: \"ATTENTION! Forming %d teams with %d ants each!\"\n", 
           num_blocks, threads_per_block);
    printf("Drill Sergeant: \"TOTAL FORCE: %d ants ready for deployment!\"\n\n", 
           num_blocks * threads_per_block);
    
    // Launch kernel with specified configuration
    ant_battalion_report<<<num_blocks, threads_per_block>>>();
    
    // Wait for all ants to report
    cudaDeviceSynchronize();
    
    printf("\nDrill Sergeant: \"AT EASE! All ants accounted for!\"\n");
    printf("==== END OF ROLL CALL ====\n\n");
    
    // Check for any CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    return 0;
}