#include <cstdio>
#include <cuda_runtime.h>

// ── GPU kernel ───────────────────────────────────────────
// Prints its grid & thread coordinates (needs CUDA 3.2+).
__global__ void say_hello()
{
    printf("Hello from block %d, thread %d\n",
           blockIdx.x, threadIdx.x);
}

int main()
{
    // Launch 1 block with 4 threads
    say_hello<<<1, 4>>>();
    cudaDeviceSynchronize();          // wait for GPU printf to finish
    return 0;
}
