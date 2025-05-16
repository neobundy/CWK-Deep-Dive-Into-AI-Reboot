#include <metal_stdlib>
using namespace metal;

/*  (5) Define kernel — K = "Kernel"  ---------------------------
    vector_add(A,B) → C : C[gid] = A[gid] + B[gid]
    Buffer indices (match Swift host):
        0 = A, 1 = B, 2 = C, 3 = N (uint)
------------------------------------------------------------------------- */
kernel void vector_add(const device float *A [[buffer(0)]],
                       const device float *B [[buffer(1)]],
                       device       float *C [[buffer(2)]],
                       constant     uint  &N [[buffer(3)]],
                       uint gid [[thread_position_in_grid]])
{
    if (gid < N) C[gid] = A[gid] + B[gid];
}
