#include <metal_stdlib>
using namespace metal;

/*  (5) Define kernel — K = "Kernel"  -----------------------------
    vector_arithmetic(A,B,op) → C with op ⟮0=+ 1=− 2=* 3=/⟯
    Buffer indices:
        0 = A, 1 = B, 2 = C, 3 = N (uint), 4 = op (uint)
------------------------------------------------------------------------- */

kernel void vector_arithmetic(const device float *A [[buffer(0)]],
                              const device float *B [[buffer(1)]],
                              device       float *C [[buffer(2)]],
                              constant     uint  &N [[buffer(3)]],
                              constant     uint  &op[[buffer(4)]],
                              uint gid [[thread_position_in_grid]])
{
    if (gid >= N) return;
    switch (op) {
        case 0: C[gid] = A[gid] + B[gid]; break;
        case 1: C[gid] = A[gid] - B[gid]; break;
        case 2: C[gid] = A[gid] * B[gid]; break;
        case 3: C[gid] = A[gid] / B[gid]; break;
    }
}
