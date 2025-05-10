// matrix_add_metal.metal
// -----------------------------------------------------------
// Element-wise addition of two row-major FP32 matrices.
// Buffer bindings:
//   0 = A, 1 = B, 2 = C, 3 = rows (uint), 4 = cols (uint), 5 = ld (uint)
// -----------------------------------------------------------

#include <metal_stdlib>
using namespace metal;

kernel void matrix_add(const device float *A [[buffer(0)]],
                       const device float *B [[buffer(1)]],
                       device       float *C [[buffer(2)]],
                       constant     uint  &rows [[buffer(3)]],
                       constant     uint  &cols [[buffer(4)]],
                       constant     uint  &ld   [[buffer(5)]],
                       uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    if (row < rows && col < cols) {
        uint idx = row * ld + col;          // row-major address
        C[idx] = A[idx] + B[idx];
    }
}
