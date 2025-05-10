// matrix_ops_metal.metal
// -----------------------------------------------------------
// Kernels: matrix_ew  (Add/Sub/Mul/Div)
//          matrix_gemv
//          matrix_gemm_naive
//          matrix_gemm_tiled
//   • Row-major buffers
//   • op enum: 0=add 1=sub 2=mul 3=div
// -----------------------------------------------------------

#include <metal_stdlib>
using namespace metal;

/* ==================== Element-wise ==================== */
kernel void matrix_ew(
    const device float *A [[buffer(0)]],
    const device float *B [[buffer(1)]],
    device       float *C [[buffer(2)]],
    constant     uint  &ROWS [[buffer(3)]],
    constant     uint  &COLS [[buffer(4)]],
    constant     uint  &LD   [[buffer(5)]],
    constant     uint  &op   [[buffer(6)]],
    uint2 tid   [[thread_position_in_threadgroup]],
    uint2 gid   [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    if (row < ROWS && col < COLS) {
        uint idx = row * LD + col;
        float a = A[idx], b = B[idx];
        float out = (op == 0 ? a + b :
                     op == 1 ? a - b :
                     op == 2 ? a * b : a / b);
        C[idx] = out;
    }
}

/* ==================== GEMV ============================= */
kernel void matrix_gemv(
    const device float *A [[buffer(0)]],
    const device float *x [[buffer(1)]],
    device       float *y [[buffer(2)]],
    constant     uint  &ROWS [[buffer(3)]],
    constant     uint  &COLS [[buffer(4)]],
    constant     uint  &LD   [[buffer(5)]],
    uint  lid   [[thread_index_in_threadgroup]],
    uint3 tgPos [[threadgroup_position_in_grid]])
{
    /* one thread-group per row, 256 threads per TG */
    threadgroup float xs[256];
    uint row = tgPos.x;
    if (row >= ROWS) return;

    float acc = 0.0f;
    for (uint tile = 0; tile < COLS; tile += 256) {
        uint col = tile + lid;
        if (col < COLS) xs[lid] = x[col];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (col < COLS)
            acc += A[row * LD + col] * xs[lid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // warp-style reduction in shared memory
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lid < stride) xs[lid] += xs[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) y[row] = xs[0] + acc;   // xs[0] already has partial sum
}

/* ==================== GEMM – naive ==================== */
kernel void matrix_gemm_naive(
    const device float *A [[buffer(0)]],
    const device float *B [[buffer(1)]],
    device       float *C [[buffer(2)]],
    constant     uint  &M [[buffer(3)]],
    constant     uint  &N [[buffer(4)]],
    constant     uint  &K [[buffer(5)]],
    constant     uint  &lda [[buffer(6)]],
    constant     uint  &ldb [[buffer(7)]],
    constant     uint  &ldc [[buffer(8)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    if (row >= M || col >= K) return;

    float acc = 0.0f;
    for (uint n = 0; n < N; ++n)
        acc += A[row * lda + n] * B[n * ldb + col];
    C[row * ldc + col] = acc;
}

/* ==================== GEMM – tiled ==================== */
#define TILE 128
#define TK    8

kernel void matrix_gemm_tiled(
    const device float *A [[buffer(0)]],
    const device float *B [[buffer(1)]],
    device       float *C [[buffer(2)]],
    constant     uint  &M [[buffer(3)]],
    constant     uint  &N [[buffer(4)]],
    constant     uint  &K [[buffer(5)]],
    constant     uint  &lda [[buffer(6)]],
    constant     uint  &ldb [[buffer(7)]],
    constant     uint  &ldc [[buffer(8)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgPos[[threadgroup_position_in_grid]])
{
    const uint rowsPerTG = TILE;
    const uint colsPerTG = TILE;


    threadgroup float As[TILE][TK];
    threadgroup float Bs[TK][TILE];

    uint globalRow = tgPos.y * rowsPerTG + tid.y;
    uint globalCol = tgPos.x * colsPerTG + tid.x;
    float acc = 0.0f;

    for (uint t = 0; t < N; t += TK) {
        uint aRow = globalRow, aCol = t + tid.x;
        uint bRow = t + tid.y,  bCol = globalCol;

        As[tid.y][tid.x] = (aRow < M && aCol < N) ? A[aRow * lda + aCol] : 0.0f;
        Bs[tid.y][tid.x] = (bRow < N && bCol < K) ? B[bRow * ldb + bCol] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint k = 0; k < TK; ++k)
            acc += As[tid.y][k] * Bs[k][tid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (globalRow < M && globalCol < K)
        C[globalRow * ldc + globalCol] = acc;
}
