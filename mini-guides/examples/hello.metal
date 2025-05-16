#include <metal_stdlib>
using namespace metal;

kernel void hello(device uint2       *out    [[buffer(0)]],
                  uint  tid_in_tg    [[thread_index_in_threadgroup]],
                  uint  tg_id_in_g   [[threadgroup_position_in_grid]])
{
    const uint threadsPerTG = 4;                     // <-- constant
    uint gid = tg_id_in_g * threadsPerTG + tid_in_tg;
    out[gid] = uint2(tg_id_in_g, tid_in_tg);
}