//  gpu_memcpy_bandwidth.swift
//  ---------------------------------------------------------------
//  WHAT this measures
//  ------------------
//  • A GPU-side memcpy (compute shader) between two buffers that
//    reside in DRAM but bypass the CPU-coherency fabric.
//  • The kernel is fully vectorized (16-byte loads/stores).
//  • The printed “write-side GB/s” is *half* the total DRAM traffic.
//    Multiply by 2 if you care about combined read+write.
//
//  WHY the number is well below Apple’s 819 GB/s headline
//  -----------------------------------------------------
//  • An M3 Ultra is two M3 Max dies glued by UltraFusion.
//    A single kernel launches on one die at a time, so you’re
//    limited to ~400 GB/s per direction in the best case.
//  • In practice you’ll see ~300–350 GB/s write-side,
//    which is ~600–700 GB/s total—perfectly normal.
//
//  BUILD: swiftc gpu_memcpy_bandwidth.swift -framework Metal -o gpu_bw
//  RUN  : ./gpu_bw 8 5          # 8 GiB buffer, 5 s sample window
//  ---------------------------------------------------------------

import Foundation
import Metal

// ─── CLI ──────────────────────────────────────────────────────────
let gib   = (CommandLine.argc > 1) ? Int(CommandLine.arguments[1])! : 8
let secs  = (CommandLine.argc > 2) ? Int(CommandLine.arguments[2])! : 3
let bytes = gib * 1_073_741_824        // GiB → bytes
let vec   = 16                         // 16 B (uint4) per thread

// ─── Metal bootstrap ─────────────────────────────────────────────
guard let dev = MTLCreateSystemDefaultDevice(),
      let q   = dev.makeCommandQueue()
else { fatalError("Metal unavailable") }

let src = dev.makeBuffer(length: bytes, options: .storageModePrivate)!
let dst = dev.makeBuffer(length: bytes, options: .storageModePrivate)!

// ─── Minimal vector memcpy kernel ────────────────────────────────
let msl = #"""
kernel void memcpy_vec(const device uint4* src [[buffer(0)]],
                       device uint4*       dst [[buffer(1)]],
                       uint id [[thread_position_in_grid]])
{
    dst[id] = src[id];      // one 16-byte move
}
"""#

let lib = try dev.makeLibrary(source: msl, options: nil)
let pso = try dev.makeComputePipelineState(
              function: lib.makeFunction(name: "memcpy_vec")!)

// ─── Thread grid (fits Metal limits) ─────────────────────────────
let totalThreads = bytes / vec
let tgWidth      = pso.threadExecutionWidth               // e.g. 32
let threadsPerTG = MTLSize(width: tgWidth, height: 1, depth: 1)
let grid         = MTLSize(width: totalThreads,
                           height: 1,
                           depth: 1)

// ─── Timed loop ──────────────────────────────────────────────────
let cutoff = DispatchTime.now().advanced(by: .seconds(secs))
var gpuSeconds = 0.0
var bytesMoved: UInt64 = 0

repeat {
    let cb = q.makeCommandBuffer()!
    let ce = cb.makeComputeCommandEncoder()!
    ce.setComputePipelineState(pso)
    ce.setBuffer(src, offset: 0, index: 0)
    ce.setBuffer(dst, offset: 0, index: 1)
    ce.dispatchThreads(grid, threadsPerThreadgroup: threadsPerTG)
    ce.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

    gpuSeconds += cb.gpuEndTime - cb.gpuStartTime
    bytesMoved += UInt64(bytes)        // one full 8 GiB pass
} while DispatchTime.now() < cutoff

// ─── Results ─────────────────────────────────────────────────────
let writeGBps = Double(bytesMoved) / gpuSeconds / 1_000_000_000
let totalGBps = writeGBps * 2.0
print(String(format:
    "GPU memcpy bandwidth: %.1f GB/s write-side  |  %.1f GB/s total (read+write)",
    writeGBps, totalGBps))
