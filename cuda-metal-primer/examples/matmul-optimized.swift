// matmul-optimized.swift – vectorized CPU vs naïve GPU example
// -------------------------------------------------------------
// This variant mirrors matmul.swift but swaps the CPU triple-loop for
// Accelerate's vDSP_mmul (SIMD-vectorized, multi-threaded) so readers can
// compare an optimized CPU baseline against the same simple one-thread-per-element
// Metal kernel.
//
// Usage (from project root):
//    swiftc mini-guides/examples/matmul-optimized.swift -o matmul_opt && ./matmul_opt
//
// Adjust M/N/K to larger sizes (e.g. 4096) if the CPU overtakes the GPU at 1024.

import Metal
import Foundation
import Accelerate

// MARK: — Timing helper
func measureTime(_ block: () -> Void) -> TimeInterval {
    let start = Date()
    block()
    return Date().timeIntervalSince(start)
}

// MARK: — Matrix dimensions
let M = 1024  // height of A and C (try 4096 for clearer GPU win)
let N = 1024  // width  of B and C
let K = 1024  // shared inner dim

// MARK: — Generate random input data
func randomMatrix(rows: Int, cols: Int) -> [Float] {
    (0..<rows*cols).map { _ in Float.random(in: 0...1) }
}

let A = randomMatrix(rows: M, cols: K)
let B = randomMatrix(rows: K, cols: N)
var C_gpu = [Float](repeating: 0, count: M * N)

print("Matrix multiplication: (\(M)×\(K)) × (\(K)×\(N))")
print("Creating Metal device, buffers, and command queue…")

// MARK: — GPU setup
let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!

// Compile the Metal kernel (reuse default.metallib if present)
let libURL = URL(fileURLWithPath: "MetalLib/default.metallib")
let library = (try? device.makeLibrary(URL: libURL)) ?? {
    // Fallback: compile simple kernel from string at runtime
    let src = """
#include <metal_stdlib>
using namespace metal;
kernel void matrixMul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint2 pos [[thread_position_in_grid]]) {
    int row = pos.y;
    int col = pos.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
"""
    return try! device.makeLibrary(source: src, options: nil)
}()
let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "matrixMul")!)

// Buffers
let aBuf = device.makeBuffer(bytes: A, length: A.count * 4, options: .storageModeShared)!
let bBuf = device.makeBuffer(bytes: B, length: B.count * 4, options: .storageModeShared)!
let cBuf = device.makeBuffer(length: C_gpu.count * 4, options: .storageModeShared)!

// Dimension buffers
var m32 = Int32(M), n32 = Int32(N), k32 = Int32(K)
let mBuf = device.makeBuffer(bytes: &m32, length: 4, options: .storageModeShared)!
let nBuf = device.makeBuffer(bytes: &n32, length: 4, options: .storageModeShared)!
let kBuf = device.makeBuffer(bytes: &k32, length: 4, options: .storageModeShared)!

// MARK: — Run on GPU
print("Running on GPU: \(device.name)")
let gpuTime = measureTime {
    let cmd = commandQueue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(aBuf, offset: 0, index: 0)
    enc.setBuffer(bBuf, offset: 0, index: 1)
    enc.setBuffer(cBuf, offset: 0, index: 2)
    enc.setBuffer(mBuf, offset: 0, index: 3)
    enc.setBuffer(nBuf, offset: 0, index: 4)
    enc.setBuffer(kBuf, offset: 0, index: 5)

    let tgSize = MTLSize(width: 16, height: 16, depth: 1)
    let grid   = MTLSize(width: N, height: M, depth: 1)
    enc.dispatchThreads(grid, threadsPerThreadgroup: tgSize)
    enc.endEncoding()
    cmd.commit(); cmd.waitUntilCompleted()
}

let cPtr = cBuf.contents().bindMemory(to: Float.self, capacity: C_gpu.count)
C_gpu = Array(UnsafeBufferPointer(start: cPtr, count: C_gpu.count))
print("GPU execution time: \(gpuTime * 1000) ms")
print("First element: \(C_gpu[0])")

// MARK: — Optimized CPU baseline (vDSP)
print("\nRunning on CPU (vDSP)…")
var C_cpu = [Float](repeating: 0, count: M * N)
let cpuTime = measureTime {
    A.withUnsafeBufferPointer { aPtr in
        B.withUnsafeBufferPointer { bPtr in
            C_cpu.withUnsafeMutableBufferPointer { cPtr in
                vDSP_mmul(
                    aPtr.baseAddress!, 1,
                    bPtr.baseAddress!, 1,
                    cPtr.baseAddress!, 1,
                    vDSP_Length(M),
                    vDSP_Length(N),
                    vDSP_Length(K))
            }
        }
    }
}
print("CPU execution time: \(cpuTime * 1000) ms")
print("First element: \(C_cpu[0])")
print("Speedup: \(cpuTime / gpuTime)x\n")

// Basic verification
let eps: Float = 1e-4
let matched = zip(C_gpu, C_cpu).prefix(10).allSatisfy { abs($0 - $1) < eps }
print(matched ? "Results verified on first 10 elements." : "WARNING: mismatch detected.") 