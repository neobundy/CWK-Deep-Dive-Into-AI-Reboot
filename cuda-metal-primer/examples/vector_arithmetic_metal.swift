//
//  vector_arithmetic_metal.swift  — Swift 6
//
//  Build:
//      mkdir -p build
//      xcrun -sdk macosx metal -c vector_arithmetic_metal.metal -o build/va_metal.air
//      xcrun -sdk macosx metallib build/va_metal.air -o build/vector_arithmetic_metal.metallib
//      swiftc -O vector_arithmetic_metal.swift -o build/vector_arithmetic_metal -framework Metal
//
//  Run:
//      ./build/vector_arithmetic_metal build/vector_arithmetic_metal.metallib mul 1 10
//

import Foundation
import Metal

// ── CLI parsing ─────────────────────────────────────────────────────────
enum Op: UInt32 { case add = 0, sub = 1, mul = 2, div = 3 }
let opMap: [String: Op] = ["add": .add, "sub": .sub, "mul": .mul, "div": .div]

let a = CommandLine.arguments
guard a.count == 5,
      let op = opMap[a[2]],
      let start = UInt64(a[3]),
      let end   = UInt64(a[4]),
      end >= start else
{
    print("Usage: \(a[0]) <lib.metallib> <add|sub|mul|div> <START> <END>")
    exit(1)
}
let libURL = URL(fileURLWithPath: a[1])

// ── Metal setup ─────────────────────────────────────────────────────────
let dev  = MTLCreateSystemDefaultDevice()!
let queue = dev.makeCommandQueue()!
let lib   = try dev.makeLibrary(URL: libURL)
let fn    = lib.makeFunction(name: "vector_arithmetic")!
let pso   = try dev.makeComputePipelineState(function: fn)

// ── Host & device buffers ───────────────────────────────────────────────
// (1) Allocate host memory — A + (2) Initialize — I
let N     = Int(end - start + 1)
let bytes = N * MemoryLayout<Float>.size

var hA = (0..<N).map { Float(start + UInt64($0)) }
var hB = (0..<N).map { Float(end   - UInt64($0)) }
if op == .div { for i in 0..<N where hB[i] == 0 { hB[i] = 1 } }

// (3) Allocate device memory & (4) Copy Host → Device — O
let dA  = dev.makeBuffer(bytes: &hA, length: bytes)!
let dB  = dev.makeBuffer(bytes: &hB, length: bytes)!
let dC  = dev.makeBuffer(length: bytes)!
var n32 = UInt32(N);   let dN  = dev.makeBuffer(bytes: &n32, length: 4)!
var op32 = op.rawValue; let dOp = dev.makeBuffer(bytes: &op32, length: 4)!

// (5) Kernel defined in Metal shader (vector_arithmetic_metal.metal)

// (6) Launch the kernel — L
// ── Encode & dispatch ───────────────────────────────────────────────────
let cb  = queue.makeCommandBuffer()!
let enc = cb.makeComputeCommandEncoder()!
enc.setComputePipelineState(pso)
enc.setBuffer(dA,  offset: 0, index: 0)
enc.setBuffer(dB,  offset: 0, index: 1)
enc.setBuffer(dC,  offset: 0, index: 2)
enc.setBuffer(dN,  offset: 0, index: 3)
enc.setBuffer(dOp, offset: 0, index: 4)

let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
let numGroups       = MTLSize(width: (N + 255) / 256, height: 1, depth: 1)

let t0 = Date()
enc.dispatchThreadgroups(numGroups, threadsPerThreadgroup: threadsPerGroup)
enc.endEncoding()
cb.commit()
cb.waitUntilCompleted()
let ms = Date().timeIntervalSince(t0) * 1000.0

// (7) Copy Device → Host — L (device→Local)
var hC = [Float](repeating: 0, count: N)
memcpy(&hC, dC.contents(), bytes)

// (8) Validate results — O (Observe / Verify)
let eps: Float = 1e-4
var ok  = true
var checksum = 0.0
for i in 0..<N {
    let expect: Float
    switch op {
        case .add: expect = hA[i] + hB[i]
        case .sub: expect = hA[i] - hB[i]
        case .mul: expect = hA[i] * hB[i]
        case .div: expect = hA[i] / hB[i]
    }
    if abs(hC[i] - expect) > eps { ok = false; break }
    checksum += Double(hC[i])
}

// (9) Garbage-collect — G (handled by ARC)

// ── Pretty-print summary (CUDA-style order) ─────────────────────────────
print("A[:8] =", hA.prefix(8).map(Int.init))
print("B[:8] =", hB.prefix(8).map(Int.init))
print("C[:8] =", hC.prefix(8).map(Int.init))
print("Checksum =", Int(checksum))

let gbMoved = Double(3 * bytes) / Double(1 << 30)
let summary = String(format: "%@  |  %.2f ms", ok ? "Verification ✓" : "Verification ✗", ms)
print(summary)

print(String(format: "Effective bandwidth    : %.2f GiB/s", gbMoved / (ms / 1000)))
