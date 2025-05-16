// dot_product_metal.swift — Swift 6 host (final, stride-bug fixed)
// ---------------------------------------------------------------
// Build:
//   xcrun -sdk macosx metal   -c dot_product.metal -o build/dp.air
//   xcrun -sdk macosx metallib build/dp.air -o build/dot_product.metallib
//   swiftc -O dot_product_metal.swift -o build/dot_product_metal -framework Metal
//
// Run (10 M elements):
//   ./build/dot_product_metal build/dot_product.metallib 10000000
// ---------------------------------------------------------------

import Foundation
import Metal

@inline(__always) func human(_ v: Double) -> String { String(format: "%.2f", v) }
func die(_ s: String) -> Never { fatalError(s) }

// ── 1. CLI ────────────────────────────────────────────────────────────────
let argv = CommandLine.arguments
guard argv.count == 3, let N64 = UInt64(argv[2]) else {
    print("Usage: \(argv[0]) <lib.metallib> <NUM_ELEMENTS>")
    exit(0)
}
let n = Int(N64)
let libURL = URL(fileURLWithPath: argv[1])

// ── 2. Metal boilerplate ─────────────────────────────────────────────────
guard let dev = MTLCreateSystemDefaultDevice() else { die("No Apple-silicon GPU") }
let queue = dev.makeCommandQueue()!
let lib   = try dev.makeLibrary(URL: libURL)
let pso   = try dev.makeComputePipelineState(function: lib.makeFunction(name: "dot_product")!)

// ── 3. Host-side vectors ────────────────────────────────────────────────
var hA = (0 ..< n).map { _ in Float.random(in: -1 ... 1) }
var hB = (0 ..< n).map { _ in Float.random(in: -1 ... 1) }

print("A[0:8] = \(hA.prefix(8).map { human(Double($0)) }.joined(separator: ", "))")
print("B[0:8] = \(hB.prefix(8).map { human(Double($0)) }.joined(separator: ", "))")
let dot8 = zip(hA.prefix(8), hB.prefix(8)).reduce(0.0) { $0 + Double($1.0 * $1.1) }
print("Dot[0:8] = \(human(dot8))")

// ── 4. Device buffers ───────────────────────────────────────────────────
let bytes = n * MemoryLayout<Float>.stride
let dA = dev.makeBuffer(bytes: &hA, length: bytes)!
let dB = dev.makeBuffer(bytes: &hB, length: bytes)!

let TPB    = 256
let groups = (n + TPB - 1) / TPB
let dBlocks = dev.makeBuffer(length: groups * MemoryLayout<Float>.stride)!
var n32 = UInt32(n)
let dN  = dev.makeBuffer(bytes: &n32, length: 4)!

// ── 5. Encode & run ─────────────────────────────────────────────────────
let cmd = queue.makeCommandBuffer()!
let enc = cmd.makeComputeCommandEncoder()!
enc.setComputePipelineState(pso)
enc.setBuffer(dA,      offset: 0, index: 0)
enc.setBuffer(dB,      offset: 0, index: 1)
enc.setBuffer(dBlocks, offset: 0, index: 2)
enc.setBuffer(dN,      offset: 0, index: 3)

let tgSize  = MTLSize(width: TPB, height: 1, depth: 1)
let gridSize = MTLSize(width: groups, height: 1, depth: 1)

let t0 = Date()
enc.dispatchThreadgroups(gridSize, threadsPerThreadgroup: tgSize)
enc.endEncoding()
cmd.commit(); cmd.waitUntilCompleted()
let ms = Date().timeIntervalSince(t0) * 1000

// ── 6. Finalize & report ────────────────────────────────────────────────
// Get results from the device
let blockResults = Array(UnsafeBufferPointer(start: dBlocks.contents().bindMemory(to: Float.self, capacity: groups), count: groups))

// Sum up all the block results
let gpuSum = blockResults.reduce(0.0) { $0 + Double($1) }

// Calculate CPU reference result
let cpuSum = zip(hA, hB).reduce(0.0) { $0 + Double($1.0 * $1.1) }

let gb = Double(bytes * 2) / Double(1 << 30)            // two vectors read
print("N = \(n)")
print("GPU sum = \(human(gpuSum))  |  CPU sum = \(human(cpuSum))")
print("Diff    = \(human(abs(gpuSum - cpuSum)))")
print("Kernel  = \(human(ms)) ms   (\(human(gb / (ms / 1000))) GB/s)")
