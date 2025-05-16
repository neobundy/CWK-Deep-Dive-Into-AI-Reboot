// matrix_add_metal.swift  — Swift 6 host (CUDA-style report)
// -----------------------------------------------------------
// Build:
//   xcrun -sdk macosx metal   -c matrix_add_metal.metal -o build/ma.air
//   xcrun -sdk macosx metallib build/ma.air -o build/matrix_add.metallib
//   swiftc -O matrix_add_metal.swift -o build/matrix_add_metal -framework Metal
//
// Run:
//   ./build/matrix_add_metal build/matrix_add.metallib <ROWS> <COLS>
// -----------------------------------------------------------

import Foundation
import Metal

@inline(__always)
func human(_ v: Double) -> String { String(format: "%.2f", v) }

// ---------- CLI -------------------------------------------------------------
let args = CommandLine.arguments
guard args.count == 4,
      let rows = UInt32(args[2]),
      let cols = UInt32(args[3]) else {
    print("Usage: \(args[0]) <lib.metallib> <ROWS> <COLS>")
    exit(0)
}
let ld   = cols
let elems = Int(rows * cols)
let bytes = elems * MemoryLayout<Float>.size
let libURL = URL(fileURLWithPath: args[1])

// ---------- Metal -----------------------------------------------------------
guard let dev = MTLCreateSystemDefaultDevice() else { fatalError("No GPU") }
let queue = dev.makeCommandQueue()!
let lib   = try dev.makeLibrary(URL: libURL)
let pso   = try dev.makeComputePipelineState(function: lib.makeFunction(name: "matrix_add")!)

// ---------- Host buffers ----------------------------------------------------
var hA = (0..<elems).map { _ in Float.random(in: -1...1) }
var hB = (0..<elems).map { _ in Float.random(in: -1...1) }
var hC = [Float](repeating: 0, count: elems)

// Print first 8 elements of first row
print("A[0,0:8] = \(hA.prefix(8).map { human(Double($0)) }.joined(separator: ", "))")
print("B[0,0:8] = \(hB.prefix(8).map { human(Double($0)) }.joined(separator: ", "))")

// ---------- Device buffers --------------------------------------------------
let dA = dev.makeBuffer(bytes: &hA, length: bytes)!
let dB = dev.makeBuffer(bytes: &hB, length: bytes)!
let dC = dev.makeBuffer(length: bytes)!
var rows32 = rows, cols32 = cols, ld32 = ld
let dRows = dev.makeBuffer(bytes: &rows32, length: 4)!
let dCols = dev.makeBuffer(bytes: &cols32, length: 4)!
let dLd   = dev.makeBuffer(bytes: &ld32,   length: 4)!

// ---------- Encode ----------------------------------------------------------
let cmd = queue.makeCommandBuffer()!
let enc = cmd.makeComputeCommandEncoder()!
enc.setComputePipelineState(pso)
enc.setBuffer(dA,   offset: 0, index: 0)
enc.setBuffer(dB,   offset: 0, index: 1)
enc.setBuffer(dC,   offset: 0, index: 2)
enc.setBuffer(dRows,offset: 0, index: 3)
enc.setBuffer(dCols,offset: 0, index: 4)
enc.setBuffer(dLd,  offset: 0, index: 5)

let tp = MTLSize(width: 16, height: 16, depth: 1)
let tg = MTLSize(width: (Int(cols)+15)/16,
                 height: (Int(rows)+15)/16,
                 depth: 1)

let t0 = Date()
enc.dispatchThreadgroups(tg, threadsPerThreadgroup: tp)
enc.endEncoding()
cmd.commit()
cmd.waitUntilCompleted()
let ms = Date().timeIntervalSince(t0) * 1000.0

// ---------- Copy back & verify ---------------------------------------------
memcpy(&hC, dC.contents(), bytes)
print("C[0,0:8] = \(hC.prefix(8).map { human(Double($0)) }.joined(separator: ", "))")

var ok = true; var checksum: Double = 0
for i in 0..<elems {
    if hC[i] != hA[i] + hB[i] { ok = false; break }
    checksum += Double(hC[i])
}

// ---------- Stats -----------------------------------------------------------
let gbMoved = Double(bytes * 3) / Double(1 << 30)
print(ok ? "Verification ✓" : "Verification ✗")
print("Checksum  = \(human(checksum))")
print(String(format: "Kernel   = %.2f ms   (%.2f GB/s)",
             ms, gbMoved / (ms / 1000.0)))
