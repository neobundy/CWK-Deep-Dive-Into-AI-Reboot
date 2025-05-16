//
//  vector_add_metal.swift  – Swift 6
//  Build & run (command-line):
//      mkdir -p build
//      xcrun -sdk macosx metal -c vector_add_metal.metal -o build/vector_add_metal.air
//      xcrun -sdk macosx metallib build/vector_add_metal.air -o build/vector_add_metal.metallib
//      swiftc -O vector_add_metal.swift -o build/vector_add_metal -framework Metal
//      # Capacity report
//      ./build/vector_add_metal build/vector_add_metal.metallib
//      # Add integers 1‥1024
//      ./build/vector_add_metal build/vector_add_metal.metallib 1 1024
//

import Foundation
import Metal

// ── Helpers ─────────────────────────────────────────────────────────────
func human(_ bytes: UInt64) -> String {
    String(format: "%.2f GiB", Double(bytes) / Double(1 << 30))
}
func check(_ cond: Bool, _ msg: String) { if !cond { fatalError(msg) } }

// ── CLI parsing ─────────────────────────────────────────────────────────
let args = CommandLine.arguments
guard args.count == 2 || args.count == 4 else {
    print("Usage: \(args[0]) <lib.metallib> [<START> <END>]"); exit(1)
}
let libURL = URL(fileURLWithPath: args[1])

// ── Metal setup ─────────────────────────────────────────────────────────
let dev  = MTLCreateSystemDefaultDevice()!
let q    = dev.makeCommandQueue()!
let lib  = try dev.makeLibrary(URL: libURL)
let fn   = lib.makeFunction(name: "vector_add")!
let pso  = try dev.makeComputePipelineState(function: fn)

// ── Capacity-report mode ────────────────────────────────────────────────
if args.count == 2 {
    let maxWS: UInt64 = dev.recommendedMaxWorkingSetSize          // UInt64
    let used  = UInt64(dev.currentAllocatedSize)                  // cast Int → UInt64
    let free  = maxWS > used ? maxWS - used : 0
    let safe  = UInt64(Double(free) * 0.80)                       // 80 % head-room
    let elt   = UInt64(MemoryLayout<Float>.size)
    let nMax  = safe / (3 * elt)                                  // 3 buffers in flight

    let tpb   = 256
    let tgs   = (nMax + UInt64(tpb) - 1) / UInt64(tpb)

    print("""
          \n=== GPU Capacity Report ===
          GPU model              : \(dev.name)
          Recommended working set: \(human(maxWS))
          Current allocated      : \(human(used))
          Free estimate          : \(human(free))
          Element type           : float32 (4 bytes)
          Resident buffers       : 3
          Safe usable bytes      : \(human(safe))  (80 % of est. free)
          Max vector length (N)  : \(nMax)
          Launch suggestion      : \(tgs) groups × \(tpb) threads
          ===========================\n
          """)
    exit(0)
}

// ── Vector-add mode ────────────────────────────────────────────────────
let start = UInt64(args[2])!
let end   = UInt64(args[3])!
check(end >= start, "END must be ≥ START")

let N      = Int(end - start + 1)
let bytes  = N * MemoryLayout<Float>.size
print("Creating vector with \(N) elements (\(human(UInt64(bytes))) per buffer)")

// (1) Allocate host memory — A
// Host buffers
var hA = (0..<N).map { Float(start + UInt64($0)) }
// (2) Initialize vectors — I  (hA already filled; hB identical)
var hB = hA                                             // identical
var hC = [Float](repeating: 0, count: N)

// (3) Allocate device memory & (4) Copy Host → Device — O
// Device buffers
let dA = dev.makeBuffer(bytes: &hA, length: bytes)!
let dB = dev.makeBuffer(bytes: &hB, length: bytes)!
let dC = dev.makeBuffer(length: bytes)!
var n32 = UInt32(N)
let dN  = dev.makeBuffer(bytes: &n32, length: MemoryLayout<UInt32>.size)!

// (5) Kernel defined in Metal shader (vector_add_metal.metal)

// (6) Launch the kernel — L
// Encode
let cmd = q.makeCommandBuffer()!
let enc = cmd.makeComputeCommandEncoder()!
enc.setComputePipelineState(pso)
enc.setBuffer(dA, offset: 0, index: 0)
enc.setBuffer(dB, offset: 0, index: 1)
enc.setBuffer(dC, offset: 0, index: 2)
enc.setBuffer(dN, offset: 0, index: 3)

let tpb = MTLSize(width: 256, height: 1, depth: 1)
let tgs = MTLSize(width: (N + 255) / 256, height: 1, depth: 1)
let t0  = Date()
enc.dispatchThreadgroups(tgs, threadsPerThreadgroup: tpb)
enc.endEncoding()
cmd.commit(); cmd.waitUntilCompleted()
let ms  = Date().timeIntervalSince(t0) * 1000

// (7) Copy Device → Host — L (device→Local)
memcpy(&hC, dC.contents(), bytes)

// (8) Validate results — O (Observe / Verify)
var ok  = true; var sum = 0.0
for i in 0..<N {
    let expect = 2 * hA[i]
    if hC[i] != expect { ok = false; break }
    sum += Double(hC[i])
}
let expectedSum = Double(start + end) * Double(N)

// Print summary … (timing, bandwidth) …
// (9) Garbage-collect — G (handled automatically by ARC at scope exit)

print(ok ? "Verification passed ✓" : "Verification FAILED ✗")
print(String(format: "Kernel time            : %.2f ms", ms))
let gbMoved = Double(3 * bytes) / Double(1 << 30)
print(String(format: "Effective bandwidth    : %.2f GiB/s", gbMoved / (ms / 1000)))
print("Sum of result elements : \(Int(sum))  (expected \(Int(expectedSum)))")
