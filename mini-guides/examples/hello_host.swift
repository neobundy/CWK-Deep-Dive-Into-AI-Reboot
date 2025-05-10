import Metal
import Foundation

func main() throws {
    // ── 1 · Device & library ───────────────────────────────────
    guard let device = MTLCreateSystemDefaultDevice() else {
        fatalError("No Metal-capable GPU found")
    }
    let libURL   = URL(fileURLWithPath: "hello.metallib")
    let library  = try device.makeLibrary(URL: libURL)      // modern API
    let function = library.makeFunction(name: "hello")!
    let pipeline = try device.makeComputePipelineState(function: function)

    // ── 2 · Launch geometry ───────────────────────────────────
    let threadsPerTG  = 4
    let threadgroups  = 1
    let totalThreads  = threadsPerTG * threadgroups

    guard let outBuf = device.makeBuffer(length: totalThreads *
                                         MemoryLayout<SIMD2<UInt32>>.stride,
                                         options: .storageModeShared) else {
        fatalError("Buffer allocation failed")
    }

    // ── 3 · Encode & submit command buffer ────────────────────
    let queue          = device.makeCommandQueue()!
    let commandBuffer  = queue.makeCommandBuffer()!
    let encoder        = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(outBuf, offset: 0, index: 0)

    let tgSize  = MTLSize(width: threadsPerTG, height: 1, depth: 1)
    let grid    = MTLSize(width: threadgroups, height: 1, depth: 1)
    encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: tgSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    // ── 4 · Read results back on the CPU ──────────────────────
    let data = outBuf.contents()
                     .bindMemory(to: SIMD2<UInt32>.self, capacity: totalThreads)

    for i in 0..<totalThreads {
        print("Hello from threadgroup \(data[i].x), thread \(data[i].y)")
    }
}

do { try main() }
catch { print("Error: \(error)"); exit(1) }
