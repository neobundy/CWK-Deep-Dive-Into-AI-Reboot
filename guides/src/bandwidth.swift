//  Bandwidth.swift
//  swiftc bandwidth.swift -framework Metal -o bandwidth
//  ./bandwidth 8 5   # 8 GiB buffer, 5 s test

import Foundation
import Metal

let gib     = (CommandLine.argc > 1) ? Int(CommandLine.arguments[1])! : 8
let secs    = (CommandLine.argc > 2) ? Int(CommandLine.arguments[2])! : 3
let bytes   = gib * 1_073_741_824   // GiB → bytes
let inflight = 4                    // #cmd buffers pipelined

guard let dev = MTLCreateSystemDefaultDevice(),
      let q   = dev.makeCommandQueue() else { fatalError() }

let src = dev.makeBuffer(length: bytes, options: .storageModeShared)!
let dst = dev.makeBuffer(length: bytes, options: .storageModePrivate)!

memset(src.contents(), 0x5A, 1)     // commit pages

let end = DispatchTime.now().advanced(by: .seconds(secs))
var copied: UInt64 = 0

while DispatchTime.now() < end {
    var bufs: [MTLCommandBuffer] = []
    for _ in 0..<inflight {
        let cb = q.makeCommandBuffer()!
        let bl = cb.makeBlitCommandEncoder()!
        bl.copy(from: src, sourceOffset: 0,
                to:   dst, destinationOffset: 0,
                size: bytes)
        bl.endEncoding()
        cb.commit()
        bufs.append(cb)
    }
    bufs.forEach { $0.waitUntilCompleted() }
    copied += UInt64(bytes * inflight)
}

let gbps = Double(copied) / Double(secs) / 1_000_000_000
print(String(format: "Sustained bandwidth: %.1f GB/s", gbps))
