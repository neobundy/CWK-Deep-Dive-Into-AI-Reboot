// matrix_ops_metal_blas.swift
// -----------------------------------------------------------
// Modes (BLAS):  (prepend --fp16 to run GEMV / GEMM in half-precision)
//   axpy <N> <alpha>            # y = alpha * x + y   (BLAS-1, Accelerate)
//   dot  <N>                    # x·y                 (BLAS-1, Accelerate)
//   gemv <ROWS> <COLS>          # y = A·x             (BLAS-2, MPS)
//   gemm <M> <N> <K>            # C = A·B             (BLAS-3, MPS)
//
// Build (ILP-64 Accelerate >= macOS 13.3):
//   swiftc matrix_ops_metal_blas.swift \
//       -o build/matrix_ops_metal_blas \
//       -Xcc -DACCELERATE_NEW_LAPACK -Xcc -DACCELERATE_LAPACK_ILP64 \
//       -framework Metal -framework MetalPerformanceShaders -framework Accelerate
// -----------------------------------------------------------

import Foundation
import Metal
import MetalPerformanceShaders
import Accelerate

func human(_ v: Double) -> String { String(format: "%.2f", v) }

// Preview first 8 elements of any floating-point array (Float, Float16…)
func preview<T: BinaryFloatingPoint>(_ tag: String, _ a: [T]) {
    print(tag, terminator: "")
    for i in 0..<min(8, a.count) {
        print(human(Double(a[i])) + (i == 7 ? "\n" : ","), terminator: "")
    }
}

var raw = CommandLine.arguments
let useFP16 = raw.contains("--fp16")
raw.removeAll { $0 == "--fp16" }
let args = raw

if args.count < 2 {
    print("Usage: matrix_ops_metal_blas [--fp16] <mode> ...\n  axpy N alpha\n  dot N\n  gemv ROWS COLS\n  gemm M N K")
    exit(0)
}
let mode = args[1]
let rng = { () -> Float in Float.random(in: -1...1) }

// BLAS-1 on CPU via Accelerate
if mode == "axpy" {
    guard args.count == 4, let N = Int(args[2]), let alpha = Float(args[3]) else { print("args: N alpha"); exit(1) }
    var x = (0..<N).map { _ in rng() }
    var y = (0..<N).map { _ in rng() }
    preview("y[0:8] before = ", y)
    let start = CFAbsoluteTimeGetCurrent()
    cblas_saxpy(N, alpha, &x, 1, &y, 1)
    let ms = (CFAbsoluteTimeGetCurrent() - start)*1000
    preview("y[0:8] after  = ", y)
    print("Kernel = \(human(ms)) ms  (\(human(Double(N)*2/1e6/(ms/1000))) MFLOP/s)")
}
else if mode == "dot" {
    guard args.count == 3, let N = Int(args[2]) else { print("args: N"); exit(1) }
    var x = (0..<N).map { _ in rng() }
    var y = (0..<N).map { _ in rng() }
    let start = CFAbsoluteTimeGetCurrent()
    let result: Float = cblas_sdot(N, &x, 1, &y, 1)
    let ms = (CFAbsoluteTimeGetCurrent() - start)*1000
    print("Dot = \(result)\nKernel = \(human(ms)) ms  (\(human(Double(N)*2/1e6/(ms/1000))) MFLOP/s)")
}
// GPU path using MPS
else {
    guard let device = MTLCreateSystemDefaultDevice() else { fatalError("No GPU") }
    let queue = device.makeCommandQueue()!

    // Upload data into fast device-private buffers
    func makePrivateBuffer<T>(_ array: [T]) -> MTLBuffer {
        let byteCount = array.count * MemoryLayout<T>.stride
        // Staging buffer in shared memory
        let staging: MTLBuffer = array.withUnsafeBytes { raw in
            device.makeBuffer(bytes: raw.baseAddress!, length: byteCount, options: .storageModeShared)!
        }
        // Private buffer on GPU-only memory
        let gpuBuf  = device.makeBuffer(length: byteCount, options: .storageModePrivate)!
        let blitCB  = queue.makeCommandBuffer()!
        let blitEnc = blitCB.makeBlitCommandEncoder()!
        blitEnc.copy(from: staging, sourceOffset: 0, to: gpuBuf, destinationOffset: 0, size: byteCount)
        blitEnc.endEncoding()
        blitCB.commit(); blitCB.waitUntilCompleted()
        return gpuBuf
    }

    if mode == "gemv" {
        guard args.count == 4, let R = Int(args[2]), let C = Int(args[3]) else { print("args: ROWS COLS"); exit(1) }
        var hy: [Float] = []
        if useFP16 {
            let hA16: [Float16] = (0..<R*C).map { _ in Float16.random(in: -1...1) }
            let hx16: [Float16] = (0..<C).map { _ in Float16.random(in: -1...1) }
            let bufA = makePrivateBuffer(hA16)
            let bufX = makePrivateBuffer(hx16)
            let bufY = device.makeBuffer(length: R*2, options: .storageModeShared)!
            let descA = MPSMatrixDescriptor(rows: R, columns: C, rowBytes: C*2, dataType: .float16)
            let descX = MPSVectorDescriptor(length: C, dataType: .float16)
            let descY = MPSVectorDescriptor(length: R, dataType: .float16)
            let matA = MPSMatrix(buffer: bufA, descriptor: descA)
            let vecX = MPSVector(buffer: bufX, descriptor: descX)
            let vecY = MPSVector(buffer: bufY, descriptor: descY)
            let op = MPSMatrixVectorMultiplication(device: device, transpose: false, rows: R, columns: C, alpha: 1, beta: 0)
            let cmd = queue.makeCommandBuffer()!
            let startCPU = CFAbsoluteTimeGetCurrent()
            op.encode(commandBuffer: cmd, inputMatrix: matA, inputVector: vecX, resultVector: vecY)
            cmd.commit(); cmd.waitUntilCompleted()
            let gpu = (cmd.gpuEndTime - cmd.gpuStartTime)*1000
            let wall = (CFAbsoluteTimeGetCurrent() - startCPU)*1000
            var hy16 = [Float16](repeating: 0, count: R)
            memcpy(&hy16, bufY.contents(), R*2)
            preview("y[0:8] = ", hy16)
            let flops = 2.0*Double(R)*Double(C)
            print("GPU = \(human(gpu)) ms  (\(human(flops/1e9/(gpu/1000))) GFLOP/s) | Wall = \(human(wall)) ms")
        } else {
            let hA = (0..<R*C).map { _ in rng() }
            let hx  = (0..<C).map { _ in rng() }
            hy = [Float](repeating: 0, count: R)
            let bufA = makePrivateBuffer(hA)
            let bufX = makePrivateBuffer(hx)
            let bufY = device.makeBuffer(length: R*4, options: .storageModeShared)!
            let descA = MPSMatrixDescriptor(rows: R, columns: C, rowBytes: C*4, dataType: .float32)
            let descX = MPSVectorDescriptor(length: C, dataType: .float32)
            let descY = MPSVectorDescriptor(length: R, dataType: .float32)
            let matA = MPSMatrix(buffer: bufA, descriptor: descA)
            let vecX = MPSVector(buffer: bufX, descriptor: descX)
            let vecY = MPSVector(buffer: bufY, descriptor: descY)
            let op = MPSMatrixVectorMultiplication(device: device, transpose: false, rows: R, columns: C, alpha: 1, beta: 0)
            let cmd = queue.makeCommandBuffer()!
            let startCPU = CFAbsoluteTimeGetCurrent()
            op.encode(commandBuffer: cmd, inputMatrix: matA, inputVector: vecX, resultVector: vecY)
            cmd.commit(); cmd.waitUntilCompleted()
            let gpu = (cmd.gpuEndTime - cmd.gpuStartTime)*1000
            let wall = (CFAbsoluteTimeGetCurrent() - startCPU)*1000
            memcpy(&hy, bufY.contents(), R*4)
            preview("y[0:8] = ", hy)
            let flops = 2.0*Double(R)*Double(C)
            print("GPU = \(human(gpu)) ms  (\(human(flops/1e9/(gpu/1000))) GFLOP/s) | Wall = \(human(wall)) ms")
        }
    }
    else if mode == "gemm" {
        guard args.count == 5,
              let M = Int(args[2]), let N = Int(args[3]), let K = Int(args[4]) else { print("args: M N K"); exit(1) }
        var hC = [Float](repeating: 0, count: M*K)
        if useFP16 {
            let hA16: [Float16] = (0..<M*N).map { _ in Float16.random(in: -1...1) }
            let hB16: [Float16] = (0..<N*K).map { _ in Float16.random(in: -1...1) }
            let bufA = makePrivateBuffer(hA16)
            let bufB = makePrivateBuffer(hB16)
            let bufC = device.makeBuffer(length: M*K*2, options: .storageModeShared)!
            let descA = MPSMatrixDescriptor(rows: M, columns: N, rowBytes: N*2, dataType: .float16)
            let descB = MPSMatrixDescriptor(rows: N, columns: K, rowBytes: K*2, dataType: .float16)
            let descC = MPSMatrixDescriptor(rows: M, columns: K, rowBytes: K*2, dataType: .float16)
            let matA = MPSMatrix(buffer: bufA, descriptor: descA)
            let matB = MPSMatrix(buffer: bufB, descriptor: descB)
            let matC = MPSMatrix(buffer: bufC, descriptor: descC)
            let gemm = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false, resultRows: M, resultColumns: K, interiorColumns: N, alpha: 1, beta: 0)
            let cmd = queue.makeCommandBuffer()!
            let startCPU = CFAbsoluteTimeGetCurrent()
            gemm.encode(commandBuffer: cmd, leftMatrix: matA, rightMatrix: matB, resultMatrix: matC)
            cmd.commit(); cmd.waitUntilCompleted()
            let gpu = (cmd.gpuEndTime - cmd.gpuStartTime)*1000
            let wall = (CFAbsoluteTimeGetCurrent() - startCPU)*1000
            var hC16 = [Float16](repeating: 0, count: M*K)
            memcpy(&hC16, bufC.contents(), M*K*2)
            preview("C[0,0:8] = ", hC16)
            let flops = 2.0*Double(M)*Double(N)*Double(K)
            print("GPU = \(human(gpu)) ms  (\(human(flops/1e12/(gpu/1000))) TFLOP/s) | Wall = \(human(wall)) ms")
        } else {
            let hA = (0..<M*N).map { _ in rng() }
            let hB = (0..<N*K).map { _ in rng() }
            let bufA = makePrivateBuffer(hA)
            let bufB = makePrivateBuffer(hB)
            let bufC = device.makeBuffer(length: M*K*4, options: .storageModeShared)!
            let descA = MPSMatrixDescriptor(rows: M, columns: N, rowBytes: N*4, dataType: .float32)
            let descB = MPSMatrixDescriptor(rows: N, columns: K, rowBytes: K*4, dataType: .float32)
            let descC = MPSMatrixDescriptor(rows: M, columns: K, rowBytes: K*4, dataType: .float32)
            let matA = MPSMatrix(buffer: bufA, descriptor: descA)
            let matB = MPSMatrix(buffer: bufB, descriptor: descB)
            let matC = MPSMatrix(buffer: bufC, descriptor: descC)
            let gemm = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false, resultRows: M, resultColumns: K, interiorColumns: N, alpha: 1, beta: 0)
            let cmd = queue.makeCommandBuffer()!
            let startCPU = CFAbsoluteTimeGetCurrent()
            gemm.encode(commandBuffer: cmd, leftMatrix: matA, rightMatrix: matB, resultMatrix: matC)
            cmd.commit(); cmd.waitUntilCompleted()
            let gpu = (cmd.gpuEndTime - cmd.gpuStartTime)*1000
            let wall = (CFAbsoluteTimeGetCurrent() - startCPU)*1000
            memcpy(&hC, bufC.contents(), M*K*4)
            preview("C[0,0:8] = ", hC)
            let flops = 2.0*Double(M)*Double(N)*Double(K)
            print("GPU = \(human(gpu)) ms  (\(human(flops/1e12/(gpu/1000))) TFLOP/s) | Wall = \(human(wall)) ms")
        }
    }
    else {
        print("Unknown mode")
    }
}