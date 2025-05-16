// matmul.swift - Matrix multiplication example for Metal on Apple Silicon
import Metal
import Foundation
// import Accelerate  // Uncomment if you want BLAS timing instead of naïve loops

// For timing the execution
func measureTime(_ block: () -> Void) -> TimeInterval {
    let start = Date()
    block()
    return Date().timeIntervalSince(start)
}

// Matrix dimensions
let M = 1024  // Output matrix height
let N = 1024  // Output matrix width
let K = 1024  // Inner dimension

// Initialize input matrices with random data
func createRandomMatrix(rows: Int, cols: Int) -> [Float] {
    var matrix = [Float](repeating: 0.0, count: rows * cols)
    for i in 0..<rows*cols {
        matrix[i] = Float.random(in: 0...1)
    }
    return matrix
}

let A = createRandomMatrix(rows: M, cols: K)
let B = createRandomMatrix(rows: K, cols: N)
var C = [Float](repeating: 0.0, count: M * N)

print("Matrix multiplication: (\(M)×\(K)) × (\(K)×\(N))")
print("Creating Metal device, buffers, and command queue...")

// Create the Metal device and command queue
guard let device = MTLCreateSystemDefaultDevice(),
      let commandQueue = device.makeCommandQueue() else {
    fatalError("GPU setup failed - no Metal device available")
}

// Create the compute pipeline
let metalLibraryPath = Bundle.main.path(forResource: "default", ofType: "metallib")
let library: MTLLibrary
do {
    // First check if we're running from an Xcode bundle
    if let metalLibraryPath = metalLibraryPath {
        if #available(macOS 13.0, *) {
            // Use newer non-deprecated API for macOS 13+
            let metalLibURL = URL(fileURLWithPath: metalLibraryPath)
            library = try device.makeLibrary(URL: metalLibURL)
        } else {
            // Fallback for older macOS versions
            library = try device.makeLibrary(filepath: metalLibraryPath)
        }
    } else {
        // Try to load from local MetalLib directory (command line execution)
        let localMetalLibPath = "MetalLib/default.metallib"
        
        if FileManager.default.fileExists(atPath: localMetalLibPath) {
            if #available(macOS 13.0, *) {
                // Use newer non-deprecated API for macOS 13+
                let metalLibURL = URL(fileURLWithPath: localMetalLibPath)
                library = try device.makeLibrary(URL: metalLibURL)
            } else {
                // Fallback for older macOS versions
                library = try device.makeLibrary(filepath: localMetalLibPath)
            }
        } else {
            // For command-line tools, compile from source as last resort
            let kernelCode = """
            #include <metal_stdlib>
            using namespace metal;

            kernel void matrixMul(
                device const float* A [[buffer(0)]],
                device const float* B [[buffer(1)]],
                device float* C [[buffer(2)]],
                constant int& M [[buffer(3)]],
                constant int& N [[buffer(4)]],
                constant int& K [[buffer(5)]],
                uint2 position [[thread_position_in_grid]]
            ) {
                // Get the output position (same as CUDA's row, col)
                int row = position.y;
                int col = position.x;
                
                // Only do work if we're within the matrix dimensions
                if (row < M && col < N) {
                    float sum = 0.0f;
                    
                    // Each thread computes one output element
                    for (int k = 0; k < K; k++) {
                        sum += A[row * K + k] * B[k * N + col];
                    }
                    
                    C[row * N + col] = sum;
                }
            }
            """
            library = try device.makeLibrary(source: kernelCode, options: nil)
        }
    }
} catch {
    fatalError("Failed to create Metal library: \(error)")
}

guard let matrixMul = library.makeFunction(name: "matrixMul"),
      let pipelineState = try? device.makeComputePipelineState(function: matrixMul) else {
    fatalError("Failed to create pipeline state")
}

// Create buffers for matrices (use shared memory on Apple Silicon)
let aBuffer = device.makeBuffer(bytes: A, length: M * K * MemoryLayout<Float>.stride, options: .storageModeShared)!
let bBuffer = device.makeBuffer(bytes: B, length: K * N * MemoryLayout<Float>.stride, options: .storageModeShared)!
let cBuffer = device.makeBuffer(length: M * N * MemoryLayout<Float>.stride, options: .storageModeShared)!

// Create buffers for dimensions
var mInt32 = Int32(M)
var nInt32 = Int32(N)
var kInt32 = Int32(K)
let mBuffer = device.makeBuffer(bytes: &mInt32, length: MemoryLayout<Int32>.size, options: .storageModeShared)!
let nBuffer = device.makeBuffer(bytes: &nInt32, length: MemoryLayout<Int32>.size, options: .storageModeShared)!
let kBuffer = device.makeBuffer(bytes: &kInt32, length: MemoryLayout<Int32>.size, options: .storageModeShared)!

print("Running on GPU: \(device.name)")
let executionTime = measureTime {
    // Create a command buffer and compute encoder
    let commandBuffer = commandQueue.makeCommandBuffer()!
    let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
    computeEncoder.setComputePipelineState(pipelineState)
    
    // Set the buffers
    computeEncoder.setBuffer(aBuffer, offset: 0, index: 0)
    computeEncoder.setBuffer(bBuffer, offset: 0, index: 1)
    computeEncoder.setBuffer(cBuffer, offset: 0, index: 2)
    computeEncoder.setBuffer(mBuffer, offset: 0, index: 3)
    computeEncoder.setBuffer(nBuffer, offset: 0, index: 4)
    computeEncoder.setBuffer(kBuffer, offset: 0, index: 5)
    
    // Calculate grid and threadgroup sizes
    let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
    let gridSize = MTLSize(
        width: N,
        height: M,
        depth: 1
    )
    
    // Dispatch the threads!
    computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
    computeEncoder.endEncoding()
    
    // Execute and wait for completion
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
}

// Copy results back to our array
let resultsPtr = cBuffer.contents().bindMemory(to: Float.self, capacity: M * N)
C = Array(UnsafeBufferPointer(start: resultsPtr, count: M * N))

print("GPU execution time: \(executionTime * 1000) ms")
print("First element: \(C[0])")

// Now let's compare with CPU
print("\nRunning on CPU (naïve triple-loop)…")
var cpuC = [Float](repeating: 0.0, count: M * N)

let cpuTime = measureTime {
    for i in 0..<M {
        for j in 0..<N {
            var sum: Float = 0.0
            for k in 0..<K {
                sum += A[i * K + k] * B[k * N + j]
            }
            cpuC[i * N + j] = sum
        }
    }
}

print("CPU execution time: \(cpuTime * 1000) ms")
print("First element: \(cpuC[0])")
print("Speedup: \(cpuTime / executionTime)x")

// Verify results match
let epsilon: Float = 1e-5
var mismatchCount = 0
for i in 0..<min(10, M * N) {
    if abs(C[i] - cpuC[i]) > epsilon {
        mismatchCount += 1
        print("Mismatch at \(i): GPU = \(C[i]), CPU = \(cpuC[i])")
    }
}
if mismatchCount == 0 {
    print("Results verified: GPU and CPU outputs match!")
} 