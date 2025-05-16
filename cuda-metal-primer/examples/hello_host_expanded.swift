import Foundation
import Metal

func main() {
    // Command line arguments
    let args = CommandLine.arguments
    
    if args.count != 3 {
        print("Usage: \(args[0]) <num_threadgroups> <threads_per_threadgroup>")
        print("Example: \(args[0]) 2 4 (launches 2 threadgroups with 4 threads each)")
        exit(1)
    }
    
    // Parse command line arguments
    guard let numThreadgroups = Int(args[1]), let threadsPerThreadgroup = Int(args[2]),
          numThreadgroups > 0, threadsPerThreadgroup > 0 else {
        print("Error: Both arguments must be positive integers")
        exit(1)
    }
    
    // Apple GPUs top out at 1024 threads per threadgroup in most cases—keep demos well below that
    // to avoid runtime validation errors.
    
    // Setup Metal device
    guard let device = MTLCreateSystemDefaultDevice() else {
        print("Error: No Metal-capable GPU found")
        exit(1)
    }
    
    do {
        // Load the Metal library and create the compute pipeline
        let libURL = URL(fileURLWithPath: "hello_expanded.metallib")
        let library = try device.makeLibrary(URL: libURL)        
        guard let function = library.makeFunction(name: "ant_battalion_report") else {
            print("Error: Failed to find the Metal kernel function")
            exit(1)
        }
        
        let computePipelineState = try device.makeComputePipelineState(function: function)
        
        // Calculate total number of ants and create buffer for results
        let totalAnts = numThreadgroups * threadsPerThreadgroup
        let bufferSize = totalAnts * MemoryLayout<SIMD3<UInt32>>.stride
        
        guard let resultsBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
            print("Error: Failed to create results buffer")
            exit(1)
        }
        
        // Create buffer for threadsPerThreadgroup constant
        var threadCount = UInt32(threadsPerThreadgroup)
        guard let threadCountBuffer = device.makeBuffer(bytes: &threadCount,
                                                      length: MemoryLayout<UInt32>.size,
                                                      options: .storageModeShared) else {
            print("Error: Failed to create thread count buffer")
            exit(1)
        }
        
        // Display battalion formation details
        print("\n==== ASSIGNMENT OF ANT BATTALION ====")
        print("Drill Sergeant: \"ATTENTION! Forming \(numThreadgroups) teams with \(threadsPerThreadgroup) ants each!\"")
        print("Drill Sergeant: \"TOTAL FORCE: \(totalAnts) ants ready for deployment!\"\n")
        
        // Create command queue and command buffer
        guard let commandQueue = device.makeCommandQueue() else {
            print("Error: Failed to create command queue")
            exit(1)
        }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            print("Error: Failed to create command buffer")
            exit(1)
        }
        
        // Create compute command encoder
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            print("Error: Failed to create compute command encoder")
            exit(1)
        }
        
        // Configure and dispatch the kernel
        computeEncoder.setComputePipelineState(computePipelineState)
        computeEncoder.setBuffer(resultsBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(threadCountBuffer, offset: 0, index: 1)
        
        let threadsPerGrid = MTLSize(width: threadsPerThreadgroup, height: 1, depth: 1)
        let gridSize = MTLSize(width: numThreadgroups, height: 1, depth: 1)
        
        computeEncoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadsPerGrid)
        computeEncoder.endEncoding()
        
        // Commit the command buffer and wait for completion
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Read and display the results
        let resultsPtr = resultsBuffer.contents().bindMemory(to: SIMD3<UInt32>.self, capacity: totalAnts)
        let results = UnsafeBufferPointer(start: resultsPtr, count: totalAnts)
        
        for result in results {
            let antId = result.x
            let teamId = result.y
            let positionId = result.z
            print("Ten-hut! Private Ant \(antId) reporting from Block \(teamId), Position \(positionId), Sir!")
        }
        
        print("\nDrill Sergeant: \"AT EASE! All ants accounted for!\"")
        print("==== END OF ROLL CALL ====\n")
        
    } catch {
        print("Error: \(error)")
        exit(1)
    }
}

main()