// matrix_ops_metal.swift  — compile-clean Swift 6 host
// -----------------------------------------------------------
// Build:
//   xcrun -sdk macosx metal   -c matrix_ops_metal.metal -o build/mo.air
//   xcrun -sdk macosx metallib build/mo.air -o build/matrix_ops.metallib
//   swiftc -O matrix_ops_metal.swift -o build/matrix_ops_metal -framework Metal
// -----------------------------------------------------------

import Foundation
import Metal

// ── helpers ────────────────────────────────────────────────
@inline(__always) func human(_ v: Double,_ u:String="") -> String {
    String(format: "%.2f%@", v, u)
}
func rng(_ n:Int) -> [Float] { (0..<n).map{ _ in Float.random(in:-1...1) } }
func preview(_ tag:String, _ a:[Float]) {
    let vals = a.prefix(8).map{ human(Double($0)) }.joined(separator: ", ")
    print("\(tag)\(vals)")
}

// ── CLI ────────────────────────────────────────────────────
let arg = CommandLine.arguments
guard arg.count >= 4 else {
    print("""
      Usage: \(arg[0]) <lib.metallib> <op> args…
        add | sub | mul | div  <ROWS> <COLS>
        gemv                   <ROWS> <COLS>
        gemm-naive | gemm      <M> <N> <K>
      """); exit(0)
}
let libURL = URL(fileURLWithPath: arg[1]); let op = arg[2]

// ── Metal setup ────────────────────────────────────────────
let dev   = MTLCreateSystemDefaultDevice()!
let queue = dev.makeCommandQueue()!
let lib   = try dev.makeLibrary(URL: libURL)

let psoEW   = try dev.makeComputePipelineState(function: lib.makeFunction(name:"matrix_ew")!)
let psoGEMV = try dev.makeComputePipelineState(function: lib.makeFunction(name:"matrix_gemv")!)
let psoGN   = try dev.makeComputePipelineState(function: lib.makeFunction(name:"matrix_gemm_naive")!)
let psoGT   = try dev.makeComputePipelineState(function: lib.makeFunction(name:"matrix_gemm_tiled")!)

// shorthand
func buf(_ v: inout UInt32) -> MTLBuffer { dev.makeBuffer(bytes:&v,length:4)! }

// ── EW kernel runner ───────────────────────────────────────
func runEW(rows:UInt32, cols:UInt32, opEnum:UInt32) {
    let e = Int(rows*cols), bytes = e*4
    var hA=rng(e), hB=rng(e), hC=[Float](repeating:0,count:e)

    preview("A[0,0:8] = ", hA)
    preview("B[0,0:8] = ", hB)

    let dA=dev.makeBuffer(bytes:&hA,length:bytes)!
    let dB=dev.makeBuffer(bytes:&hB,length:bytes)!
    let dC=dev.makeBuffer(length:bytes)!
    var r=rows,c=cols,ld=cols,op=opEnum
    let bufs:[MTLBuffer]=[dA,dB,dC,buf(&r),buf(&c),buf(&ld),buf(&op)]

    let cmd=queue.makeCommandBuffer()!
    let enc=cmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(psoEW)
    for(i,b) in bufs.enumerated(){ enc.setBuffer(b,offset:0,index:i) }
    let tpt=MTLSize(width:16,height:16,depth:1)
    let tg = MTLSize(width:(Int(cols)+15)/16,height:(Int(rows)+15)/16,depth:1)
    let t0=Date()
    enc.dispatchThreadgroups(tg,threadsPerThreadgroup:tpt)
    enc.endEncoding();cmd.commit();cmd.waitUntilCompleted()
    let ms=Date().timeIntervalSince(t0)*1000

    memcpy(&hC,dC.contents(),bytes)
    preview("C[0,0:8] = ", hC)

    var ok=true; var checksum=0.0
    for i in 0..<e {
        let exp:Float = switch opEnum {
            case 0: hA[i]+hB[i]; case 1: hA[i]-hB[i]
            case 2: hA[i]*hB[i]; default: hA[i]/hB[i] }
        if hC[i] != exp { ok=false; break }
        checksum += Double(hC[i])
    }
    let gb = Double(bytes*3) / Double(1 << 30)
    print(ok ? "Verification ✓":"Verification ✗")
    print("Checksum  = \(human(checksum))")
    print("Kernel    = \(human(ms," ms"))   (\(human(gb/(ms/1000)," GB/s")))")
}

// ── GEMV runner ───────────────────────────────────────────
func runGEMV(rows:UInt32, cols:UInt32) {
    let m=Int(rows), n=Int(cols)
    let Abytes=m*n*4, Xbytes=n*4, Ybytes=m*4
    var hA=rng(m*n), hx=rng(n), hy=[Float](repeating:0,count:m)

    preview("x[0:8]   = ", hx)
    preview("A[0,0:8] = ", hA)

    let dA=dev.makeBuffer(bytes:&hA,length:Abytes)!
    let dx=dev.makeBuffer(bytes:&hx,length:Xbytes)!
    let dy=dev.makeBuffer(length:Ybytes)!
    var r=rows,c=cols,ld=cols
    let bufs:[MTLBuffer]=[dA,dx,dy,buf(&r),buf(&c),buf(&ld)]

    let cmd=queue.makeCommandBuffer()!
    let enc=cmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(psoGEMV)
    for(i,b)in bufs.enumerated(){enc.setBuffer(b,offset:0,index:i)}
    let tpt=MTLSize(width:256,height:1,depth:1)
    let tg = MTLSize(width:m,height:1,depth:1)
    let t0=Date()
    enc.dispatchThreadgroups(tg,threadsPerThreadgroup:tpt)
    enc.endEncoding();cmd.commit();cmd.waitUntilCompleted()
    let ms=Date().timeIntervalSince(t0)*1000

    memcpy(&hy,dy.contents(),Ybytes)
    preview("y[0:8]   = ", hy)
    let gb=Double(Abytes+Xbytes+Ybytes)/Double(1<<30)
    print("Kernel    = \(human(ms," ms"))   (\(human(gb/(ms/1000)," GB/s")))")
}

// ── GEMM runner ───────────────────────────────────────────
func runGEMM(M:UInt32,N:UInt32,K:UInt32,tiled:Bool){
    let m=Int(M), n=Int(N), k=Int(K)
    let bA=m*n*4,bB=n*k*4,bC=m*k*4
    var hA=rng(m*n),hB=rng(n*k),hC=[Float](repeating:0,count:m*k)

    preview("A[0,0:8] = ", hA)
    preview("B[0,0:8] = ", hB)

    let dA=dev.makeBuffer(bytes:&hA,length:bA)!
    let dB=dev.makeBuffer(bytes:&hB,length:bB)!
    let dC=dev.makeBuffer(length:bC)!
    var M32=M,N32=N,K32=K, lda=N, ldb=K, ldc=K
    let bufs:[MTLBuffer]=[dA,dB,dC,
        buf(&M32),buf(&N32),buf(&K32),
        buf(&lda),buf(&ldb),buf(&ldc)]

    let cmd=queue.makeCommandBuffer()!
    let enc=cmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(tiled ? psoGT : psoGN)
    for(i,b)in bufs.enumerated(){enc.setBuffer(b,offset:0,index:i)}

    let tpt=tiled ? MTLSize(width:128,height:8,depth:1)
                  : MTLSize(width:16 ,height:16,depth:1)
    let tg=tiled
      ? MTLSize(width:(k+127)/128,height:(m+127)/128,depth:1)
      : MTLSize(width:(k+15)/16 ,height:(m+15)/16 ,depth:1)
    let t0=Date()
    enc.dispatchThreadgroups(tg,threadsPerThreadgroup:tpt)
    enc.endEncoding();cmd.commit();cmd.waitUntilCompleted()
    let ms=Date().timeIntervalSince(t0)*1000

    memcpy(&hC,dC.contents(),bC)
    preview("C[0,0:8] = ", hC)
    let gflops=2.0*Double(m)*Double(n)*Double(k)/(ms*1e6)
    print("Kernel    = \(human(ms," ms"))   (\(human(gflops," GFLOP/s")))")
}

// ── dispatch by op ────────────────────────────────────────
switch op {
case "add": runEW(rows:UInt32(arg[3])!,cols:UInt32(arg[4])!,opEnum:0)
case "sub": runEW(rows:UInt32(arg[3])!,cols:UInt32(arg[4])!,opEnum:1)
case "mul": runEW(rows:UInt32(arg[3])!,cols:UInt32(arg[4])!,opEnum:2)
case "div": runEW(rows:UInt32(arg[3])!,cols:UInt32(arg[4])!,opEnum:3)
case "gemv": runGEMV(rows:UInt32(arg[3])!,cols:UInt32(arg[4])!)
case "gemm-naive":
    runGEMM(M:UInt32(arg[3])!,N:UInt32(arg[4])!,K:UInt32(arg[5])!,tiled:false)
case "gemm":
    runGEMM(M:UInt32(arg[3])!,N:UInt32(arg[4])!,K:UInt32(arg[5])!,tiled:true)
default: fatalError("Unknown op")
}
