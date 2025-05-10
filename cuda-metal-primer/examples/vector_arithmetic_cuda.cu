// vector_arithmetic_cuda.cu
// -----------------------------------------------------------
// Expanded CUDA demo: element-wise ADD / SUB / MUL / DIV on two
// vectors built from an integer range. Lets curious readers see
// how a single kernel can branch on the operator without four
// duplicate files.
//
//  Build:
//      mkdir -p build
//      nvcc -std=c++17 -arch=sm_89 vector_arithmetic_cuda.cu -o build/vector_arithmetic_cuda
//
//  Run examples:
//      ./build/vector_arithmetic_cuda add 1 10     #  A[i]=1..10,  B[i]=10..1
//      ./build/vector_arithmetic_cuda sub 1 10
//      ./build/vector_arithmetic_cuda mul 1 10
//      ./build/vector_arithmetic_cuda div 1 10
//
//  Notes:
//      • B is generated in descending order to avoid trivial 0 / 1 results.
//      • Division safeguards against zero denominators.
//      • Prints first 8 elements plus checksum so learners can eyeball.
// -----------------------------------------------------------

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cinttypes>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>

enum class Op : int { Add=0, Sub=1, Mul=2, Div=3 };

__global__ void vectorArithmetic(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float*       __restrict__ C,
                                 size_t N, Op op)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    switch (op) {
        case Op::Add: C[idx] = A[idx] + B[idx]; break;
        case Op::Sub: C[idx] = A[idx] - B[idx]; break;
        case Op::Mul: C[idx] = A[idx] * B[idx]; break;
        case Op::Div: C[idx] = A[idx] / B[idx]; break; // B guaranteed non-zero
    }
}

static void usage(const char* prog) {
    std::cout << "Usage: " << prog << " <add|sub|mul|div> <START> <END>\n";
}

static Op parseOp(const std::string& s) {
    if (s=="add") return Op::Add;
    if (s=="sub") return Op::Sub;
    if (s=="mul") return Op::Mul;
    if (s=="div") return Op::Div;
    throw std::invalid_argument("Unknown op");
}

int main(int argc, char** argv)
{
    if (argc != 4) { usage(argv[0]); return 0; }

    Op op;
    try { op = parseOp(argv[1]); }
    catch(...) { usage(argv[0]); return 1; }

    long long start = std::atoll(argv[2]);
    long long end   = std::atoll(argv[3]);
    if (end < start) { std::cerr << "END must be ≥ START\n"; return 1; }

    size_t N     = static_cast<size_t>(end - start + 1);
    size_t bytes = N * sizeof(float);
    std::cout << "Vector length: " << N << " (\"" << argv[1] << "\" op)\n";

    // (1) Allocate host memory — A
    float *h_A=(float*)malloc(bytes), *h_B=(float*)malloc(bytes), *h_C=(float*)malloc(bytes);
    if(!h_A||!h_B||!h_C){ std::cerr<<"Host malloc failed\n"; return 1; }

    // (2) Initialize vectors — I
    // Fill A ascending, B descending (avoid div-by-zero)
    for(size_t i=0;i<N;++i){
        h_A[i] = static_cast<float>(start + i);
        h_B[i] = static_cast<float>(end   - i);
        if(op==Op::Div && h_B[i]==0.0f) h_B[i]=1.0f; // safeguard
    }

    // (3) Allocate device memory — O (first half)
    float *d_A=nullptr,*d_B=nullptr,*d_C=nullptr;
    cudaMalloc(&d_A,bytes); cudaMalloc(&d_B,bytes); cudaMalloc(&d_C,bytes);

    // (4) Copy Host → Device — O (second half)
    cudaMemcpy(d_A,h_A,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,bytes,cudaMemcpyHostToDevice);

    // (6) Launch the kernel — L
    const int TPB=256; int blocks=(N+TPB-1)/TPB;
    auto t0=std::chrono::high_resolution_clock::now();
    vectorArithmetic<<<blocks,TPB>>>(d_A,d_B,d_C,N,op);
    cudaDeviceSynchronize();
    auto t1=std::chrono::high_resolution_clock::now();

    // (7) Copy Device → Host — L (device→Local)
    cudaMemcpy(h_C,d_C,bytes,cudaMemcpyDeviceToHost);

    // (8) Validate results — O (Observe / Verify)
    bool ok=true; double checksum=0.0;
    for(size_t i=0;i<N;++i){
        float expected;
        switch(op){
            case Op::Add: expected=h_A[i]+h_B[i]; break;
            case Op::Sub: expected=h_A[i]-h_B[i]; break;
            case Op::Mul: expected=h_A[i]*h_B[i]; break;
            case Op::Div: expected=h_A[i]/h_B[i]; break;
        }
        if(h_C[i]!=expected){ ok=false; break; }
        checksum+=h_C[i];
    }

    // Show first 8 elements for human glance
    std::cout << "A[:8]   = "; for(int i=0;i<8&&i<N;++i) std::cout<<h_A[i]<<" "; std::cout<<"...\n";
    std::cout << "B[:8]   = "; for(int i=0;i<8&&i<N;++i) std::cout<<h_B[i]<<" "; std::cout<<"...\n";
    std::cout << "C[:8]   = "; for(int i=0;i<8&&i<N;++i) std::cout<<h_C[i]<<" "; std::cout<<"...\n";
    std::cout << std::fixed << std::setprecision(0);
    std::cout << "Checksum = " << checksum << "\n";
    std::cout.unsetf(std::ios::fixed);

    // (9) Garbage-collect both sides — G
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    // Timing
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();

    std::cout << (ok?"Verification ✓":"Verification ✗") << "  |  " << std::fixed << std::setprecision(2)
              << ms << " ms\n";

    return ok?0:1;
}
