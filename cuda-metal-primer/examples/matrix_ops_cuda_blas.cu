// matrix_ops_cuda_blas.cu
// -----------------------------------------------------------
// Modes (BLAS):
//   axpy <N> <alpha>            # y = alpha * x + y           (BLAS-1)
//   dot  <N>                    # x·y                        (BLAS-1)
//   gemv <ROWS> <COLS>          # y = A·x                    (BLAS-2)
//   gemm <M> <N> <K>            # C = A·B                    (BLAS-3)
//
// Build (Ada-class GPU):
//   nvcc -std=c++17 -O3 -arch=sm_89 matrix_ops_cuda_blas.cu -lcublas -o build/matrix_ops_cuda_blas
// -----------------------------------------------------------

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <numeric>

#define CUDA_CHECK(x) { cudaError_t e=(x); if(e!=cudaSuccess){              \
    printf("CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
    exit(1);} }
#define CUBLAS_CHECK(x) { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
    printf("cuBLAS err %s:%d status=%d\n",__FILE__,__LINE__,s); exit(1);} }

static inline std::string human(double v){ char b[32]; std::snprintf(b,sizeof(b),"%.2f",v); return b; }

template<typename It> void preview(const char* tag, It b){
    std::cout<<tag; for(int i=0;i<8;++i) std::cout<<human(*(b+i))<<(i==7?'\n':',');
}

int main(int argc,char** argv){
    if(argc<2){
        std::puts("Usage: matrix_ops_cuda_blas <mode> ...\n"
                  "  axpy N alpha\n  dot N\n  gemv ROWS COLS\n  gemm M N K");
        return 0;
    }
    std::string mode = argv[1];
    cublasHandle_t handle; CUBLAS_CHECK(cublasCreate(&handle));

    std::mt19937 rng(42); std::uniform_real_distribution<float>d(-1,1);

    if(mode=="axpy"){
        if(argc!=4){ std::puts("args: N alpha"); return 0; }
        int N = std::atoi(argv[2]); float alpha = std::atof(argv[3]);
        size_t bytes = N* sizeof(float);
        std::vector<float> hx(N), hy(N);
        for(float &x: hx) x=d(rng);
        for(float &y: hy) y=d(rng);
        float *dx,*dy; CUDA_CHECK(cudaMalloc(&dx,bytes)); CUDA_CHECK(cudaMalloc(&dy,bytes));
        cudaMemcpy(dx,hx.data(),bytes,cudaMemcpyHostToDevice);
        cudaMemcpy(dy,hy.data(),bytes,cudaMemcpyHostToDevice);

        auto t0=std::chrono::high_resolution_clock::now();
        CUBLAS_CHECK(cublasSaxpy(handle,N,&alpha,dx,1,dy,1));
        CUDA_CHECK(cudaDeviceSynchronize());
        double ms=std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-t0).count();
        cudaMemcpy(hy.data(),dy,bytes,cudaMemcpyDeviceToHost);
        preview("y[0:8] = ", hy.begin());
        std::cout<<"Kernel = "<<human(ms)<<" ms  ("<<human( double(N)*2/1e6/(ms/1000) )<<" MFLOP/s)\n"; // 2 FLOP per axpy element
        cudaFree(dx); cudaFree(dy);
    }
    else if(mode=="dot"){
        if(argc!=3){ std::puts("args: N"); return 0; }
        int N=std::atoi(argv[2]); size_t bytes=N*sizeof(float);
        std::vector<float> hx(N), hy(N); for(float &x:hx)x=d(rng); for(float &y:hy)y=d(rng);
        float *dx,*dy; CUDA_CHECK(cudaMalloc(&dx,bytes)); CUDA_CHECK(cudaMalloc(&dy,bytes));
        cudaMemcpy(dx,hx.data(),bytes,cudaMemcpyHostToDevice);
        cudaMemcpy(dy,hy.data(),bytes,cudaMemcpyHostToDevice);
        float result=0.f;
        auto t0=std::chrono::high_resolution_clock::now();
        CUBLAS_CHECK(cublasSdot(handle,N,dx,1,dy,1,&result));
        CUDA_CHECK(cudaDeviceSynchronize());
        double ms=std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-t0).count();
        std::cout<<"Dot = "<<result<<"\nKernel = "<<human(ms)<<" ms ("<<human(double(N)*2/1e6/(ms/1000))<<" MFLOP/s)\n";
        cudaFree(dx); cudaFree(dy);
    }
    else if(mode=="gemv"){
        if(argc!=4){ std::puts("args: ROWS COLS"); return 0; }
        int R=std::atoi(argv[2]), C=std::atoi(argv[3]);
        size_t bA=size_t(R)*C*4, bX=C*4, bY=R*4;
        std::vector<float> hA(R*C), hx(C), hy(R);
        for(float &v:hA) v=d(rng); for(float &v:hx) v=d(rng);
        float *dA,*dx,*dy; CUDA_CHECK(cudaMalloc(&dA,bA)); CUDA_CHECK(cudaMalloc(&dx,bX)); CUDA_CHECK(cudaMalloc(&dy,bY));
        cudaMemcpy(dA,hA.data(),bA,cudaMemcpyHostToDevice); cudaMemcpy(dx,hx.data(),bX,cudaMemcpyHostToDevice);
        const float alpha=1.f,beta=0.f;
        auto t0=std::chrono::high_resolution_clock::now();
        CUBLAS_CHECK(cublasSgemv(handle,CUBLAS_OP_N,R,C,&alpha,dA,R,dx,1,&beta,dy,1)); // row-major? We treat row-major via lda=R and opN
        CUDA_CHECK(cudaDeviceSynchronize());
        double ms=std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-t0).count();
        cudaMemcpy(hy.data(),dy,bY,cudaMemcpyDeviceToHost);
        preview("y[0:8] = ", hy.begin());
        double flops = 2.0*R*C;
        std::cout<<"Kernel = "<<human(ms)<<" ms ("<<human(flops/1e9/(ms/1000))<<" GFLOP/s)\n";
        cudaFree(dA); cudaFree(dx); cudaFree(dy);
    }
    else if(mode=="gemm"){
        if(argc!=5){ std::puts("args: M N K"); return 0; }
        int M=std::atoi(argv[2]), N=std::atoi(argv[3]), K=std::atoi(argv[4]);
        size_t bA=size_t(M)*N*4, bB=size_t(N)*K*4, bC=size_t(M)*K*4;
        std::vector<float> hA(M*N), hB(N*K), hC(M*K);
        for(float &v:hA) v=d(rng); for(float &v:hB) v=d(rng);
        float *dA,*dB,*dC; CUDA_CHECK(cudaMalloc(&dA,bA)); CUDA_CHECK(cudaMalloc(&dB,bB)); CUDA_CHECK(cudaMalloc(&dC,bC));
        cudaMemcpy(dA,hA.data(),bA,cudaMemcpyHostToDevice); cudaMemcpy(dB,hB.data(),bB,cudaMemcpyHostToDevice);
        const float alpha=1.f,beta=0.f;
        auto t0=std::chrono::high_resolution_clock::now();
        // cuBLAS expects column-major; easiest trick is opT both and swap dims so we keep row-major in memory.
        CUBLAS_CHECK(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,K,M,N,&alpha,dB,K,dA,N,&beta,dC,K));
        CUDA_CHECK(cudaDeviceSynchronize());
        double ms=std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-t0).count();
        cudaMemcpy(hC.data(),dC,bC,cudaMemcpyDeviceToHost);
        preview("C[0,0:8] = ", hC.begin());
        double flops=2.0*M*N*K;
        std::cout<<"Kernel = "<<human(ms)<<" ms ("<<human(flops/1e12/(ms/1000))<<" TFLOP/s)\n";
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
    else {
        std::puts("Unknown mode");
    }

    cublasDestroy(handle);
    return 0;
}