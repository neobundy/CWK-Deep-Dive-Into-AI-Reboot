// matrix_ops_cuda.cu
// -----------------------------------------------------------
// Modes:
//   add | sub | mul | div   <ROWS> <COLS>        # element-wise
//   gemv                     <ROWS> <COLS>       # y = A·x
//   gemm-naive               <M> <N> <K>         # untiled baseline
//   gemm                     <M> <N> <K>         # 16×16 tiled, shared-mem
//
// Build (Ada-class GPU):
//   nvcc -std=c++17 -O3 -arch=sm_89 matrix_ops_cuda.cu -o build/matrix_ops_cuda
//
// Example:
//   ./build/matrix_ops_cuda gemm 2048 2048 2048
// -----------------------------------------------------------

#include <cuda_runtime.h>
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

static inline std::string human(double v){
    char b[32]; std::snprintf(b,sizeof(b),"%.2f",v); return b;
}

// ───────────── element-wise kernels ───────────────────────
enum class EW {Add,Sub,Mul,Div};
template<EW Op> __device__ float fop(float a,float b);
template<> __device__ float fop<EW::Add>(float a,float b){return a+b;}
template<> __device__ float fop<EW::Sub>(float a,float b){return a-b;}
template<> __device__ float fop<EW::Mul>(float a,float b){return a*b;}
template<> __device__ float fop<EW::Div>(float a,float b){return a/b;}

template<EW Op>
__global__ void ewKernel(const float* A,const float* B,float* C,
                         int rows,int cols,int ld){
    int r = blockIdx.y*blockDim.y + threadIdx.y;
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if(r<rows && c<cols){
        int idx = r*ld + c;
        C[idx] = fop<Op>(A[idx],B[idx]);
    }
}

// ───────────── GEMV kernel ────────────────────────────────
constexpr int GEMV_TPB = 256;

__global__ void gemvKernel(const float* A,const float* x,float* y,
                           int rows,int cols,int ld){
    __shared__ float xs[GEMV_TPB];
    int row = blockIdx.x; if(row>=rows) return;

    float acc = 0.0f;
    for(int t=0;t<cols;t+=GEMV_TPB){
        int tid=threadIdx.x, col=t+tid;
        if(col<cols) xs[tid]=x[col];
        __syncthreads();
        if(col<cols) acc += A[row*ld+col]*xs[tid];
        __syncthreads();
    }
    for(int off=warpSize/2; off; off>>=1)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if(threadIdx.x==0) y[row]=acc;
}

// ───────────── GEMM kernels ───────────────────────────────
// Naïve bandwidth-bound baseline
__global__ void gemmNaive(const float* A,const float* B,float* C,
                          int M,int N,int K,int lda,int ldb,int ldc){
    int r=blockIdx.y*blockDim.y+threadIdx.y;
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(r>=M||c>=K) return;
    float acc=0.f;
    for(int n=0;n<N;++n)
        acc += A[r*lda+n]*B[n*ldb+c];
    C[r*ldc+c]=acc;
}

// 16×16-thread block, 16×16 tile, loop over K in 16-wide panels
constexpr int BLK = 16;     // threads per block dim  (16×16 = 256 threads)
constexpr int TK  = 16;     // depth of one K-panel

__global__ void gemmTiled(const float* A,const float* B,float* C,
                          int M,int N,int K,int lda,int ldb,int ldc){
    __shared__ float As[BLK][TK];
    __shared__ float Bs[TK][BLK];

    int globalRow = blockIdx.y*BLK + threadIdx.y;   // C row this thread computes
    int globalCol = blockIdx.x*BLK + threadIdx.x;   // C col this thread computes
    float acc = 0.0f;

    for(int t=0; t<N; t+=TK){
        // load A panel (row, k)  and B panel (k, col)
        int aRow = globalRow, aCol = t + threadIdx.x;
        int bRow = t + threadIdx.y, bCol = globalCol;

        As[threadIdx.y][threadIdx.x] = (aRow<M && aCol<N) ? A[aRow*lda + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow<N && bCol<K) ? B[bRow*ldb + bCol] : 0.0f;
        __syncthreads();

        #pragma unroll
        for(int k=0; k<TK; ++k)                      // dot product of 16-wide slice
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if(globalRow<M && globalCol<K) C[globalRow*ldc + globalCol] = acc;
}

// ───────────── host helpers ───────────────────────────────
void fill(std::vector<float>& v){
    std::mt19937 rng(42); std::uniform_real_distribution<float>d(-1,1);
    for(float& x: v) x = d(rng);
}
template<typename It>
void preview(const char* tag, It b){
    std::cout<<tag;
    for(int i=0;i<8;++i) std::cout<<human(*(b+i))<<(i==7?'\n':',');
}

// ───────────── main ───────────────────────────────────────
int main(int argc,char** argv){
    if(argc<2){ std::puts("Usage: matrix_ops_cuda <op> ..."); return 0; }
    std::string op = argv[1];

    // ---------- element-wise ------------------------------------------------
    if(op=="add"||op=="sub"||op=="mul"||op=="div"){
        if(argc!=4){ std::puts("args: ROWS COLS"); return 0; }
        int R=atoi(argv[2]), C=atoi(argv[3]);
        size_t elem=size_t(R)*C, bytes=elem*4;
        std::vector<float> hA(elem),hB(elem),hC(elem); fill(hA); fill(hB);

        preview("A[0,0:8] = ", hA.begin());
        preview("B[0,0:8] = ", hB.begin());

        float *dA,*dB,*dC;
        CUDA_CHECK(cudaMalloc(&dA,bytes));
        CUDA_CHECK(cudaMalloc(&dB,bytes));
        CUDA_CHECK(cudaMalloc(&dC,bytes));
        cudaMemcpy(dA,hA.data(),bytes,cudaMemcpyHostToDevice);
        cudaMemcpy(dB,hB.data(),bytes,cudaMemcpyHostToDevice);

        dim3 blk(16,16), grd((C+15)/16,(R+15)/16);
        auto t0 = std::chrono::high_resolution_clock::now();
        if(op=="add") ewKernel<EW::Add><<<grd,blk>>>(dA,dB,dC,R,C,C);
        if(op=="sub") ewKernel<EW::Sub><<<grd,blk>>>(dA,dB,dC,R,C,C);
        if(op=="mul") ewKernel<EW::Mul><<<grd,blk>>>(dA,dB,dC,R,C,C);
        if(op=="div") ewKernel<EW::Div><<<grd,blk>>>(dA,dB,dC,R,C,C);
        cudaDeviceSynchronize();
        double ms = std::chrono::duration<double,std::milli>(
                    std::chrono::high_resolution_clock::now()-t0).count();
        cudaMemcpy(hC.data(),dC,bytes,cudaMemcpyDeviceToHost);

        preview("C[0,0:8] = ", hC.begin());

        bool ok=true; double checksum=0.0;
        for(size_t i=0;i<elem;++i){
            float expect = (op=="add"?hA[i]+hB[i]:
                           op=="sub"?hA[i]-hB[i]:
                           op=="mul"?hA[i]*hB[i]:
                                     hA[i]/hB[i]);
            if(hC[i]!=expect){ ok=false; break; }
            checksum += hC[i];
        }
        std::cout<<(ok?"Verification ✓":"Verification ✗")<<"\n"
                  <<"Checksum  = "<<human(checksum)<<"\n"
                  <<"Kernel    = "<<human(ms)<<" ms   ("
                  <<human(3.0*bytes/(1ULL<<30)/(ms/1000))<<" GB/s)\n";
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    // ---------- GEMV -------------------------------------------------------
    else if(op=="gemv"){
        if(argc!=4){ std::puts("args: ROWS COLS"); return 0; }
        int R=atoi(argv[2]), C=atoi(argv[3]);
        size_t bA=size_t(R)*C*4, bX=C*4, bY=R*4;
        std::vector<float> hA(R*C), hx(C), hy(R); fill(hA); fill(hx);

        preview("x[0:8]   = ", hx.begin());
        preview("A[0,0:8] = ", hA.begin());

        float *dA,*dx,*dy;
        CUDA_CHECK(cudaMalloc(&dA,bA));
        CUDA_CHECK(cudaMalloc(&dx,bX));
        CUDA_CHECK(cudaMalloc(&dy,bY));
        cudaMemcpy(dA,hA.data(),bA,cudaMemcpyHostToDevice);
        cudaMemcpy(dx,hx.data(),bX,cudaMemcpyHostToDevice);

        dim3 blk(GEMV_TPB), grd(R);
        auto t0 = std::chrono::high_resolution_clock::now();
        gemvKernel<<<grd,blk>>>(dA,dx,dy,R,C,C);
        cudaDeviceSynchronize();
        double ms = std::chrono::duration<double,std::milli>(
                    std::chrono::high_resolution_clock::now()-t0).count();
        cudaMemcpy(hy.data(),dy,bY,cudaMemcpyDeviceToHost);

        preview("y[0:8]   = ", hy.begin());
        double gb = double(bA+bX+bY)/(1ULL<<30);
        std::cout<<"Kernel    = "<<human(ms)<<" ms   ("
                 <<human(gb/(ms/1000))<<" GB/s)\n";
        cudaFree(dA); cudaFree(dx); cudaFree(dy);
    }

    // ---------- GEMM -------------------------------------------------------
    else if(op=="gemm-naive" || op=="gemm"){
        if(argc!=5){ std::puts("args: M N K"); return 0; }
        int M=atoi(argv[2]), N=atoi(argv[3]), K=atoi(argv[4]);
        size_t bA=size_t(M)*N*4, bB=size_t(N)*K*4, bC=size_t(M)*K*4;
        std::vector<float> hA(M*N), hB(N*K), hC(M*K); fill(hA); fill(hB);

        preview("A[0,0:8] = ", hA.begin());
        preview("B[0,0:8] = ", hB.begin());

        float *dA,*dB,*dC;
        CUDA_CHECK(cudaMalloc(&dA,bA));
        CUDA_CHECK(cudaMalloc(&dB,bB));
        CUDA_CHECK(cudaMalloc(&dC,bC));
        cudaMemcpy(dA,hA.data(),bA,cudaMemcpyHostToDevice);
        cudaMemcpy(dB,hB.data(),bB,cudaMemcpyHostToDevice);

        dim3 blk,grd;
        auto t0 = std::chrono::high_resolution_clock::now();
        if(op=="gemm-naive"){
            blk = dim3(16,16);
            grd = dim3((K+15)/16,(M+15)/16);
            gemmNaive<<<grd,blk>>>(dA,dB,dC,M,N,K,N,K,K);
        }else{
            blk = dim3(BLK,BLK);      // 16×16
            grd = dim3((K+BLK-1)/BLK,(M+BLK-1)/BLK);
            gemmTiled<<<grd,blk>>>(dA,dB,dC,M,N,K,N,K,K);
        }
        cudaDeviceSynchronize();
        double ms = std::chrono::duration<double,std::milli>(
                    std::chrono::high_resolution_clock::now()-t0).count();
        cudaMemcpy(hC.data(),dC,bC,cudaMemcpyDeviceToHost);

        preview("C[0,0:8] = ", hC.begin());
        double gflops = 2.0*M*N*K / (ms*1e6);
        std::cout<<"Kernel    = "<<human(ms)<<" ms   ("
                 <<human(gflops)<<" GFLOP/s)\n";
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    else std::puts("Unknown op");
    return 0;
}
