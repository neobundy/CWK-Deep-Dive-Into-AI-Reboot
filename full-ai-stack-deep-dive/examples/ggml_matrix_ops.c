// ggml_matrix_ops.c - BLAS-style matrix operations for GGML
//
// Simplified implementation that strictly follows GGML test examples
//
// Build:
// cc -std=c11 -O3 -Iinclude \
//    extra/bench/ggml_matrix_ops.c \
//    -Lbuild/src -lggml -lggml-base -lggml-cpu -pthread \
//    -framework Metal -framework Accelerate \
//    -Wl,-rpath,./build/src \
//    -o extra/bench/ggml_matrix_ops
//
// Usage:
//   axpy <N> <alpha>            # y = alpha * x + y
//   dot  <N>                    # x·y dot product
//   gemv <ROWS> <COLS>          # y = A·x matrix-vector multiplication
//   gemm <M> <N> <K>            # C = A·B matrix-matrix multiplication

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "ggml.h"
#include "ggml-backend.h"

// Utility functions
static inline double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec*1000.0 + (double)ts.tv_nsec/1e6;
}

static inline char* human(double v) {
    static char buf[32];
    // For very small values, use more precision
    if (v < 0.01 && v > 0) {
        snprintf(buf, sizeof(buf), "%.4f", v);
    }
    // For typical values, use 3 decimal places
    else if (v < 100) {
        snprintf(buf, sizeof(buf), "%.3f", v);
    }
    // For larger values, 2 decimal places is fine
    else {
        snprintf(buf, sizeof(buf), "%.2f", v);
    }
    return buf;
}

static inline void preview(const char *tag, const float *v) {
    printf("%s", tag);
    for (int i = 0; i < 8; ++i) printf("%s%c", human(v[i]), (i==7?'\n':','));
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <mode> ...\n"
            "  axpy <N> <alpha>\n"
            "  dot  <N>\n"
            "  gemv <ROWS> <COLS>\n"
            "  gemm <M> <N> <K>\n", argv[0]);
        return 1;
    }

    const char *mode = argv[1];
    int n_threads = 8; // Use 8 threads for computation
    
    // Set thread count via environment variable for older GGML versions
    char thread_count[16];
    snprintf(thread_count, sizeof(thread_count), "%d", n_threads);
    setenv("GGML_N_THREADS", thread_count, 1);
    
    // Initialize a CPU backend via registry
    ggml_backend_reg_t cpu_reg = ggml_backend_reg_by_name("CPU");
    if (!cpu_reg) {
        fprintf(stderr, "CPU backend not found\n");
        return 1;
    }
    
    ggml_backend_dev_t cpu_dev = ggml_backend_reg_dev_get(cpu_reg, 0);
    ggml_backend_t backend = ggml_backend_dev_init(cpu_dev, NULL);
    if (!backend) {
        fprintf(stderr, "Failed to initialize CPU backend\n");
        return 1;
    }

    // --- AXPY -----------------------------------------------------------------
    if (strcmp(mode,"axpy")==0 && argc>=4) {
        if (argc < 4) { puts("args: N alpha"); return 0; }
        int    N     = atoi(argv[2]);
        float  alpha = atof(argv[3]);
        
        // Calculate memory size conservatively
        size_t bytes_per_vec = N * sizeof(float);
        size_t mem_size = 100*1024*1024 + 4*bytes_per_vec; // 100MB + 4x vector size
        
        printf("Allocating %zu bytes for AXPY operation (each vector: %zu bytes)\n", 
               mem_size, bytes_per_vec);
        
        struct ggml_init_params params = {
            .mem_size   = mem_size,
            .mem_buffer = NULL,
            .no_alloc = false,
        };

        struct ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            fprintf(stderr, "Failed to initialize GGML context\n");
            return 1;
        }

        // Create input tensors
        struct ggml_tensor * X = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
        struct ggml_tensor * Y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);

        // Fill with random data
        float *hx = (float*)X->data;
        float *hy = (float*)Y->data;
        for (int i=0; i<N; ++i) {
            hx[i] = (float)rand()/RAND_MAX * 2.0f - 1.0f;
            hy[i] = (float)rand()/RAND_MAX * 2.0f - 1.0f;
        }

        // Mark inputs as parameters
        ggml_set_param(ctx, X);
        ggml_set_param(ctx, Y);

        // Define the operation: alpha*X + Y
        struct ggml_tensor * scaled_x = ggml_scale(ctx, X, alpha);
        struct ggml_tensor * result = ggml_add(ctx, scaled_x, Y);

        // Create computation graph
        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, result);

        // Time the computation
        double t0 = now_ms();
        ggml_backend_graph_compute(backend, graph);
        double t1 = now_ms();
        double ms = t1 - t0;

        // Preview results and print performance metrics
        float* out_data = (float*)result->data;
        preview("y[0:8] = ", out_data);
        double mflops = (double)N * 2 / (ms/1000.0) / 1e6;
        char ms_buf[32], flops_buf[32];
        snprintf(ms_buf, sizeof(ms_buf), "%.3f", ms);
        snprintf(flops_buf, sizeof(flops_buf), "%.3f", mflops);
        printf("Kernel = %s ms  (%s MFLOP/s)\n", ms_buf, flops_buf);

        ggml_free(ctx);
        // Free the backend before returning
        ggml_backend_free(backend);
        return 0;
    }

    // --- DOT ------------------------------------------------------------------
    if (strcmp(mode,"dot")==0 && argc>=3) {
        if (argc < 3) { puts("args: N"); return 0; }
        int N = atoi(argv[2]);
        
        // Calculate memory conservatively
        size_t bytes_per_vec = N * sizeof(float);
        size_t mem_size = 100*1024*1024 + 4*bytes_per_vec; // 100MB + 4x vector size
        
        printf("Allocating %zu bytes for DOT operation (each vector: %zu bytes)\n", 
               mem_size, bytes_per_vec);
        
        struct ggml_init_params params = {
            .mem_size   = mem_size,
            .mem_buffer = NULL,
            .no_alloc = false,
        };

        struct ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            fprintf(stderr, "Failed to initialize GGML context\n");
            return 1;
        }

        // Create input tensors
        struct ggml_tensor * X = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
        struct ggml_tensor * Y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);

        // Fill with random data
        float *hx = (float*)X->data;
        float *hy = (float*)Y->data;
        for (int i=0; i<N; ++i) {
            hx[i] = (float)rand()/RAND_MAX * 2.0f - 1.0f;
            hy[i] = (float)rand()/RAND_MAX * 2.0f - 1.0f;
        }

        // Mark inputs as parameters
        ggml_set_param(ctx, X);
        ggml_set_param(ctx, Y);

        // Define the operation: dot product = sum(X*Y)
        struct ggml_tensor * mul = ggml_mul(ctx, X, Y);
        struct ggml_tensor * result = ggml_sum(ctx, mul);

        // Create computation graph
        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, result);

        // Time the computation
        double t0 = now_ms();
        ggml_backend_graph_compute(backend, graph);
        double t1 = now_ms();
        double ms = t1 - t0;

        // Get the result (scalar)
        float dot_result = *(float*)result->data;
        double mflops = (double)N * 2 / (ms/1000.0) / 1e6;
        char ms_buf[32], flops_buf[32];
        snprintf(ms_buf, sizeof(ms_buf), "%.3f", ms);
        snprintf(flops_buf, sizeof(flops_buf), "%.3f", mflops);
        printf("Dot = %f\nKernel = %s ms (%s MFLOP/s)\n",
               dot_result, ms_buf, flops_buf);

        ggml_free(ctx);
        // Free the backend before returning
        ggml_backend_free(backend);
        return 0;
    }

    // --- GEMV -----------------------------------------------------------------
    if (strcmp(mode,"gemv")==0 && argc>=4) {
        if (argc < 4) { puts("args: ROWS COLS"); return 0; }
        int R = atoi(argv[2]), C = atoi(argv[3]);
        
        // Calculate memory size conservatively
        size_t matrix_bytes = (size_t)R * C * sizeof(float);
        size_t vec_x_bytes = C * sizeof(float);
        size_t vec_y_bytes = R * sizeof(float);
        size_t mem_size = 200*1024*1024 + 4*(matrix_bytes + vec_x_bytes + vec_y_bytes);
        
        printf("Allocating %zu bytes for GEMV operation (matrix: %zu, x: %zu, y: %zu bytes)\n", 
               mem_size, matrix_bytes, vec_x_bytes, vec_y_bytes);
        
        struct ggml_init_params params = {
            .mem_size   = mem_size,
            .mem_buffer = NULL,
            .no_alloc = false,
        };

        struct ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            fprintf(stderr, "Failed to initialize GGML context\n");
            return 1;
        }

        // Create input tensors - note GGML's matrix layout
        struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, R); // row-major
        struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
        struct ggml_tensor * y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, R);

        // Fill with random data
        float *hA = (float*)A->data;
        float *hx = (float*)x->data;
        float *hy = (float*)y->data;
        for (int i=0; i<R*C; ++i) hA[i] = (float)rand()/RAND_MAX * 2.0f - 1.0f;
        for (int i=0; i<C; ++i)   hx[i] = (float)rand()/RAND_MAX * 2.0f - 1.0f;

        // Mark inputs as parameters
        ggml_set_param(ctx, A);
        ggml_set_param(ctx, x);
        
        // Define the operation: y = A·x
        struct ggml_tensor * result = ggml_mul_mat(ctx, A, x);

        // Create computation graph
        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, result);

        // Time the computation
        double t0 = now_ms();
        ggml_backend_graph_compute(backend, graph);
        double t1 = now_ms();
        double ms = t1 - t0;

        // Access the result
        float* out_data = (float*)result->data;
        preview("y[0:8] = ", out_data);
        double gflops = 2.0*R*C / (ms/1000.0) / 1e9;
        char ms_buf[32], flops_buf[32];
        snprintf(ms_buf, sizeof(ms_buf), "%.3f", ms);
        snprintf(flops_buf, sizeof(flops_buf), "%.3f", gflops);
        printf("Kernel = %s ms (%s GFLOP/s)\n", ms_buf, flops_buf);

        ggml_free(ctx);
        // Free the backend before returning
        ggml_backend_free(backend);
        return 0;
    }

    // --- GEMM -----------------------------------------------------------------
    if (strcmp(mode,"gemm")==0 && argc>=5) {
        if (argc < 5) { puts("args: M N K"); return 0; }
        int M = atoi(argv[2]);
        int N = atoi(argv[3]);
        int K = atoi(argv[4]);
        
        // Calculate memory requirements
        size_t A_bytes = (size_t)M * K * sizeof(float);
        size_t B_bytes = (size_t)K * N * sizeof(float);
        size_t C_bytes = (size_t)M * N * sizeof(float);
        
        // Allocate way more memory than needed for large matrices
        size_t mem_size = 1024*1024*1024 + 4*(A_bytes + B_bytes + C_bytes);
        
        printf("Running GEMM with dimensions M=%d, N=%d, K=%d\n", M, N, K);
        printf("Allocating %zu bytes for GEMM operation (A: %zu, B: %zu, C: %zu bytes)\n", 
               mem_size, A_bytes, B_bytes, C_bytes);
        
        struct ggml_init_params params = {
            .mem_size   = mem_size,
            .mem_buffer = NULL,
            .no_alloc = false,
        };

        struct ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            fprintf(stderr, "Failed to initialize GGML context\n");
            return 1;
        }

        // Create input matrices - note GGML's matrix layout
        // In GGML, ggml_mul_mat(A, B) expects:
        // A: [K, M] and B: [N, K] to produce C: [N, M]
        struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
        struct ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, K);
        
        printf("Matrix A dimensions: %d x %d\n", (int)A->ne[1], (int)A->ne[0]);
        printf("Matrix B dimensions: %d x %d\n", (int)B->ne[1], (int)B->ne[0]);

        // Fill with random data
        float *hA = (float*)A->data;
        float *hB = (float*)B->data;
        for (int i=0; i<M*K; ++i) hA[i] = (float)rand()/RAND_MAX * 2.0f - 1.0f;
        for (int i=0; i<K*N; ++i) hB[i] = (float)rand()/RAND_MAX * 2.0f - 1.0f;

        // Mark inputs as parameters (this is important)
        ggml_set_param(ctx, A);
        ggml_set_param(ctx, B);
        
        // Define the operation: C = A·B
        struct ggml_tensor * C = ggml_mul_mat(ctx, A, B);
        printf("Result matrix C dimensions: %d x %d\n", (int)C->ne[1], (int)C->ne[0]);

        // Create computation graph
        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, C);

        // Time the computation
        printf("Starting computation...\n");
        double t0 = now_ms();
        ggml_backend_graph_compute(backend, graph);
        double t1 = now_ms();
        double ms = t1 - t0;
        printf("Computation completed in %.2f ms\n", ms);

        // Access and preview results
        float* hC = (float*)C->data;
        preview("C[0,0:8] = ", hC);
        double tflops = 2.0*M*N*K / (ms/1000.0) / 1e12;
        char ms_buf[32], flops_buf[32];
        snprintf(ms_buf, sizeof(ms_buf), "%.3f", ms);
        snprintf(flops_buf, sizeof(flops_buf), "%.3f", tflops);
        printf("Kernel = %s ms  (%s TFLOP/s)\n", ms_buf, flops_buf);

        ggml_free(ctx);
        // Free the backend before returning
        ggml_backend_free(backend);
        return 0;
    }

    fprintf(stderr, "Unknown or malformed command\n");
    
    // Free the backend before returning
    ggml_backend_free(backend);
    return 1;
}