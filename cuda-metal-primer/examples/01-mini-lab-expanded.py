"""Linear-Algebra Primer: one-file demo
Requires: numpy ≥ 1.20
"""
import numpy as np
np.random.seed(42)  # reproducible randomness
import time

# 1. Dot product (Level-1 BLAS)
x = np.random.randn(1024).astype(np.float32)
w = np.random.randn(1024).astype(np.float32)
scalar = x @ w  # y = Σ x_i * w_i
print(f"Dot product → {scalar:8.2f}")

# 2. AXPY (Level-1 BLAS)  y ← αx + y
alpha = 0.1
y = np.random.randn(1024).astype(np.float32)
y = alpha * x + y
print(f"AXPY sample  → {y[:3]}")

# 3. GEMV (Level-2 BLAS)  y ← A·x + y
A = np.random.randn(512, 1024).astype(np.float32)
y2 = A @ x + y[:512]
print(f"GEMV checksum → {y2.sum():8.2f}")

# 4. GEMM (Level-3 BLAS)  C ← αA·B + βC
B = np.random.randn(1024, 256).astype(np.float32)
C = np.random.randn(512, 256).astype(np.float32)
beta = 0.5
C = alpha * (A @ B) + beta * C
print(f"GEMM slice    → {C[0, :5]}")

# 5. Memory-layout + dtype micro-bench: C- vs F-order, row vs col, FP32 vs FP16

results = []

def bench(matrix: np.ndarray, tag: str):
    import time, numpy as np
    t0 = time.perf_counter(); _ = matrix[0, :].sum(); t_row = time.perf_counter() - t0
    t0 = time.perf_counter(); _ = matrix[:, 0].sum(); t_col = time.perf_counter() - t0
    results.append((tag, t_row, t_col))
    print(f"{tag:<18}  row {t_row*1e6:6.1f} µs   col {t_col*1e6:6.1f} µs")

print("\nMemory-layout / dtype timing (lower is better)…")
A_C32 = np.ascontiguousarray(A)            # C-order FP32
A_F32 = np.asfortranarray(A)               # F-order FP32
bench(A_C32, "C-order  fp32")
bench(A_F32, "F-order  fp32")

try:
    A16 = A.astype(np.float16)
    bench(np.ascontiguousarray(A16), "C-order  fp16")
    bench(np.asfortranarray(A16),    "F-order  fp16")
except TypeError:
    pass

# Summarize best layout per access pattern
best_row = min(results, key=lambda x: x[1])
best_col = min(results, key=lambda x: x[2])
print(f"\nFastest row  access : {best_row[0]}  ({best_row[1]*1e6:5.1f} µs)")
print(f"Fastest column access : {best_col[0]}  ({best_col[2]*1e6:5.1f} µs)")

print("Hint: C-order favours row walks, F-order favours column walks.\n"  
      "     Mixed dtypes show bandwidth vs compute trade-offs.")

# -----------------------------------------------------------------------------
# 6. Scaling experiments: problem size and batch boost

print("\nScale test — dot-product latency (FP32)")
dot_times = {}
for n in (1024, 4096):
    x_s = np.random.randn(n).astype(np.float32)
    w_s = np.random.randn(n).astype(np.float32)
    t0 = time.perf_counter(); _ = x_s @ w_s; dt = (time.perf_counter() - t0)*1e6
    dot_times[n] = dt
    print(f"  n={n:<5}  {dt:6.1f} µs")

ratio = dot_times[4096] / dot_times[1024]
print(f"Observed scaling factor ≈ {ratio:.2f}  (ideal linear = 4.0)")

print("\nBatch-boost GEMM (αA·B + βC) timing")
configs = [
    (512, 1024, 256),   # baseline
    (2048, 4096, 256),  # boosted
]
throughputs = {}
for m,k,n in configs:
    A_big = np.random.randn(m, k).astype(np.float32)
    B_big = np.random.randn(k, n).astype(np.float32)
    C_big = np.random.randn(m, n).astype(np.float32)
    t0 = time.perf_counter(); _ = alpha * (A_big @ B_big) + beta * C_big; dt = (time.perf_counter() - t0)
    gflops = (2*m*k*n)/dt/1e9
    throughputs[(m,k,n)] = gflops
    print(f"  {m}x{k} @ {k}x{n}  → {dt:5.3f} s  ({gflops:5.1f} GFLOP/s)")

speedup = throughputs[(2048,4096,256)] / throughputs[(512,1024,256)]
print(f"\nGEMM throughput speed-up (big vs baseline) ≈ {speedup:4.1f}×")
print("If this boost is < theoretical (×8 here), you're already bandwidth-bound; if ~linear, you're compute-bound.")