"""Linear-Algebra Primer: one-file demo
Requires: numpy ≥ 1.20
"""
import numpy as np
np.random.seed(42)  # reproducible randomness

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

# 5. Memory-layout gotcha: row- vs column-major timing sneak peek
row_major = np.ascontiguousarray(A)           # C-order
col_major = np.asfortranarray(A)              # F-order

# Time a stride-1 access versus a strided column pull
import time
start = time.perf_counter(); _ = row_major[0, :].sum(); t_row = time.perf_counter() - start
start = time.perf_counter(); _ = col_major[:, 0].sum(); t_col = time.perf_counter() - start
print(f"Row-major contiguous read  : {t_row*1e6:5.1f} µs")
print(f"Column-major strided read   : {t_col*1e6:5.1f} µs  ← slower on C-order array")