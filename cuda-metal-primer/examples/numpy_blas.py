#!/usr/bin/env python3
"""Compare hand-rolled loops with NumPy (OpenBLAS/MKL) GEMM."""
import numpy as np, time
m = n = k = 1024
A = np.random.rand(m, k).astype(np.float32)
B = np.random.rand(k, n).astype(np.float32)
C = np.zeros((m, n), dtype=np.float32)

print(f"[WARN] Running naive triple-loop GEMM for {m}×{k}×{n}. This may take several minutes. Press Ctrl+C to skip.", flush=True)

ts0 = time.time();
try:
    for i in range(m):
        for j in range(n):
            for l in range(k):
                C[i, j] += A[i, l] * B[l, j]
    print("naïve ms", (time.time() - ts0) * 1e3)
except KeyboardInterrupt:
    print("\n[SKIPPED] Naive loop interrupted by user.")

ts1 = time.time();
C2 = A @ B
print("BLAS ms", (time.time() - ts1) * 1e3)
if 'C' in locals() and C is not None:
    assert np.allclose(C, C2)