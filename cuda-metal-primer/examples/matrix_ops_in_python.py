#!/usr/bin/env python3
"""
CPU baseline: Dot → GEMV → GEMM in NumPy

* Prints the first 8 elements of every vector/matrix row it touches
* Echoes the dot product result so readers can 'see' the math
* Verifies that GEMV / GEMM agree with explicit NumPy reductions
"""
import numpy as np, textwrap as tw

np.random.seed(42)                    # reproducible demo
m, n, k = 4_096, 4_096, 4_096         # big enough to bust L3 cache

# ── data ------------------------------------------------------------
A = np.random.rand(m, n).astype(np.float32)
B = np.random.rand(n, k).astype(np.float32)
x = np.random.rand(n).astype(np.float32)

print("A[0,0:8] :", ", ".join(f"{v:6.3f}" for v in A[0, :8]))
print("B[0,0:8] :", ", ".join(f"{v:6.3f}" for v in B[0, :8]))
print("x[0:8]   :", ", ".join(f"{v:6.3f}" for v in   x[:8]))
print()

# ── dot product (first 8 just for show) -----------------------------
dot8 = np.dot(A[0, :8], B[:8, 0])
print("Dot(A[0,0:8], B[0:8,0]) =", f"{dot8:9.3f}")

# ── GEMV  y = A·x ---------------------------------------------------
y = A @ x
print("y[0:8]   :", ", ".join(f"{v:9.3f}" for v in y[:8]))

# Verify GEMV on row-0 by hand
ref0 = (A[0, :] * x).sum()
assert np.isclose(ref0, y[0]), "GEMV mismatch on row 0"

# ── GEMM  C = A·B ---------------------------------------------------
C = A @ B
print("\nC[0,0:8] :", ", ".join(f"{v:9.3f}" for v in C[0, :8]))

# Spot-check one entry with an explicit reduction
entry_check = (A[0, :] * B[:, 0]).sum()
assert np.isclose(entry_check, C[0, 0]), "GEMM spot-check failed"

print(
    tw.dedent(f"""
    Summary
      • m,n,k           : {m:,}, {n:,}, {k:,}
      • dtype           : float32
      • Dot(8-elt)      : {dot8:9.3f}
      • GEMV row-0 ok?  : ✓
      • GEMM C[0,0] ok? : ✓
    """).strip()
)