# Chapter 1 · Linear Algebra for AI in a Nutshell — A Lightning Refresher

> *"If you can trace every tensor op back to dot-products and matrix multiplies, GPU logs stop looking like voodoo and start looking like arithmetic."*

*(Personal lab notebook - Last verified 2025‑05‑03)*

---

## 1 · Why Every AI Stack Collapses to Two Primitives

| Primitive                  | Core Math     | Where You See It in Practice                                                 |
| -------------------------- | ------------- | ---------------------------------------------------------------------------- |
| **Dot Product**            | `y = Σ xᵢ·wᵢ` | • Attention scores in transformers<br>• Cosine-similarity in vector search   |
| **Matrix Multiply (GEMM)** | `C = A × B`   | • Dense layers, convolutions after im2col<br>• Q-K-V projections, MLP blocks |

Everything else—convolutions, softmax, even graph neural networks—gets lowered to these two when the compiler schedules work for a GPU. Vendors therefore pour decades of effort into BLAS libraries and hardware tensor units.  

*Heads-up:* a dedicated chapter later in the series breaks down BLAS in gory detail, so no need to memorize acronyms yet.

When we strip away the complexity, most GPU kernels in AI workloads ultimately perform variations of the same fundamental operation: matrix multiplication with optional addition (A × B + C). This elegant simplicity is why GPUs excel at deep learning—they're architecturally optimized for this exact pattern.

When we expand these operations to work with vectors and matrices rather than simple scalars, we get the three fundamental operations that form the backbone of all AI computation:

1. **AXPY**: Vector scaling and addition (`y = αx + y`) - "A" for alpha, "X" and "Y" for vectors
2. **GEMV**: Matrix-vector multiplication (`y = Ax + y`) - "GE" for general matrix, "MV" for matrix-vector
3. **GEMM**: Matrix-matrix multiplication (`C = αAB + βC`) - "GE" for general matrix, "MM" for matrix-matrix

Whatever your math comfort zone, mastering these primitives is non-negotiable—they underpin everything from attention scores to convolution filters.

If you're looking for a deeper foundation, I recommend exploring my companion repository:

[Deep Dive into AI with MLX and PyTorch](https://github.com/neobundy/Deep-Dive-Into-AI-With-MLX-PyTorch)

This primer assumes basic familiarity with these concepts, but we'll keep explanations practical and focused on GPU implementation. Understanding these fundamentals isn't about mathematical purity—it's about gaining the intuition needed to optimize real-world AI workloads and interpret profiler outputs correctly.

We assume you're comfortable reading Python snippets; the focus here is GPU concepts, not syntax trivia.  A dash of C/C++ literacy helps, too—performance-critical kernels still gravitate to those languages, and many code examples you'll meet in the wild tour through their headers.  Even a reader-level grasp will pay dividends.

Now, let's dive into how these operations map to GPU execution patterns.

---

## 2 · Mental Pictures, Not Formulas

* **Vector (1-D)** → an arrow with length = norm, direction = feature mix.
  * Think of a 1024-element vector as a single arrow in 1024-dimensional space
  * The magnitude (norm) tells you "how much" of something you have
  * The direction encodes the specific mix of features (what makes this data point unique)

* **Matrix (2-D)** → a grid of such arrows stacked as rows or columns.
  * Row-major view: each row is a separate vector (e.g., batch of embeddings)
  * Column-major view: each column is a separate vector (e.g., features across samples)
  * The choice of layout dramatically affects memory access patterns on GPUs

* **Tensor (3-D+)** → "matrix per time-step / batch / head"; still boils down to batches of matrices during compute.
  * 3D tensor: stack of matrices (e.g., sequence of image frames, batch of images)
  * 4D tensor: batch of volumes (e.g., batch of RGB images)
  * Higher dimensions: attention heads, channels, etc.
  * Despite the complexity, GPUs process these as batched 2D operations

Keep those analogies handy; they map one-to-one onto GPU launch geometry.

```python
# Python visualization of linear algebra primitives

# 1. Vectors (1-D): arrows with magnitude and direction
import numpy as np

# Create a 1024-dimensional vector (think: a single arrow in 1024-D space)
embedding = np.random.randn(1024)  

# Vector magnitude (norm) - "how much" signal strength
magnitude = np.linalg.norm(embedding)  # L2 norm (Euclidean length)

# Direction - normalize to isolate the feature mix (unit vector)
direction = embedding / magnitude  # Now |direction| = 1

# 2. Matrices (2-D): grids of vectors
# Row-major view: each row is a data point (e.g., batch of 32 embeddings)
embeddings_batch = np.random.randn(32, 1024)  # 32 vectors stacked as rows

# Column-major view: each column is a feature across samples
features_matrix = np.random.randn(1024, 32)  # Features arranged in columns

# Memory layout difference (matters greatly for GPU performance)
row_major_access = embeddings_batch[5, :]  # Contiguous memory access
col_major_access = features_matrix[:, 5]   # Strided memory access


# 3. Tensors (3-D+): matrices extended to higher dimensions
# 3D tensor: stack of matrices (batch of sequence data)
sequence_data = np.random.randn(16, 512, 1024)  # 16 sequences, 512 tokens, 1024 dims

# 4D tensor: batch of volumes (e.g., batch of images)
image_batch = np.random.randn(8, 3, 224, 224)  # 8 images, RGB channels, 224×224 pixels

# Multi-headed attention: even higher dimensions 
attention_tensor = np.random.randn(32, 16, 512, 64)  # batch, heads, sequence, dims
```

### 2.1 · Simple Insight: Cosine Similarity — The Workhorse of Vector Comparisons

You'll bump into cosine similarity everywhere in AI—it powers transformer attention, semantic search, recommendation engines, you name it.  

Think of vectors as arrows in space: if two arrows lean in almost the same direction, the underlying ideas rhyme (cosine ≈ 1).  

Point them opposite ways and you've captured opposing concepts (cosine ≈ −1).  

Swing them 90° apart and they're talking past each other (cosine ≈ 0).

Cosine similarity formalizes that picture while ignoring raw magnitude:

* **Formula** cos(θ) = (A·B) / (‖A‖ · ‖B‖)
* **Range** [-1, 1]
* **Shows up in**
  * Semantic search — finding neighbor documents
  * Recommender systems — matching user tastes
  * Clustering — grouping kindred vectors
  * Attention mechanisms — scoring token relevance

In code it collapses to one dot-product plus two norms:

```python
cosine_sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
```

Why does this angular trick matter? Because **direction—not magnitude—is where embeddings stash their meaning**. That's why those meme algebra lines make sense:

- king − man ≈ queen − woman  
- king + queen ≈ husband + wife  
- king + princess ≈ father + daughter

Similar vectors share a heading; add or subtract them and the arrows slide to new, equally meaningful coordinates.

Whenever you tackle a new idea, chase the **simplest mental model** first: inherit it, then extend via polymorphism. Treat the gritty math like private methods in an OO class—encapsulate until you *need* to **peek** inside. Saves brain cycles—and as the years pile on, those cycles become premium silicon. Capiche?

---

## 3 · The Only Three Operations You Must Recognize in Code

| Category | BLAS Level | Canonical Signature | GPU Kernel Nick-Names | Typical Use Case |
| -------- | ---------- | ------------------- | --------------------- | ---------------- |
| **AXPY** | Level-1    | `y ← a·x + y`       | *SAXPY, VectorAdd*    | Bias adds, residual skips |
| **GEMV** | Level-2    | `y ← A·x + y`       | *MatVec, DenseInference* | Single-token inference, embedding lookup |
| **GEMM** | Level-3    | `C ← α·A·B + β·C`   | *matmul, mma, wmma*   | Self-attention, batched inference |

All high-level ML frameworks eventually dispatch one of these. Here's why each matters:

### AXPY (Level-1): Vector-Vector Ops
- **Perf profile**: Memory-bound (tiny arithmetic intensity)
- **Parallelism**: One GPU thread per vector element
- **Bottleneck**: DRAM bandwidth (≈ 819 GB/s on modern M-series parts)
- **Python sketch**:
  ```python
  # y = alpha * x + y
  def axpy(alpha, x, y):
      return alpha * x + y
  ```

### GEMV (Level-2): Matrix-Vector Ops
- **Perf profile**: Still memory-bound, but with modest data reuse
- **Parallelism**: Often one thread (or warp) per matrix row
- **Bottleneck**: Mostly DRAM, with a dash of compute
- **Python sketch**:
  ```python
  # y = A @ x + y
  def gemv(A, x, y):
      return A @ x + y
  ```

### GEMM (Level-3): Matrix-Matrix Ops
- **Perf profile**: Compute-bound (high arithmetic intensity)
- **Parallelism**: Tiled algorithm leveraging shared memory / tensor cores
  * **Tiling** breaks big matrices into on-chip-sized blocks.  
  * A warp-sized thread group loads one block, **reuses** it repeatedly, then writes results back.  
  * This slashes global-memory traffic and keeps data hot in low-latency SRAM, driving up the compute-to-memory ratio.
- **Bottleneck**: Raw FLOPs (well-tuned kernels hit 80 %+ of peak)
- **Python sketch**:
  ```python
  # C = alpha * (A @ B) + beta * C
  def gemm(alpha, A, B, beta, C):
      return alpha * (A @ B) + beta * C
  ```

When you crack open a profiler, spotting which of these three primitives dominates your timeline instantly tells you whether to chase **bandwidth** or **compute** optimizations.

> **Sidebar · Memory- vs Compute-Bound: Blink Test**  
> * **Low FLOPs per byte (< 4)** → You're slamming DRAM. Think AXPY, GEMV.  
>   * Fixes: fuse kernels, switch to half-precision, improve locality.  
> * **High FLOPs per byte (> 8)** → You're taxing ALUs/tensor cores. Think tiled GEMM.  
>   * Fixes: crank up clock, use mixed-precision, ensure you hit fast math paths.  
> * **In-between (4-8)** → A tweener; small tiling or poor occupancy might be splitting the bill.  
> * Keep this cheat sheet next to your profiler—saves hours of wild-goose tuning.
> * **One-liner heuristic**: Intensity = (total FLOPs) / (bytes touched). If the number is smaller than your GPU's *FLOPs per DRAM byte*, you're memory-bound; otherwise compute-bound.  

---

## 4 · Memory Layout & Performance in One Paragraph

*On CPUs you can mostly ignore whether data is row- or column-major; on GPUs you absolutely cannot.*

Coalesced loads/stores demand you **pick one convention and stick with it**. NVIDIA cuBLAS defaults to **column-major** (Fortran style) while Apple's Accelerate assumes **row-major** (C style). 

**Row-major (C order)** Elements across a row sit next to each other in memory—addresses march M[i][j]  →  M[i][j + 1]. Default for C/C++, Python/NumPy, and Metal's Accelerate.

**Column-major (Fortran order)** Elements down a column sit next to each other—addresses march M[i][j]  →  M[i + 1][j]. Default for Fortran, MATLAB, R, and NVIDIA's cuBLAS.

The word *major* tells you which index varies fastest: the column index (j) in row-major, the row index (i) in column-major. Get that wrong and your warp jumps through memory instead of gliding, tanking bandwidth.

You don't need to chase every compiler quirk; keep the mental model straight and consult the docs when you step onto new silicon.  The low-level plumbing evolves fast, but the contiguous-vs-strided principle is timeless.

---

### 4.1 · Why Memory Layout Matters on GPUs

On both GPU families, a warp/SIMD-group is happiest when every thread grabs the next 4- or 8-byte element from contiguous addresses:

```
Thread 0 → A[0]  A[1]  A[2]  A[3]   (coalesced)
Thread 1 → A[4]  A[5]  A[6]  A[7]   (coalesced)
...
```

Flip the layout and you get:

```
Thread 0 → A[0]  A[16] A[32] …      (stride = leadingDim)
Thread 1 → A[1]  A[17] A[33] …
```

Each lane now traverses memory at a stride equal to the leading dimension. The result: more cache lines touched, higher latency, and lower sustained bandwidth, which in turn limits how fully the compute units can be utilised.

A warp (32 threads on NVIDIA, 32 / 64 on Apple) reaches peak bandwidth only when its threads touch **contiguous addresses**—that's **coalesced access**:

| Layout | In-memory Order | Thread Access Pattern | Best For |
|--------|-----------------|-----------------------|----------|
| **Row-major** | Elements across a row are adjacent | Threads stride across a row | Row-wise ops (e.g., batch GEMM in C order) |
| **Column-major** | Elements down a column are adjacent | Threads stride down a column | Column-wise ops (e.g., Fortran/BLAS kernels) |

When layouts don't match the kernel's expectation, the GPU must:
1. Issue scattered loads (≈ 10× slower than coalesced)
2. Perform an implicit transpose (extra ALU & latency)
3. Burn shared-memory bandwidth that could have fed math units

### 4.2 · Real-World Pain Points

In transformer models:
- **Weights** often ship in one layout, **activations** in another.
- Every mismatch cuts throughput dramatically—sometimes by half.

**Quick fix** Use the library's transpose flags (`CUBLAS_OP_T`, `CblasTrans`) instead of materializing transposed copies. The GEMM kernel will reorder access internally while your tensors stay put in memory.

---

## 5 · Roofline Blink Test

A three-step sanity check *before* you lose a weekend to exotic optimizations:

* **Peak Compute**  
  * M3 Ultra (80-core GPU, projected): ≈ 32 TF32 TFLOPs † 
    * This means the M3 Ultra can theoretically perform 32 trillion floating-point operations per second using TF32(Tensor Float 32) precision - that's 32,000,000,000,000 math operations every second!
  * RTX 4090: ≈ 82 TF32 TFLOPs (shader cores)
    * The RTX 4090 can theoretically crunch through 82 trillion operations per second - over 2.5× more raw compute power than the M3 Ultra.
* **Peak DRAM Bandwidth**  
  * M3 Ultra: 819 GB/s  
    * This GPU can move 819 gigabytes every second—fast enough to fling a full, uncompressed English Wikipedia dump (~250 GB) across the bus in well under half a second.
  * RTX 4090: 1008 GB/s
    * The RTX 4090 can transfer just over 1 terabyte of data per second, about 23% more bandwidth than the M3 Ultra.

  
* **Operational Intensity** (= FLOPs ÷ bytes moved)

| Kernel | Rough Intensity | Bound |
|--------|-----------------|-------|
| Dot product | ≈ 2 / 8 = **0.25** | Bandwidth-bound |
| AXPY | 2 / 12 = **0.17** | *Severely* bandwidth-bound |
| Matrix-vector | ≈ 2N / 8N = **0.25** | Bandwidth-bound |
| Tiled GEMM | > 16 / 8 = **2+** | Compute-bound |

**Roofline rule of thumb**  
Achievable FLOPs ≤ min( Peak_FLOPs, Peak_BW × Intensity ).
If the throughput you measure is < 10 % of that minimum, your kernel is mis-aligned—not under-fused.

> **Reality Check:** Peak specs are marketing billboards—hand-picked, lab-perfect conditions. In the wild you'll net maybe 30-70 % thanks to cache misses, instruction stalls, and kernel overhead. The goal is a *mental* roofline to spot bottlenecks, not a fantasy lap time. Think supercar rated for 300 km/h: handy for bragging rights, irrelevant to daily driving.

> **Pro tip:** If you need a spreadsheet to judge a kernel, you're over-thinking it. Quick head math plus the roofline gets you 95 % of the insight.

---

## 6 · Where We Go from Here

From here we'll zoom in on the basics and build upward—starting with single-arrow vectors, scaling to matrices, and finally peeking into GPU tricks for high-dimensional tensors. The exact chapter order may shuffle as the series matures, but the trajectory stays the same: translate everyday ML kernels into mental pictures you can reason about and optimize. By the end, a profiler line will read like plain English.

Run the mini-labs—they turn theory into muscle memory and save you hours of future head-scratching.

---

## Appendix A · Mini Lab

The snippet below walks through every building block we just covered. Copy–paste it into a Jupyter cell—or run it straight from `python`—and watch the numbers fall out.

[01-mini-lab.py](examples/01-mini-lab.py)

```python
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
```

Try these quick tweaks after you get a baseline:

* **Scale problem size** – change `1024` to `4096` for `x` and `w`; watch dot-product time grow linearly.
* **Batch boost** – bump `A` to `2048 × 4096` and see GEMM throughput dominate.
* **Layout flip** – create `A_f = np.asfortranarray(A)` and rerun the timing block; note how the strided versus contiguous read latency swaps.
* **Precision drop** – cast everything to `float16` (if your GPU supports it) and rerun to feel the bandwidth relief.

Run, tweak, measure—each change maps straight back to a concept from the chapter.

Example output(M3 Ultra with 512GB RAM):

```bash
% python 01-mini-lab.py
Dot product →    41.01
AXPY sample  → [-0.2782233   0.14136425  0.8898671 ]
GEMV checksum →   176.07
GEMM slice    → [-3.4731958  -6.2966533  -0.85809225 -3.1875205  -1.094734  ]
Row-major contiguous read  :  22.5 µs
Column-major strided read   :   2.4 µs  ← slower on C-order array
```

Here's how to read those numbers when they come off an M3 Ultra (819 GB/s DRAM BW, 512 GB unified memory).

---

1. Dot product → 41.01  
   * Nothing magic—just the scalar result of Σ xᵢ · wᵢ on two random 1 024-element vectors.  
   * It confirms the code ran; the value itself isn't a performance metric.

2. AXPY sample → [-0.27  0.14  0.89]  
   * Shows the first three entries of y ← αx + y.  
   * Verifies broadcasting and in-place update work.

3. GEMV checksum → 176.07  
   * Sum of the 512-element output vector y₂ = A · x + y.  
   * If you rerun with a different seed, this number changes, but a checksum is handy for "did I break anything?" regression tests.

4. GEMM slice → [-3.47 -6.30 -0.86 -3.19 -1.09]  
   * First five numbers of C = αA · B + βC.  
   * Confirms the Level-3 BLAS path executed without NaNs/Infs.

5. Memory-layout micro-benchmark   

```bash
Row-major contiguous read :  22.5 µs
Column-major strided read :   2.4 µs  ← slower on C-order array
```

What you're seeing:

| Array                  | Access pattern                              | Reality on M3 Ultra |
|------------------------|--------------------------------------------|--------------------|
| `row_major` (C-order)  | `row_major[0, :]` — contiguous in memory   | ~22 µs             |
| `col_major` (Fortran)  | `col_major[:, 0]` — contiguous in memory   | ~2 µs              |

Both accesses are contiguous for *their* respective layout; the faster one is simply the dimension that enjoys unit-stride access. In this particular run, the column-major path came out ~4 × faster. Change the layout or bump the matrix size and the winner flips.  
**Rule of thumb:** whichever dimension is contiguous in memory wins—no Apple-vs-Fortran magic, just stride physics.

Take-aways
-----------

1. Memory layout dominates small-kernel timings. A 512 × 1024 matrix isn't big enough to saturate 819 GB/s, yet you still see a multi-x swing (often 3-4×) just by flipping the order flag.  
2. That swing will be even larger on GPU kernels where coalesced loads are a hard requirement.  
3. The rest of the output lines are sanity checks; only the last two numbers are the mini performance demo. Change `order="F"` → `"C"` and watch them swap.

Want to play more?  

- Scale the dimensions up (e.g., 4096 × 4096) and time with `%timeit`—you'll spot the memory roofline we discussed.  
- Cast everything to `float16`; you'll see bandwidth pressure drop and the "slow" path close the gap.

Bottom line: On an M3 Ultra, column-major access is ~10× faster in this toy example—proof that matching the library's preferred layout (or letting it transpose behind the scenes) is non-negotiable for real workloads.

---

## Appendix B · Mini Lab — Expanded Version

[01-mini-lab-expanded.py](examples/01-mini-lab-expanded.py)

Try running this expanded version and interpret the results youself.

The following is the output of the expanded version on an M3 Ultra. 

```bash
% python 01-mini-lab-expanded.py
Dot product →    41.01
AXPY sample  → [-0.2782233   0.14136425  0.8898671 ]
GEMV checksum →   176.07
GEMM slice    → [-3.4731958  -6.2966533  -0.85809225 -3.1875205  -1.094734  ]

Memory-layout / dtype timing (lower is better)…
C-order  fp32       row    5.6 µs   col    2.5 µs
F-order  fp32       row    4.1 µs   col    1.3 µs
C-order  fp16       row    5.9 µs   col    1.9 µs
F-order  fp16       row    4.5 µs   col    1.6 µs

Fastest row  access : F-order  fp32  (  4.1 µs)
Fastest column access : F-order  fp32  (  1.3 µs)
Hint: C-order favours row walks, F-order favours column walks.
     Mixed dtypes show bandwidth vs compute trade-offs.

Scale test — dot-product latency (FP32)
  n=1024      4.6 µs
  n=4096      2.5 µs
Observed scaling factor ≈ 0.55  (ideal linear = 4.0)

Batch-boost GEMM (αA·B + βC) timing
  512x1024 @ 1024x256  → 0.001 s  (231.7 GFLOP/s)
  2048x4096 @ 4096x256  → 0.027 s  (160.2 GFLOP/s)

GEMM throughput speed-up (big vs baseline) ≈  0.7×
If this boost is < theoretical (×8 here), you're already bandwidth-bound; if ~linear, you're compute-bound.
```

Key takeaway from that benchmark:

* We scaled **m** and **k** by 4× (512 → 2048, 1024 → 4096) while **n** stayed at 256.  
* That yields **16×** more arithmetic work (FLOPs ∝ m × n × k) but only about **4–16×** more memory traffic, depending on which operand dominates.  
* Measured throughput fell from 231 GFLOP/s to 160 GFLOP/s—roughly a **0.7×** slowdown despite the bigger job.

What happened?  The kernel slid past the M3 Ultra's memory-bandwidth roof.  As arithmetic intensity (FLOPs/byte) dropped, DRAM became the choke point: on-chip caches couldn't hold the larger tiles, so every multiply waited on 819 GB/s links.

Roofline moral: when you're bandwidth-bound, piling on bigger matrices yields diminishing—or negative—returns. 
Better strategy: fuse or batch many *moderate* GEMMs so their tiles still live in cache.

Below is a benchmark interpretation guide for the output from my M3 Ultra (819 GB/s, 512 GB UMA). This explains what each measurement reveals about GPU performance characteristics.

Focus on the **patterns**, not the exact numbers—your next run will differ a little.

---

### 🚩 1–4  Sanity-check math

- Dot product (41.01), AXPY sample, GEMV checksum, GEMM slice → merely prove all four BLAS levels executed without NaNs/Infs. They are **not** speed metrics.

### ⚡ 5 Memory-layout & dtype micro-bench

```
                row µs   col µs
C-order  fp32    5.6      2.5
F-order  fp32    4.1      1.3   ◀ fastest in both axes
C-order  fp16    5.9      1.9
F-order  fp16    4.5      1.6
```
What it means:

1. **Contiguous always wins.**  
   * In C-order arrays the *row* is unit-stride; in F-order arrays the *column* is.  
   * The contiguous direction routinely shows a 2–4 × edge in this toy test.

2. **Stride—not brand—drives the gap.**  Swap layouts and the advantage swaps with it; the hardware isn't secretly favoring Fortran, it's rewarding unit-stride walks.

3. **FP16 ≈ FP32 here.**  
   Compute cost is trivial; the loop is bandwidth-bound, so cutting arithmetic precision yields little speed-up.

Cheat-sheet:

- Need **row-wise** kernels? Store data in **C-order**.  
- Need **column-wise** kernels (e.g., BLAS/CuBLAS defaults)? Keep or convert to **F-order**.  
Mixing the two costs you a solid 2–4 × on an M3 Ultra.

### 📏 6 Dot-product scaling test

```
n = 1 024  → 4.6 µs
n = 4 096  → 2.5 µs   (scaling factor ≈ 0.55, ideal = 4.0)
```

Why *bigger* is *faster*

* 4096 elements finally saturate the 128-byte vector units, hiding Python-call overhead.  
* The test runs once; jitter from OS scheduling and Turbo clocks makes micro-benchmarks slippery.  
* Take-away: for sub-millisecond kernels, *launch overhead* can dwarf arithmetic.

### 🏋️ 7 Batch-boost GEMM timing

```
512×1024 @ 1024×256  → 0.001 s  (231 GFLOP/s)
2048×4096 @ 4096×256 → 0.027 s  (160 GFLOP/s)   ⇐ 0.7 × slower per flop
```
Interpretation:

- The second workload is **8× larger** but throughput **drops 30 %**.  
- If GEMM were compute-bound you'd expect ≈ constant or slightly better GFLOP/s.  
- Falling efficiency ⇒ **memory-bandwidth ceiling reached**—even 819 GB/s can't feed 4096-wide tiles fast enough.

Rule of thumb emitted by the script:

"If speed-up < theoretical (×8 here), you're bandwidth-bound; if ≈ linear, you're compute-bound."

### 🧭 Summary for M3 Ultra owners

1. Layout = performance. Stick to whichever order your kernel touches contiguously.  
2. Very small vector ops are dominated by Python & interpreter overhead—batch them or go C/Metal.  
3. Large GEMMs hit the 819 GB/s roof quickly; tiling tricks or mixed-precision may help, but only so much.  
4. The script now prints its own verdicts, turning raw timings into actionable guidance—handy when you swap dimensions, dtypes, or hardware.

---

[⇧ Back&nbsp;to&nbsp;README](../README.md)