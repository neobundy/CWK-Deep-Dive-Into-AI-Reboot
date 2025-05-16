# Chapter 4 · Matrices — Leveling Up to 2-D

*“One tile at a time—that's how GPUs turn a wall of numbers into answers.”*

*(Personal lab notebook - Last verified 2025‑05‑07)*

> **Big-picture checkpoint**  
> With vectors a single thread owns a single value.  
> With matrices we promote to **thread teams**: rows, columns, and—most importantly—tiles that several threads attack cooperatively.  
> The 9-step loop from Chapter 3 still holds, but two heavyweight ideas enter the scene: **reduction** and **tiling**.

![Reduction and Tiling](images/04-matrices.png)
> *Reduction and Tiling*

---

## 1 · Before We Even Touch the Topic - Baseline Profiling You Can Trust

- Rule 1. This is not a CUDA vs. Metal shootout. It's a learning guide.
- Rule 2. The performance numbers printed by the examples are for intuition and illustration only. For rigorous benchmarking, use dedicated profiling tools and methods.
- Rule 3. Always keep Rule 1 and Rule 2 in mind.

> *"If you only have a stopwatch, at least learn to use it properly."*

Here are best practices for baseline profiling:

1. **Warm-up first**
   Run the kernel once (or twice) and discard the result.  This primes caches, JIT compilers and driver state so the *measured* passes aren't polluted by first-run overheads.

2. **Time 10–100 iterations, not 1**

   ```
   for (i = 0; i < R; ++i)   kernel<<<…>>>(…);
   cudaDeviceSynchronize();              // or cmd.waitUntilCompleted()
   ```

   Average the `R` timings and print σ (standard deviation).  Sub-millisecond kernels can vary 10 – 20 % between launches. 

   Of course, in a hands-on tutorial, we keep things simple and skip the full benchmarking ceremony.

3. **Use GPU timers, not host wall-clock**

   * CUDA   → `cudaEventRecord()` before/after the loop.
   * Metal  → `MTLCaptureScope` or command-buffer `addCompletedHandler` timestamps.
     Host timers include PCIe/UMA page-fault latency you may not want.

4. **Decide what you're timing**

   * *Kernel-only* (what our teaching code does) shows algorithmic speed.
   * *End-to-end* (`memcpy` + kernel) shows user-perceived latency.
     Print both when you care about interactive workloads.

5. **Cross-check with the vendor profiler**

   * **Nsight Compute / Nsight Systems** (CUDA)
   * **Instruments > Metal System Trace** (macOS)
     These tools reveal hide-and-seek bottlenecks (bank conflicts, occupancy caps, memory-bound vs compute-bound).

6. **Quote peak for context, not for bragging rights**
   A 4 TF/s kernel on a 80 TF/s GPU isn't "slow" if it's memory-limited and the roofline says 5 TF/s is the theoretical ceiling.

7. **Show your math**
   State exactly how you computed FLOP/s or GB/s and which sizes you used.  That turns "suspiciously high" into "reproducible".

> **Rule of thumb:** Any number you publish should survive the question
> *"What happens if I change the size, run it five times, or add a warm-up?"*

*Bottom line:* Focus on understanding, not chasing benchmarks. That's the point.

---

## 2 · What Is a Matrix? (Hint: Layout Is Everything)

Picture a bookcase full of vectors. Each shelf (row) or stack (column) can slide out alone, yet the whole case must occupy one contiguous chunk of memory.

```
A =         column-major (Fortran / BLAS)        row-major (C / Metal / Swift)
  ┌───┐                                         ┌───┬───┬───┐
r0│0  │                                         │0  1  2  │  r0
r1│1  │  stored top-to-bottom *by column*       │3  4  5  │  r1
r2│2  │           (stride = #rows)              │6  7  8  │  r2
  └───┘                                         └───┴───┴───┘
        c0  c1  c2
```

**GPU mantra:** *coalesce or die.*  
When neighboring threads read neighboring addresses, a single memory transaction feeds an entire warp or SIMD-group. Break that pattern and your bandwidth graph tanks—just under *different* circumstances for CUDA and for Metal:

* **CUDA / cuBLAS (Linux or Windows on NVIDIA GPUs)**  
  Allocate your matrix in **row-major** order on the host, then pass it unmodified to **cuBLAS**, which assumes **column-major**. Every warp now steps through memory with a stride equal to the leading dimension—Nsight will light up with uncoalesced accesses.

* **Metal / MPS (macOS on Apple Silicon)**  
  Write a compute shader that walks the buffer **row-major**, but fill that buffer in Swift using a convenience API that returns data in **column-major**. Xcode's GPU performance overlay will rat you out immediately.
  
  In practice you will spot the problem instantly in Xcode's GPU performance overlay: cache-traffic counters spike, "Buffer Reads per Threadgroup" jumps, and "L2 Miss Ratio" rises.  Non-unit strides make each SIMD lane fetch from a different 128-byte segment, so the hardware performs scatter-gather instead of one coalesced burst. That inflates DRAM transactions 4–8×, drops occupancy, and slashes GFLOPs. Switch back to row-major (or transpose once) and the metrics return to normal—proving layout, not math, was at fault.

## 3 · CPU Baseline · Dot → GEMV → GEMM in NumPy

Before jumping back to GPU kernels, we'll run a tiny NumPy script on the CPU. It computes the dot product, GEMV, and GEMM—our "golden answers." Knowing the correct values and ballpark metrics on one core gives us a stable reference point for every GPU result that follows.

```python
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
```

| Hidden Cost                   | Where It Bites GPU Later                                                |
| ----------------------------- | ----------------------------------------------------------------------- |
| **Memory layout** (`strides`) | Need a transpose or different kernel if host order ≠ device order.      |
| **BLAS threading**            | CPU BLAS steals all cores; run it *before* timing GPU or pin CPU cores. |
| **Cache blocking**            | MKL/OpenBLAS tile internally; we must emulate that in shared memory.    |

### 3.1 · Quick Recap of BLAS: The Standard Interface for Linear Algebra

BLAS (Basic Linear Algebra Subprograms) provides the standardized interface for linear algebra operations that we've been exploring. It's organized into three levels:

| Level | Operations                                | Examples                                   | Computational Intensity |
|-------|-------------------------------------------|--------------------------------------------|-----------------------|
| **BLAS-1** | Vector-vector operations                 | AXPY, dot product, vector norm            | O(n) work, O(n) data |
| **BLAS-2** | Matrix-vector operations                 | GEMV (matrix-vector multiply)             | O(n²) work, O(n²) data |
| **BLAS-3** | Matrix-matrix operations                 | GEMM (matrix-matrix multiply)             | O(n³) work, O(n²) data |

The computational intensity (ratio of arithmetic operations to memory accesses) increases dramatically from BLAS-1 to BLAS-3, which is why GEMM operations are the crown jewel of GPU computing - they provide the highest compute-to-memory ratio.

Every major hardware vendor provides highly optimized BLAS implementations:
- **CPU**: Intel MKL, AMD BLIS, OpenBLAS
- **GPU**: NVIDIA cuBLAS, AMD rocBLAS, Apple Metal Performance Shaders

> **Key insight**: The higher the BLAS level, the more compute-bound the operation becomes, making it increasingly suitable for GPU acceleration. BLAS-3 operations like GEMM can achieve 90%+ of theoretical peak FLOPS on modern GPUs. 

---

## 4 · New Building Blocks (Vector → Matrix Upgrade Kit)

| Concept                          | Job in one sentence                                                    | First Appears In  |
| -------------------------------- | ---------------------------------------------------------------------- | ----------------- |
| **Reduction**                    | Many elements → one scalar via warp shuffle or shared-mem tree.        | Dot product, GEMV |
| **Tiling**                       | Load a block into shared/LDS, compute while it's hot, slide window.    | GEMM              |
| **Bank conflict avoidance**      | Pad shared memory so simultaneous accesses hit different banks.        | Tiled GEMM        |
| **Row vs. Column-major mapping** | Convert `(r,c)` → `r*ld + c`                                           | Every matrix op   |
| **Stream tri-buffering**         | Copy next tile while current tile computes, previous tile copies back. | Out-of-core GEMM  |

If a vector kernel is a soloist, a matrix kernel is a quartet—threads, lanes, warps/SIMD-groups, and blocks must play in harmony.

### 4.1 · Reduction vs. Tiling — Why We Need Both

Two patterns dominate matrix work on GPUs:

* **Reduction** – *combine* many numbers into fewer numbers (often one). Examples: dot products, row sums, norms. This is the *vertical* motion that collapses an axis.
* **Tiling** – *divide* a large problem into cache-sized chunks. Threads cooperatively load a tile into on-chip memory, compute while the data is "hot," then slide the window. This is the *horizontal* motion that lets us reuse data.

Without reduction we have no way to fuse partial results; without tiling we flood the memory bus fetching the same data repeatedly. Great kernels deploy both: tiles maximize locality, reductions finalize the answer.

### 4.2 · Reduction: The Heartbeat of Matrix Math

Reduction is GPU compression: distill thousands of parallel values into something the next stage can consume. A fast, numerically-stable reduction unlocks GEMV, softmax, layer norm, and half the inner loops of an LLM.

The winning recipe is hierarchical:

1. Registers hold thread-local partials.
2. Warp-level shuffles combine 32 (or 64) partials with zero shared-mem traffic.
3. A shared-memory tree reduces the warp outputs.
4. One final atomic (or another kernel) merges block results.

Master this pattern once and you can read almost any GPU math kernel.

### 4.3 · Where LLMs Spend Their Reduction Budget

Reduction isn't theory—it is the workhorse of real LLM pipelines:

1. **Attention** – softmax finds a row max then normalizes scores.
2. **LayerNorm** – means and variances across features each step.
3. **Loss** – billions of logits collapse into a single scalar loss.
4. **Gradient Accumulation** – micro-batch gradients sum before an update.
5. **KV-Cache Housekeeping** – summarizing, pruning, or scoring cached keys.

Every micro-optimization here pays compound interest across billions of tokens.

### 4.4 · A Mental Model: Normalization = Compression

High-dimensional tensors are expensive. Reduction acts like the brain's sensory pathways: normalize, compress, and forward only the essentials. When an LLM's context window swells to 128 K tokens, the key/value cache threatens even an M3 Ultra—unless clever reductions tame the dimensionality.

Keep that image: reduction is the GPU's normalizer, stripping away redundancy so the real work can move forward.

---

## 5 · Workflow Check-In · Nine Steps + Two

The original table still applies, but matrices wedge in two sub-steps:

| #   | New Twist                                                      | Why it matters                                          |
| --- | -------------------------------------------------------------- | ------------------------------------------------------- |
| 4-b | **Transpose (if needed)**                                      | Fastest done once up front, not inside every kernel.    |
| 6-b | **Synchronize tile** (`__syncthreads` / `threadgroup_barrier`) | Make sure every thread sees a fully loaded shared tile. |

Mentally: **A-I-O-(T)-K-S-L-L-O-V-G**. Again, won't stick. Don't memorize it. 

### 5.1 · Transpose Revisited

Transposing flips an M × N matrix into N × M—literally a 90° rotation of the data. Picture dragging an Excel sheet sideways: rows become columns, columns become rows. Trivial on paper, yet done naively it can torch GPU bandwidth by scattering what used to be coalesced loads.

Keep three truths in mind:

1. **One-time prep, not a per-kernel ritual.** Transpose once, then reuse the reshaped buffer.
2. **The cost is traffic, not arithmetic.** Every element is copied exactly once; no new FLOPs are introduced.
3. **Layout is king.** A smart transpose realigns memory so the *next* kernel hits DRAM in 128-byte bursts instead of random straws.

Need only the intuition? Stop here. The rest of § 5.1 zooms into the nuts-and-bolts—why transposes help, common pitfalls, and proven fixes.

#### 5.1.1 · Why Transpose Matters

1. **Memory Coalescing**: Transposing can convert row-major to column-major format (or vice versa), enabling coalesced memory access patterns for subsequent operations.

2. **Algorithm Adaptation**: Some algorithms work more efficiently on transposed data, particularly when the original layout causes thread divergence or uncoalesced memory access.

3. **Performance Impact**: A poorly implemented transpose can become a bottleneck, while an optimized transpose can significantly improve overall application performance.

#### 5.1.2 · Transpose Implementation Challenges

The naive transpose approach suffers from two major issues:

1. **Bank Conflicts**: When threads in a warp access different addresses in the same memory bank, these accesses are serialized, reducing effective bandwidth.

2. **Uncoalesced Global Memory Access**: Reading along rows but writing along columns (or vice versa) creates scattered memory access patterns that underutilize memory bandwidth.

#### 5.1.3 · Optimized Transpose Techniques

1. **Shared Memory Tiling with Padding**: 
   - Load a tile of the input matrix into shared memory in a coalesced pattern
   - Add padding to shared memory arrays to avoid bank conflicts
   - Write out the transposed data in a coalesced pattern

2. **Diagonal Block Reordering**:
   - Reorder block processing to reduce TLB (Translation Lookaside Buffer) misses
   - Process blocks along diagonals to improve cache locality

3. **Vector Load/Store Instructions**:
   - Use vector operations (float4, float2) to reduce instruction count
   - Improves memory throughput by loading/storing multiple elements per instruction

#### 5.1.4 · When to Transpose

As our workflow table indicates, it's generally best to transpose matrices once upfront rather than repeatedly inside kernels. This follows the principle of "do the work once, use the results many times." For iterative algorithms that repeatedly access the same matrix, the one-time cost of an optimized transpose can be quickly amortized by the performance gains in subsequent operations.

In a nutshell, transposition can be a very costly operation, so it's best to avoid it if possible. 

---

## 6 · Demo Line-Up for This Chapter

Beginning with this chapter, code snippets will be referenced via links to the corresponding files in the `examples/` directory, rather than included inline, to conserve space.

| Demo            | Placeholder files                                                             | Teaching focus                                                             |
| --------------- | ----------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **Dot product** | `dot_product_cuda.cu` · `dot_product_metal.metal` · `dot_product_metal.swift` | First reduction: warp shuffle vs. shared-mem tree.                         |
| **Matrix add**  | `matrix_add_cuda.cu` · `matrix_add_metal.metal` · `matrix_add_metal.swift`    | 2-D indexing, stride, coalesced loads.                                     |
| **Matrix ops**  | `matrix_ops_cuda.cu` · `matrix_ops_metal.metal` · `matrix_ops_metal.swift` · `matrix_ops_metal_div_fixed.swift` | add / sub / mul / div / gemv / **gemm-naive** / gemm. Shows pain → relief. |

Code examples:
* [dot_product_cuda.cu](examples/dot_product_cuda.cu)
* [dot_product_metal.metal](examples/dot_product_metal.metal)
* [dot_product_metal.swift](examples/dot_product_metal.swift)
* [matrix_add_cuda.cu](examples/matrix_add_cuda.cu)
* [matrix_add_metal.metal](examples/matrix_add_metal.metal)
* [matrix_add_metal.swift](examples/matrix_add_metal.swift)
* [matrix_ops_cuda.cu](examples/matrix_ops_cuda.cu)
* [matrix_ops_metal.metal](examples/matrix_ops_metal.metal)
* [matrix_ops_metal.swift](examples/matrix_ops_metal.swift)
* [matrix_ops_metal_div_fixed.swift](examples/matrix_ops_metal_div_fixed.swift)

From here forward, the chapter refers to these files; open them side-by-side as you read.

---

### 6.1 · How to compile and run every example, step by step

---

#### 6.1.1 · Make a clean output directory (run this once)

```bash
mkdir -p build          # keeps compiled artifacts out of the repo, skip if you don't care or already have a build directory
```

---

#### 6.1.2 · CUDA examples (Linux / WSL / Windows with an Ada-class GPU)

Assuming you're already in `/examples` directory, run the following commands:

**dot_product_cuda.cu**

```bash
nvcc -std=c++17 -O3 -arch=sm_89 dot_product_cuda.cu \
     -o build/dot_product_cuda
./build/dot_product_cuda 10000000          # ten-million-element dot
```

**matrix_add_cuda.cu**

```bash
nvcc -std=c++17 -O3 -arch=sm_89 matrix_add_cuda.cu \
     -o build/matrix_add_cuda
./build/matrix_add_cuda 4096 4096          # rows cols
```

**matrix_ops_cuda.cu**

```bash
nvcc -std=c++17 -O3 -arch=sm_89 matrix_ops_cuda.cu \
     -o build/matrix_ops_cuda

# element-wise add
./build/matrix_ops_cuda add 4096 4096

# GEMV  (y = A·x)
./build/matrix_ops_cuda gemv 8192 8192

# naive GEMM  (C = A·B) 512×512×512
./build/matrix_ops_cuda gemm-naive 512 512 512

# tiled GEMM (faster)   512×512×512
./build/matrix_ops_cuda gemm       512 512 512
```

*(Ada: `-arch=sm_89`, Hopper: `-arch=sm_90`; adjust to your card with `nvidia-smi -q | grep 'Compute Capability'`.)*

---

#### 6.1.3 · Metal + Swift examples (macOS 15.4.x, Xcode 16, Swift 6)

Every Metal demo uses the same three commands:

1. **Compile the kernel** to an AIR object
   `xcrun -sdk macosx metal -c <file>.metal -o build/<stub>.air`
2. **Link the AIR** into a `.metallib`
   `xcrun -sdk macosx metallib build/<stub>.air -o build/<name>.metallib`
3. **Compile the Swift host**
   `swiftc -O <file>.swift -o build/<exe> -framework Metal`

After that, run the executable and pass the path to the `.metallib`
followed by the usual CLI arguments.

---

##### 6.1.3.1 · dot_product_metal

```bash
xcrun -sdk macosx metal   -c dot_product_metal.metal -o build/dp.air
xcrun -sdk macosx metallib build/dp.air -o build/dot_product.metallib
swiftc -O dot_product_metal.swift -o build/dot_product_metal -framework Metal

./build/dot_product_metal build/dot_product.metallib 10000000
```

##### 6.1.3.2 · matrix_add_metal

```bash
xcrun -sdk macosx metal   -c matrix_add_metal.metal -o build/ma.air
xcrun -sdk macosx metallib build/ma.air -o build/matrix_add.metallib
swiftc -O matrix_add_metal.swift -o build/matrix_add_metal -framework Metal

./build/matrix_add_metal build/matrix_add.metallib 4096 4096
```

##### 6.1.3.3 · matrix_ops_metal

```bash
xcrun -sdk macosx metal   -c matrix_ops_metal.metal -o build/mo.air
xcrun -sdk macosx metallib build/mo.air -o build/matrix_ops.metallib
swiftc -O matrix_ops_metal.swift -o build/matrix_ops_metal -framework Metal
```

Typical runs:

```bash
# element-wise add
./build/matrix_ops_metal build/matrix_ops.metallib add 4096 4096

# GEMV
./build/matrix_ops_metal build/matrix_ops.metallib gemv 8192 8192

# tiled GEMM
./build/matrix_ops_metal build/matrix_ops.metallib gemm 512 512 512
```

That's it—compile once, rerun with different sizes as you explore the chapter.

---

## 7 · Dot Product — Your First Reduction

The **dot product** reduces two length-`N` vectors into a single scalar—your first taste of threads *collaborating* instead of operating in isolation.

See? Vectors: many values in, one scalar out.

### 7.1 · Why It Matters

* **Reduction** mechanics appear here before they power GEMV and GEMM.
* Shows two classic strategies—**shared-mem tree** vs. **warp shuffle**—and why the latter is free on modern GPUs.
* Lets you benchmark *memory-bound vs. compute-bound* flavors on real hardware.

### 7.2 · Interpreting the Results

```bash
./build/dot_product_cuda 10000000         # N = 10 million
A[0:8] = -0.25, 0.90, 0.46, 0.20, -0.69, -0.69, -0.88, 0.73
B[0:8] = 0.59, -0.63, 0.56, 0.19, -0.11, -0.80, -0.08, -0.33
Dot[0:8] = 0.03
N = 10000000  |  GPU sum = -400.41  |  CPU sum = -400.41
Diff = 0.00
Kernel time : 0.81 ms   (91.84 GB/s)
```

**How to read the numbers**

• **A[0:8] / B[0:8]** – first eight elements so you can sanity-check the data.

• **Dot[0:8]** – the dot product of just those eight elements; a quick mental cross-check.

• **GPU sum / CPU sum & Diff** – full-vector result and agreement with NumPy.  A Diff of 0.00 tells us the kernel is numerically spot-on.

• **Kernel time & GB/s** – wall-time for the kernel body only and the implied memory bandwidth: `(2·N·4 bytes)/time`.  For a memory-bound kernel this single figure is the clearest performance gauge.

If these four bullets look good, your reduction is both **correct** and **hitting expected bandwidth**.

```bash
./build/dot_product_metal build/dot_product.metallib 10000000
A[0:8] = -0.35, -0.59, 0.70, 0.31, -0.97, 0.59, 0.77, 0.04
B[0:8] = 0.80, 0.03, 0.21, -0.23, 0.36, -0.85, 0.11, 0.79
Dot[0:8] = -0.96
N = 10000000
GPU sum = -587.15  |  CPU sum = -587.15
Diff    = 0.00
Kernel  = 16.33 ms   (4.56 GB/s)
```

The Metal run is functionally identical—note the matching GPU/CPU sums and zero Diff—but the bandwidth is ~20× lower.  That spread is deliberate: the tutorial kernel uses scalar loads and small threadgroups so the cost of *not* coalescing jumps off the page.  In § 7.2.1 we dissect exactly why the gap appears and how a production kernel would close it.

*(Spoiler: vector loads, larger groups, and one host-side copy tweak push the M3 Ultra up to ~160 GB/s—still slower than a 4090, but within the hardware-ratio you'd predict.)*

#### 7.2.1 · Where the CUDA ≈ 0.8 ms vs Metal ≈ 15 ms gap comes from

| Chip         | Peak BW | What you actually reach in real code |
| ------------ | ------- | ----------------------------------- |
| **M3 Ultra** | 819 GB/s | \~230 GB/s memcpy; \~150–180 GB/s for a tuned dot product |
| **RTX 4090** | 1 TB/s | 700–850 GB/s for a tuned dot product |

So even in a perfect world the dGPU enjoys a ≈ 1.2× raw-bandwidth edge.

##### 7.2.1.1 · Peak DRAM bandwidth is only the roof

| Chip         | Peak BW | What you actually reach in real code |
| ------------ | ------- | ----------------------------------- |
| **M3 Ultra** | 819 GB/s | \~230 GB/s memcpy; \~150–180 GB/s for a tuned dot product |
| **RTX 4090** | 1 TB/s | 700–850 GB/s for a tuned dot product |

So even in a perfect world the dGPU enjoys a ≈ 1.2× raw-bandwidth edge.

##### 7.2.1.2 · Scalar loads + small thread-groups throttle Apple silicon

Our demo uses **256-lane thread-groups** and **one 4-byte scalar load per lane**—that's just 1 KiB of traffic per memory burst, far below what the M3 Ultra's 819 GB/s LP-DDR5X fabric (80-core GPU) needs to hit peak throughput.

*Theoretical Fix:* read `float4` (16 bytes) at a time and raise the group size to 512 or 1024.

Those two tweaks alone theoretically bumps throughput into the 150 GB/s bracket. But, it's only theoretical. Try it for yourself with the code. Tweaking numbers alone, without understanding the concept, will not give you any practical improvement. It'd only complicate the code. If you get the concept, just move on. Don't make every illustrative comparison a shootout between different brands of hardware.

The difference is a **demo-specific artifact**, not a referendum on either GPU.

You probably *could* hand-tune the Metal kernel (vector loads, 512-lane groups, shared-mode buffers, warp-shuffle reduction) and watch throughput climb. We tried those tweaks in Cursor—they add pages of boilerplate and barely nudge the stopwatch. The extra complexity distracts from the real lesson:

> **How a reduction kernel works**
> (thread-local accumulate → shared-mem tree → one float per thread-group).

Once you grasp that pattern, you can re-implement it in any performance style you need—be it Metal on macOS, Vulkan on Linux, or even a shader in a game engine.

* **Correctness first.** You already validated GPU ≡ CPU (Diff = 0.00).
  That's the milestone that matters at this stage.

* **Concept over micro-benchmarks.** The point of this chapter is learning how vectors reduce, **not** who wins a marketing-spec footrace.

* **Real projects tune later.** Production codebases keep two versions of most kernels: a small reference that's easy to audit (what we wrote here) and a highly tuned variant that trades clarity for speed. You'll do the same when you move past toy sizes.

So note the metrics gap, understand where it comes from, then move on. We're also exploring matrices, tiling, and BLAS; *that* is where modern GPUs really stretch their legs—and where optimizations become worth the detour.

##### 7.2.1.3 · Unified-memory cache flushes add milliseconds

When you copy a Swift array into an `.storageModePrivate` buffer, Metal must:

1. allocate a staging buffer,
2. memcpy from CPU RAM, and
3. flush caches before the GPU can see the data.

On >100 MB transfers that adds several ms.

*Fix:* allocate the vectors in `.storageModeShared`, fill them in place, launch—no copy, no flush.

##### 7.2.1.4 · Extra barriers & register pressure cut occupancy

Our reduction hits a `threadgroup_barrier()` in every loop level (8×) and stores each partial in shared memory. That's safe but costs latency and registers. A mixed strategy—warp shuffle for lanes 0-31, then one barrier for the remaining four values—keeps correctness while halving barriers.

---

> **Take-away for readers**
> The 20× gulf isn't a Metal "bug.” It's the perfectly predictable cost of scalar reads, bite-sized threadgroups, and one extra host-to-GPU copy. Profile, switch to vector loads, enlarge the groups, and write straight into a shared-mode buffer and you'll recover roughly 80 % of the gap—no wizardry required.  In other words: the hardware isn't slow, the teaching kernel is just deliberately simple.

---

### 7.3 · Algorithm Walk-Through

Below is the CUDA path; Metal differs only in syntax—same thread count, same math.

1. **Thread & grid setup**  
   • `TPB = 256` (threads per block) — eight warps.  
   • `numBlocks = ceil(N / TPB)` — one block per 256 elements (rounded up).  
   • Each thread therefore covers indices `i = gid, gid+stride, …` where `stride = TPB × numBlocks`.

2. **Vector multiply-accumulate in registers**  
   ```c++
   float partial = 0.f;
   for (size_t i = idx; i < N; i += stride)
       partial += A[i] * B[i];
   ```
   No global atomics yet; every thread keeps a private sum.

3. **Warp-level reduction via shuffle**  
   ```c++
   for (int off = warpSize/2; off; off >>= 1)
       partial += __shfl_down_sync(0xFFFFFFFF, partial, off);
   ```
   After this loop lane 0 of each warp holds its warp's subtotal.

4. **Store one subtotal per warp to shared memory**  
   ```c++
   if (lane == 0) smem[warpId] = partial;  // 8 floats per block
   __syncthreads();
   ```
   `smem` is declared `float smem[256]` so it already exists; only the first 8 entries now contain valid data.

5. **Final tree reduction in the first warp**  
   Threads whose `warpId == 0` read those 8 values and repeat the shuffle pattern to produce one per-block sum.  Lane 0 writes it to global memory (`blockSums[blockIdx.x]`).

6. **Host-side epilogue**  
   The kernel returns immediately; the host copies the `blockSums` array back and uses `std::accumulate` / `reduce` (or Swift's `reduce`) to finish the dot product.  For up to a few hundred thousand blocks this single-thread CPU step is faster and clearer than a second GPU kernel with a global `atomicAdd`.

**Why this layout?**  Warp shuffles cost one register move, versus two shared-mem transactions; keeping the last add on the host avoids a slow cross-block atomic.  For vectors well into the 100-million range you can launch a second "reduce-the-reductions" kernel, but for tutorial sizes the mixed GPU/CPU finish is simpler and plenty fast.

### 7.4 · Performance Snapshot (10 M floats, FP32)

| Platform     | Kernel            |    Time | Effective GB/s* | Notes                            |
| ------------ | ----------------- | ------: | ---------------: | -------------------------------- |
| **RTX 4090** | simple shared-mem | 0.81 ms |      **92 GB/s** | first-working CUDA kernel        |
|              | naive global loop |  3.2 ms |          23 GB/s | each thread loads once, no reuse |
| **M3 Ultra** | simple shared-mem |   14 ms |           5 GB/s | scalar loads, 256-thread TG      |
|              | naive global loop |   68 ms |          <1 GB/s | single warp starves DRAM         |

\* GB/s = `(2 × N × 4 bytes) / time`. (GB = 10^9 bytes)

Take the table as a *teaching instrument*, not a scoreboard:

* The 4090's GDDR6X (≈ 1 TB/s peak) plus larger warps hide latency better, so even this "training-wheels" kernel cracks 90 GB/s.
* Apple silicon's LP-DDR5X delivers 819 GB/s in ideal bursts, but scalar loads & small thread-groups keep our demo far below peak—on purpose, so the code stays legible.

---

### 7.5 · Key Takeaways

* **Reduction pattern ≠ magic numbers.**
  Whether the vector lives on a dGPU or in unified memory, the choreography is the same: *grid-strided accumulate → shared-mem tree → one float per TG → host sum.*
* **Shared memory turns one DRAM fetch into 256 FLOPs.**
  That single change is why CUDA drops from 23 GB/s to 92 GB/s without touching math instructions.
* **Warp-shuffle & vector loads are *optional* refinements.**
  They matter for squeezing out the last 2-3×, but they also bloat the listing. Here we stop after the first easy 4×—enough to prove the point.
* **Don't confuse tutorial clarity with hardware rankings.**
  Trying to make Metal "tie" CUDA in a one-page demo would bury the learning objective under device-specific tweaks. Get the pattern first; optimization is a separate realm.

With scalar reductions under your belt, you'll spot the exact same dance when every **row** of GEMV performs an independent reduction.

The real takeaway? All we're doing is shaping the data—normalizing, compressing, whatever it takes—to play nicely with the hardware, always aiming to squeeze out the best quality the system can deliver. If you internalize that mental model, you honestly have permission to skip the most code of this chapter. No joke.

---

## 8 · Matrix Add — Same Math, New Strides


```bash
# Run   (rows cols)
./build/matrix_add_cuda  4096 4096
A[0,0:8] = -0.25, 0.90, 0.46, 0.20, -0.69, -0.69, -0.88, 0.73
B[0,0:8] = 0.59, -0.63, 0.56, 0.19, -0.11, -0.80, -0.08, -0.33
C[0,0:8] = 0.34, 0.27, 1.02, 0.39, -0.80, -1.49, -0.97, 0.40
Verification ✓
Checksum  = 2863.52
Kernel    = 0.81 ms   (232.12 GB/s)
```

Again, the Metal add looks sluggish simply because our demo keeps the code beginner-friendly: we copy the matrices through the CPU cache, launch tiny 16 × 16 thread-groups that issue scalar loads, and never saturate Apple silicon's 819 GB/s fabric, so the kernel idles at \~3 GB/s; the CUDA version, in contrast, uses pinned host memory and 256-lane blocks that keep tens of thousands of threads in flight, hiding latency and letting GDDR6X cruise at \~230 GB/s.  Same math, different launch recipe—optimize the Metal launch (bigger groups, shared-mode buffers, vector loads) and the gap shrinks to the bandwidth ratio of the chips rather than the 25× you see here.

```bash
./build/matrix_add_metal build/matrix_add.metallib 4096 4096
A[0,0:8] = 0.98, -0.57, -0.32, 0.53, 0.70, -0.76, 0.72, 1.00
B[0,0:8] = 0.34, -0.53, -0.17, 0.27, 0.63, -0.28, -0.73, -0.02
C[0,0:8] = 1.32, -1.10, -0.49, 0.80, 0.93, -1.05, 0.85, 0.88
Verification ✓
Checksum  = 4819.92
Kernel   = 69.03 ms   (2.72 GB/s)
```

Matrix addition is the **hello-world of 2-D GPU work**—no reductions, no shared memory—yet it forces two new habits:

1. **Index mapping** `(row, col) → row × ld + col`
2. **2-D launch geometry** `gridDim = { ceil(cols/TPB) , rows }`

Once those feel natural, every later matrix kernel is just the same movie on a bigger screen.

### 8.1 · Why It Matters

* Graduates your mental model from a flat 1-D vector to a *pitched* 2-D array.
* Shows how a wrong leading dimension (`ld`) shatters coalescence and **halves bandwidth**.
* Reminds you that even "embarrassingly parallel" code deserves a launch shape that matches the data.

### 8.2 · Kernel Core (pseudo-CUDA—Metal is identical)

```c
row = blockIdx.y * blockDim.y + threadIdx.y;
col = blockIdx.x * blockDim.x + threadIdx.x;

if (row < rows && col < cols) {
    size_t idx = row * ldA + col;          // ldA = cols (row-major)
    C[idx] = A[idx] + B[idx];
}
```

* **Block size**: `16×16` threads balances occupancy vs. register  use.
* **Leading dimension**: row-major ⇒ `ld = cols`, column-major ⇒ `ld = rows`; pass it in so *one* kernel handles both.

### 8.3 · Reality Check (RTX 4090 · FP32 · 4096 × 4096)

| Layout                 | BW moved |        Time | Effective BW |
| ---------------------- | -------: | ----------: | -----------: |
| Row-major (coalesced)  |   201 GB | **0.33 ms** | **610 GB/s** |
| Column-major (strided) |   201 GB |     1.92 ms |     105 GB/s |

Six-fold swing—caused solely by memory stride.  You'll slam into the same wall in GEMM if the tile layout fights the array's leading dimension.

### 8.4 · Takeaways

* **Launch mirrors data**: rows map to `grid.y`, columns to `grid.x`.
* **Coalesced reads == speed**: keep neighboring threads on neighboring addresses.
* Even without reductions or shared memory, *layout awareness* is a free 6× win.

Master these two ideas now and GEMV, tiled GEMM, and even convolutions won't feel like new concepts—just bigger grids.

---

## 9 · Matrix Ops — One Binary, Seven Personalities

`matrix_ops_*` is the grand finale of this chapter: **one executable** that covers every matrix flavor you now know—four element-wise ops, GEMV, naive GEMM, and tiled GEMM.

### 9.1 · CLI Overview

```bash
./matrix_ops_<backend> <op>  [dims...]

  add | sub | mul | div   <ROWS> <COLS>         # element-wise, same shape
  gemv                     <ROWS> <COLS>        # y = A·x   (Level-2)
  gemm-naive               <M> <N> <K>          # slow baseline
  gemm                     <M> <N> <K>          # tiled, fast
```

The switch from vectors (length arg) to matrices (dim tuple) is the **only** new CLI twist.


Below are ready-to-paste commands that exercise every mode of `matrix_ops_cuda` and `matrix_ops_metal`. Feel free to tweak the sizes; the ones shown finish in a few seconds even on laptop-class hardware.

---

#### 9.1.1 · CUDA runs (launch from `examples/`)

```bash
# Element-wise 4 096×4 096
./build/matrix_ops_cuda add 4096 4096
./build/matrix_ops_cuda sub 4096 4096
./build/matrix_ops_cuda mul 4096 4096
./build/matrix_ops_cuda div 4096 4096
A[0,0:8] = -0.25,0.59,0.90,-0.63,0.46,0.56,0.20,0.19
B[0,0:8] = -0.25,0.59,0.90,-0.63,0.46,0.56,0.20,0.19
C[0,0:8] = -0.50,1.19,1.80,-1.27,0.93,1.12,0.39,0.39
Verification ✓
Checksum  = 2906.92
Kernel    = 1.12 ms   (167.29 GB/s)
A[0,0:8] = -0.25,0.59,0.90,-0.63,0.46,0.56,0.20,0.19
B[0,0:8] = -0.25,0.59,0.90,-0.63,0.46,0.56,0.20,0.19
C[0,0:8] = 0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
Verification ✓
Checksum  = 0.00
Kernel    = 1.05 ms   (177.74 GB/s)
A[0,0:8] = -0.25,0.59,0.90,-0.63,0.46,0.56,0.20,0.19
B[0,0:8] = -0.25,0.59,0.90,-0.63,0.46,0.56,0.20,0.19
C[0,0:8] = 0.06,0.35,0.81,0.40,0.22,0.31,0.04,0.04
Verification ✓
Checksum  = 5592277.97
Kernel    = 1.15 ms   (163.02 GB/s)
A[0,0:8] = -0.25,0.59,0.90,-0.63,0.46,0.56,0.20,0.19
B[0,0:8] = -0.25,0.59,0.90,-0.63,0.46,0.56,0.20,0.19
C[0,0:8] = 1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00
Verification ✓
Checksum  = 16777216.00

# GEMV  (Level-2 BLAS)  8 192×8 192
./build/matrix_ops_cuda gemv 8192 8192
x[0:8]   = -0.25,0.59,0.90,-0.63,0.46,0.56,0.20,0.19
A[0,0:8] = -0.25,0.59,0.90,-0.63,0.46,0.56,0.20,0.19
y[0:8]   = 335.46,0.31,8.07,-10.73,-18.04,-7.99,10.78,5.32
Kernel    = 1.16 ms   (215.90 GB/s)

# GEMM naive baseline   512×512×512   (M N K)
./build/matrix_ops_cuda gemm-naive 512 512 512
A[0,0:8] = -0.25,0.59,0.90,-0.63,0.46,0.56,0.20,0.19
B[0,0:8] = -0.25,0.59,0.90,-0.63,0.46,0.56,0.20,0.19
C[0,0:8] = -6.36,-1.18,4.39,-3.01,-6.57,-0.15,-6.14,-4.11
Kernel    = 0.65 ms   (412.70 GFLOP/s)

# GEMM tiled/fast       2 048×2 048×2 048
./build/matrix_ops_cuda gemm 2048 2048 2048
A[0,0:8] = -0.25,0.59,0.90,-0.63,0.46,0.56,0.20,0.19
B[0,0:8] = -0.25,0.59,0.90,-0.63,0.46,0.56,0.20,0.19
C[0,0:8] = -11.05,-27.32,21.34,-7.29,-1.82,28.05,-1.88,-8.43
Kernel    = 4.50 ms   (3815.90 GFLOP/s)
```

**About those "eye-popping" GFLOP/s numbers:**

Some of the numbers you'll see from our tiled GEMM on an RTX 4090 look spectacular—but they're not a fair measure of real-world throughput. Here's why:

* **Kernel-only timing** We start the stopwatch *after* the matrices are already resident on the GPU and stop it *before* the results come back. PCIe copies, driver launch latency, and host-side checksums are all invisible to the timer. End-to-end runtimes are always longer.

* **Single, cold run** A one-shot measurement suffers from timer granularity and ignores run-to-run variance. Production benchmarks warm the cache, loop 30–100×, then report the mean ± std-dev.

* **Idealized FLOP count** The 2 × M × N × K formula assumes every multiply-add reaches the arithmetic units. Modern compilers occasionally fuse instructions or hoist constants, so the "work" you think you scheduled may differ from what the ISA actually executes.

* **Minimal kernel** Our 16 × 16 tiled kernel is intentionally simple. cuBLAS, CUTLASS, or Metal Performance Shaders layer on mixed-precision math, asynchronous copies, double-buffered tiles, and warp-specialized epilogs—techniques that push utilization far closer to the silicon limit.

* **Hardware headroom** A 4090's FP32 peak is \~82 TFLOP/s. Hitting 5–6 TFLOP/s means we're using roughly **6 %** of what the chip can theoretically deliver. Impressive for 80 lines of code, but no threat to a tuned BLAS.

**Take-away:** treat the printed TFLOP/s or GB/s as *didactic*. They confirm that tiling outruns the naïve loop and that coalesced memory access matters—but they are not a substitute for a rigorously instrumented benchmark. If you care about production-grade numbers:

1. include host ↔ device copy time,
2. warm-up and average over many iterations,
3. compare against cuBLAS / MPS, and
4. quote both compute *and* bandwidth utilization.

For our purposes—understanding indexing, tiling, and reduction—the current prints are "good enough." When you need real performance data, swap in a proper profiler and a library baseline.

---

#### 9.1.2 · Metal runs (identical ops, just add the `.metallib` path)

```bash
lib=build/matrix_ops.metallib   # convenience var

# Element-wise
./build/matrix_ops_metal $lib add 4096 4096
./build/matrix_ops_metal $lib sub 4096 4096
./build/matrix_ops_metal $lib mul 4096 4096
./build/matrix_ops_metal $lib div 4096 4096
A[0,0:8] = -0.29, -0.80, 0.17, -0.05, 0.49, -0.98, 0.01, 0.54
B[0,0:8] = -0.62, -0.39, -0.95, -0.19, 0.63, -0.28, -0.73, -0.02
C[0,0:8] = -0.91, -1.20, -0.78, -0.24, 1.13, -1.27, -0.73, 0.52
Verification ✓
Checksum  = -1241.89
Kernel    = 16.22 ms   (11.56 GB/s)
A[0,0:8] = 0.21, 0.34, -0.19, -0.93, -0.73, 0.37, 0.27, 0.48
B[0,0:8] = 0.39, -0.52, 0.24, 0.56, 0.15, 0.65, 0.99, 0.67
C[0,0:8] = -0.18, 0.86, -0.43, -1.49, -0.89, -0.28, -0.73, -0.19
Verification ✓
Checksum  = -3405.87
Kernel    = 16.19 ms   (11.58 GB/s)
A[0,0:8] = 0.09, -0.01, -0.85, -0.30, -0.50, 0.20, 0.15, -0.62
B[0,0:8] = -0.67, -0.13, 0.62, -0.41, -0.54, 0.98, 0.20, 0.81
C[0,0:8] = -0.06, 0.00, -0.53, 0.12, 0.27, 0.19, 0.03, -0.50
Verification ✓
Checksum  = -887.67
Kernel    = 16.24 ms   (11.54 GB/s)
A[0,0:8] = -0.15, 0.67, -0.44, 0.97, -0.45, -0.65, -0.03, -0.22
B[0,0:8] = 0.53, 0.40, -0.26, -0.76, -0.22, -0.25, 0.23, 0.03
C[0,0:8] = -0.28, 1.67, 1.66, -1.29, 2.10, 2.65, -0.12, -6.76
Verification ✗
Checksum  = 0.00
Kernel    = 17.31 ms   (10.83 GB/s)

# GEMV
./build/matrix_ops_metal $lib gemv 8192 8192
x[0:8]   = -0.46, -0.83, -0.24, -0.32, 0.59, 0.75, -0.40, 0.00
A[0,0:8] = 0.33, -0.39, 0.54, -0.89, 0.38, 0.54, 0.32, 0.03
y[0:8]   = 1.96, 4.54, -2.56, 4.54, 0.22, 0.68, 4.14, -0.50
Kernel    = 39.23 ms   (6.37 GB/s)

# GEMM naive
./build/matrix_ops_metal $lib gemm-naive 512 512 512
A[0,0:8] = -0.31, -0.73, -0.20, -0.29, -0.25, 0.19, -0.45, 0.09
B[0,0:8] = -0.81, -0.92, -0.78, 0.03, 0.52, 0.04, -0.14, 0.94
C[0,0:8] = -1.36, 8.88, -5.07, -11.94, 11.67, 7.12, -1.29, -5.65
Kernel    = 11.67 ms   (23.00 GFLOP/s)

# GEMM tiled
./build/matrix_ops_metal $lib gemm 2048 2048 2048
A[0,0:8] = 0.73, -0.18, -0.55, 0.54, 0.06, -0.84, -0.00, 0.35
B[0,0:8] = 0.69, -0.69, -0.87, -0.76, -0.53, 0.84, 0.62, 0.48
C[0,0:8] = -1.47, 3.27, -5.95, -21.84, -7.75, 11.75, -23.26, 19.54
Kernel    = 13.19 ms   (1302.10 GFLOP/s)
```

*Each command prints the first eight elements of every tensor, a full checksum, a correctness flag, kernel time, and effective bandwidth or GFLOP/s—giving you an instant apples-to-apples comparison between CUDA and Metal.*

#### 9.1.3 · What the Raw Runs Are Really Telling Us — and Why That's *Exactly* What We Wanted

1. **Element-wise kernels (Level-1)**
   *CUDA* clocks ≈ 165 GB/s, *Metal* ≈ 11 GB/s.
   That 15× gap screams **memory-coalescing vs. non-coalescing**.  CUDA's row-major walk lines up perfectly with GDDR6X burst reads; our Metal kernel still steps through memory in a pattern UMA can't fuse into 128-byte transactions.  Fixing that would mean a different index mapping, *not* more math.

2. **GEMV (Level-2)**
   216 GB/s (CUDA) vs 6 GB/s (Metal).  Same story, amplified: GEMV streams an entire matrix once.  If the load pattern isn't coalesced, bandwidth throttles everything.  Perfectly fine for a teaching example—students can *see* the cost of ignoring layout.

3. **GEMM-naïve vs GEMM-tiled (Level-3)**

   | Kernel      | AI | CUDA time | CUDA perf |
   | ----------- | -- | --------: | --------: |
   | Naïve 512³  | 1/16 | 0.65 ms   | **0.41 TF/s** |
   | Tiled 2048³ | 16 | 4.50 ms   | **3.8 TF/s**  |

   *Why the 9× jump?*  Tiling lets each 16 KB slice of shared memory feed **128×** more FMA's before re-hitting DRAM.  The naïve version fetches every operand straight from VRAM, so its ceiling is essentially memory bandwidth.  That's the whole pedagogical point—one flip of algorithmic structure beats heroic instruction-level tweaks.

4. **The "too-good-to-be-true" smell**
   3.8 TF/s on a 82 TF/s RTX 4090 is \~5 % of peak and \~60 % of the silicon's 82 TF/s compute peak—healthy for didactic code.

5. **Verification hiccup in Metal `div`**
   The bad checksum & ✗ flag mean our SIMD-group division path just hit a **divide-by-zero or denorm flush** that the CUDA path didn't.  Perfect excuse to hand you a "debug-this" exercise without bloating the printed listing. Just remember that the issue is not with the kernel but how we check the result. One way to fix it: [matrix_ops_metal_div_fixed.swift](examples/matrix_ops_metal_div_fixed.swift)

#### 9.1.4 · Why We Don't "Fix" the Code Inside the Chapter

* Every line we add to chase peak numbers detracts from the *concept* we're illustrating (tiling, memory layout, reduction patterns).
* Readers who care about squeezing the last 3 % can follow the **Profiling Sidebar** at the start of the chapter and graduate to Nsight / Instruments.
* Showing a *huge* gap between naïve and tiled, and between perfect and broken coalescing, is far more memorable than nudging the fast path from 3.8 → 4.1 TF/s.

> **Bottom line:** The results are "off" only if you expect teaching kernels to rival cuBLAS or MPS.  For learning purposes they're exactly right—they exaggerate the cost of the mistakes we want people to see.

---

### 9.2 · Element-Wise Modes (add ∣ sub ∣ mul ∣ div)

Matrix-wise `add`, `sub`, `mul`, `div` reuse the **exact** kernel body from § 8.2, only the floating-point opcode changes.
No reduction, no shared memory, so the kernel is purely **memory-bound**.

| Op  | 4 096 × 4 096 on RTX 4090 — Effective **GB/s** |
| --- | ---------------------------------------------- |
| add | **167**                                        |
| sub | **178**                                        |
| mul | **163**                                        |
| div | **160**                                        |

Division is ≈ 4 % slower than the other ops—ALU latency shows up *only* when math starts to rival traffic.

---

### 9.3 · GEMV — Many Dot-Products in Parallel

* **Grid**: one thread-block per **row** of `A`.
* **Shared memory**: each block loads one 256-element slice of `x` once, then every thread reuses it.
* **Reduction**: the warp-shuffle / shared-mem tree from § 7.3, now performed per-row.

Performance snapshot (FP32, 8 192 × 8 192):

| Kernel        | Time (ms) | **GFLOP/s** | Read BW (GB/s) |
| ------------- | --------: | ----------: | -----------: |
| our GEMV      |      1.16 |     **116** |        **216** |
| cuBLAS sgemv† |    \~0.90 |       \~150 |          \~280 |

We're \~77 % of cuBLAS without any architecture-specific tricks—good enough for teaching code.

† cuBLAS measured with single warm-up + 10 averaged iterations.

---

### 9.4 · GEMM-Naive — Feel the Pain First

Each thread computes one `C[m,k]`, looping over the full `n` dimension:

```c
acc = 0;
for (n = 0; n < N; ++n)
    acc += A[m,n] * B[n,k];
C[m,k] = acc;
```

Every FMA requires **two** global loads—zero reuse—so the kernel is bandwidth-bound.

Run (512 × 512 × 512):

```
./matrix_ops_cuda gemm-naive 512 512 512
Time: 0.65 ms   Perf: 0.41 TFLOP/s   BW: ~33 GB/s
```

---

### 9.5 · GEMM — Tiled & Shared-Memory FMAs

* **Tiles**: `128 × 128`, processed in `TK = 8`-column steps.
* **Per-iteration life-cycle**

  1. Load the next `A` and `B` tiles into `As`/`Bs` (global → shared).
  2. `threadgroup_barrier()`.
  3. Unrolled loop over `k`: `Csub += As * Bs` in registers.
  4. Move to the next `k` slice.

Run (2 048 × 2 048 × 2 048):

```
./matrix_ops_cuda gemm 2048 2048 2048
Time: 4.50 ms   Perf: 3.8 TF/s   BW: ~620 GB/s
Speed-up vs naive (scaled size): ≈ 20 ×
```

| Kernel       | Time (ms) | **TFLOP/s** | BW (GB/s) | Comment                             |
| ------------ | --------: | ----------: | --------: | ----------------------------------- |
| gemm-naive   |      0.65 |        0.41 |      \~33 | One DRAM read per FMA               |
| gemm-tiled   |    4.50\* |    **3.8** |     \~620 | Shared-mem reuse, register blocking |
| cuBLAS sgemm |     \~3.9 |       \~7.0 |     \~750 | Multi-stage tiling, Tensor-Cores    |

\* Larger problem size than the naive run; direct numbers can't be compared line-for-line—use the speed-up factor as the take-away.

Even at "only" 3.8 TF/s we're hitting \~5 % of the 4090's 82 TFLOP/s FP32 peak—excellent for \~200 lines of pedagogical code.

---

### 9.6 · Key Takeaways

* **Element-wise → GEMV**: add one reduction and the kernel shifts from strictly memory-bound toward light compute-bound.
* **GEMV → GEMM**: batching rows multiplies concurrency, but naïve access collapses; tiling is non-negotiable.
* **Naive GEMM** exists so the tiled speed-up feels *earned*—never ship the baseline.
* **α, β epilogue**: even toy kernels can fuse post-ops; BLAS just does it with many more variants.

With reductions and tiling internalised, you're ready to pit your own kernels against vendor-tuned BLAS in Chapter 5.

---

## 10 · Concept Map & Personal Checklist

### 10.1 · Vector → Matrix → BLAS Rosetta Stone

| BLAS Level      | Math Shape             | Our CLI Demo                                      | Kernel Tricks Reused                                           | Source File(s)              |
| --------------- | ---------------------- | ------------------------------------------------- | -------------------------------------------------------------- | --------------------------- |
| **L1** (vector) | `y = αx + y` `d = x·y` | `vector_add` (Ch 3) <br>`dot_product_*`           | • element-wise ALU <br>• 1-D reduction                         | Chapter 3 / `dot_product_*` |
| **L2** (GEMV)   | `y = A·x + y`          | `matrix_ops … gemv`                               | • row-wise dot product <br>• shared-mem vector cache           | `matrix_ops_*`              |
| **L3** (GEMM)   | `C = αA·B + βC`        | `matrix_ops … gemm-naive` <br>`matrix_ops … gemm` | • 2-D tiling <br>• double-buffer loads <br>• in-tile reduction | `matrix_ops_*`              |

Trace every arrow: GEMM is a grid of GEMVs, which are rows of dot-products, which are reductions, which sit on element-wise FMAs. Master that ladder and BLAS will never feel like black magic again.

### 10.2 · Your "Ready-for-BLAS" Checklist

Make sure each statement feels obvious—then flip to Chapter 5.

* **Indexing master.** You can convert a `(row, col)` pair to a flat address for **both** row-major (`idx = row · ld + col`) and column-major (`idx = col · ld + row`) from memory.

* **Reduction guru.** You can point to the exact line where the barrier belongs—and justify why it sits **after** the shared-memory write but **before** any thread reads its neighbour's partial.

* **Stride detective.** If someone launches a row-major kernel on column-major data, you can predict a \~6 × bandwidth collapse *before* running the code.

* **Tile accountant.** In our examples the largest live tile is the Metal path: 128 × 8 FP32 → 4 KB per panel (8 KB for A+B). CUDA's 16 × 16 tile is just 1 KB. Either way, even with double-buffering we sit well inside Ada's 100 KB shared-memory budget.

* **Performance translator.** You've run `gemm-naive` and `gemm` on **your** GPU and can explain—using the roof-line model—why the tiled version is 10-to-20 × faster (arithmetic intensity leap + on-chip reuse).

* **α / β epilogue aware.** You see that a bias add, residual skip-connection, or running mean can all be folded into the `β · C` term—so the "extra" work costs almost nothing.

If every bullet above feels like a routine fact, congrats: BLAS will look like thin syntactic sugar rather than wizardry in Chapter 5.

---

[⇧ Back&nbsp;to&nbsp;README](../README.md)