# CUDA & Metal Primer

## Scope & Learning Lens

This primer is first and foremost a *reading-comprehension workout* for AI-server administrators. Every code snippet you encounter is meant to sharpen your ability to **read and reason about GPU-side code** (CUDA on NVIDIA, Metal on Apple)—that's it. We are **not** trying to turn you into a production-grade AI kernel author.

The ultimate objective is to help you become a more confident, effective **AI server operator / administrator** who can:

- audit and troubleshoot GPU workloads
- understand the performance implications of model-server choices
- speak the language of GPU compute well enough to hold your own in engineering conversations

### Benchmarking Disclaimer

Any performance numbers in any example are *illustrative only*. Hardware and driver stacks differ wildly—treat the printed metrics as intuition pumps, **not** definitive benchmarks. For trustworthy data use vendor profilers (Nsight, Instruments, etc.) and a rigorous methodology (warm-ups, multiple runs, GPU timers).

---

## Testing Rigs

### macOS (Apple Silicon)

- **SoC**  Apple M3 Ultra — 32-core CPU / 80-core GPU
- **Memory**  512 GB unified
- **OS**  macOS 15.x (Sequoia)
- **Toolchain**  Xcode 16.x · Swift 6.x · Metal 3 · Accelerate ILP-64

### Windows (NVIDIA)

- **CPU**  AMD Ryzen 9 7950X3D (16C/32T)
- **GPU**  NVIDIA GeForce RTX 4090 24 GB (Ada Lovelace)
- **Memory**  128 GB DDR5-6000
- **OS**  Windows 11 Pro (25H2)
- **Toolchain**  CUDA 12.9.x · Nsight 2025.x · Visual Studio 2025

---

## Roadmap

1. [CUDA Primer Prologue — Setting Up Your Dev Environment](00-CUDA-Primer-Prologue.md)
2. [Metal Primer Prologue — Setting Up a Compute-Only Metal Dev Environment](00-Metal-Primer-Prologue.md)
3. [Chapter 1 · Linear Algebra for AI in a Nutshell — A Lightning Refresher](01-linear-algebra-for-AI-in-a-nutshell.md)
4. [Chapter 2 · Hello GPU — Reinforcing the Mental Model of GPU Computation](02-hello-gpu-reinforced.md)
5. [Chapter 3 · Vectors — From "Hello, GPU" to First Real Math](03-vector-refresher.md)
6. [Chapter 4 · Matrices — From Vectors to Linear Transformations](04-matrix-refresher.md)
7. [Chapter 5 · BLAS 101 — The Library Layer Every GPU Kernel Falls Back On](05-blas-101.md)
8. [Chapter 6 · BLAS Deep Dive — How Vendor Kernels Hit 90 % of Peak](06-blas-deep-dive.md)