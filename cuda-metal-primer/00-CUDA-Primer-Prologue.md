# Chapter 0 · CUDA Primer Prologue — Setting Up Your Dev Environment

*(Personal lab notebook - Last verified 2025‑05-02)*

This guide shows you how to set up a CUDA workstation on Windows (RTX 4090) and reach it from macOS via SSH/Cursor.

This *CUDA Guide* assumes you've already read the following mini-guides:

- [GPU Primer – Understanding GPUs for AI: CUDA vs Metal](../mini-guides/02-GPU-Primer.md)
- [CUDA & Ada: How Dedicated GPUs Rewired Parallel Computing](../mini-guides/05-NVidia-CUDA-GPUs.md)

*I highly recommend reading both the CUDA and Metal primers in parallel. This "parallel processing" approach (see what I did there?) will give you a more comprehensive understanding of GPU computing paradigms, even if you're primarily interested in just one platform. The comparative perspective highlights the core concepts that transcend specific APIs.*

Note: Remote SSH connections from Cursor or WindSurf may encounter issues with default configurations. In my testing, configuring WSL as the default shell in Windows resolved these connection problems. Results may vary depending on your specific setup.

[How to set WSL as the default shell when SSHing into a Windows system](../guides/troubleshooting/05-default-wsl-shell.md)

> 🎯 **Tested Rig**
>
> * Windows 11 Pro, AMZ Ryzen 9 7950X3D 16-Core, RTX 4090, CUDA 12.9 

## 1 · What *CUDA C++* Really Is  
At first glance it looks like plain C/C++, but three key extensions change the game:

| Surface-level Syntax | What It Means in Practice |
|----------------------|---------------------------|
| `__global__`, `__device__`, `__host__` **function qualifiers** | Decide **where** the function runs and **who** may call it. A `__global__` *kernel* runs on the GPU but is launched by the CPU. |
| The triple-angle launch `kernel<<<grid, block>>>(args);` | Compile-time metadata that the runtime converts into a command-buffer (how many threads, which stream, etc.). |
| Built-ins like `threadIdx.x`, `blockDim.x` | The compiler injects register reads that expose the SIMT thread's coordinates—no equivalent exists in vanilla C/C++. |

Otherwise you inherit the full C++17 tool-chain: lambdas, templates, `<algorithm>`, even `std::vector` on the host side. The twist is that device code is funneled through **NVCC**, which splits your translation unit, compiles GPU sections with PTX/SASS back-ends, then relinks everything into a single binary.  ([1. Introduction — NVIDIA CUDA Compiler Driver 12.9 documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/?utm_source=chatgpt.com))

---

## 2 · Setting Up on a Windows + RTX 4090 Workstation  

1. **Install the latest Studio/Game-Ready Driver** that matches CUDA 12.9.  
2. **Grab "CUDA Toolkit 12.9"** from NVIDIA's site and run the installer. Pick "Express (Visual Studio Integration)."  ([CUDA Toolkit Archive - NVIDIA Developer](https://developer.nvidia.com/cuda-toolkit-archive?utm_source=chatgpt.com), [1. Introduction — Installation Guide Windows 12.9 documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html?utm_source=chatgpt.com))  
3. **Verify NVCC**: open "*Developer Command Prompt for VS 2022*" and run  
   ```bash
   nvcc --version     # should print "release 12.9, V12.9.xx"
   ```  
4. **Test 'hello_cuda.cu' example provided below**:  
   Run the hello-cuda sample below to verify everything instead of the now-removed NVIDIA samples: [hello_cuda.cu](examples/hello_cuda.cu) 

---

## 3 · Common Gotchas (Old-School C Devs vs. Python-First Folks)

| Habit | Trip-Wire on the GPU | Quick Fix |
|-------|---------------------|-----------|
| C pros forget that host and device pointers occupy separate address spaces | Passing a host pointer to a kernel silently produces garbage or a segmentation fault | Treat every pointer like it wears a color-coded wrist-band: *green* (host) vs. *violet* (device). |
| Measuring time with `std::chrono` right after a launch | Kernels are **asynchronous**; the CPU races ahead | Wrap timing with `cudaEventRecord / Synchronize` or add `cudaDeviceSynchronize()` before you stop the clock. |
| Assuming `malloc`/`free` semantics | GPU memory lives behind `cudaMalloc`/`cudaFree` and is *not* reclaimed at program exit on some drivers | RAII wrappers (`thrust::device_vector`, `std::unique_ptr` + custom deleter) keep leaks away. |
| Python-only newcomers expect *garbage collection* & no manual copies | There is no automatic move of `numpy` arrays to the GPU | Use **cuPy** or PyTorch tensors when you want Python ergonomics; under the hood they still call `cudaMalloc` for you. |
| Ignoring error codes | Every CUDA API returns `cudaError_t`; failure to check can leave the GPU in a wedged state | Append `cudaGetLastError()` or use the `NVCC` flag `-lineinfo` so Nsight Compute pinpoints the faulting PC. |

--- 

## 4 · Hello CUDA Example

[hello_cuda.cu](examples/hello_cuda.cu)

```cpp
#include <cstdio>
#include <cuda_runtime.h>

// ── GPU kernel ───────────────────────────────────────────
// Prints its grid & thread coordinates
__global__ void say_hello()
{
    printf("Hello from block %d, thread %d\n",
           blockIdx.x, threadIdx.x);
}

int main()
{
    // Launch 1 block with 4 threads
    say_hello<<<1, 4>>>();
    cudaDeviceSynchronize();          // wait for GPU printf to finish
    return 0;
}
```

*What is happening here?*

1. We define a kernel function `say_hello` that prints a message to the console.
   - The `__global__` keyword marks this as a function that runs on the GPU but can be called from the CPU
   - Inside the kernel, we access built-in variables `blockIdx.x` and `threadIdx.x` to identify which thread is executing

2. We define a main function that launches the kernel.
   - This runs on the CPU (host) and coordinates the GPU execution

3. We launch the kernel with 1 block and 4 threads using the special CUDA syntax `<<<1, 4>>>`.
   - The triple angle brackets are CUDA's way of specifying execution configuration
   - First parameter (1): Number of blocks in the grid
   - Second parameter (4): Number of threads per block
   - This creates a total of 1×4 = 4 parallel threads on the GPU

4. We wait for the kernel to finish with `cudaDeviceSynchronize()`.
   - CUDA kernel launches are asynchronous - the CPU continues execution immediately
   - Without this synchronization, the program might exit before the GPU finishes printing

5. We return 0 to indicate successful program completion.

### "Python-style" Pseudocode - *Conceptual* Comparison

*(mirrors the behavior of **hello_cuda.cu** without real GPU syntax)*

```python
# ── 1 · Define what *each GPU thread* will do ────────────────────
def say_hello_kernel(block_idx: int, thread_idx: int) -> None:
    """
    Pretend this function runs on a single GPU thread.
    In CUDA we'd mark it __global__ and it would execute in parallel.
    """
    print(f"Hello from block {block_idx}, thread {thread_idx}")

# ── 2 · Decide launch geometry (grid & block dimensions) ─────────
BLOCKS = 1               # gridDim.x  in CUDA
THREADS_PER_BLOCK = 4    # blockDim.x in CUDA

# ── 3 · "Launch" the kernel for every thread ---------------------
# In real CUDA, this happens inside the GPU driver.
# Here we emulate it with nested loops (runs serially, one CPU thread).
for block in range(BLOCKS):                     # grid loop
    for thread in range(THREADS_PER_BLOCK):     # block loop
        # In CUDA this call would be *asynchronous* and parallel.
        say_hello_kernel(block, thread)

# ── 4 · Synchronize ------------------------------------------------
# CUDA needs an explicit cudaDeviceSynchronize() if you launch
# asynchronously and then use the output.  Our loop is synchronous
# already, so there's nothing to do here.
```

#### How this maps to real CUDA concepts

| Pseudocode piece            | Real CUDA counterpart                        | Mental takeaway for Python coders                                                                   |
| --------------------------- | -------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `say_hello_kernel()`        | `__global__ void say_hello()`                | A function that *cannot* be called from other GPU code directly; it's triggered by the host.        |
| Nested `for` loops          | Hardware **grid → block → thread** hierarchy | CUDA launches *all* iterations at once; our loop is just a sequential sketch.                       |
| `print()` inside the kernel | `printf()` inside the GPU kernel             | CUDA's `printf` buffers per-thread messages until the kernel finishes, then flushes them to stdout. |
| No explicit data movement   | Same in hello_cuda.cu                       | Because the kernel only prints, there's no memory copy.                                |
| "Synchronize" comment       | `cudaDeviceSynchronize()`                    | In real code you must wait for the GPU if you need deterministic ordering or timing.                |

**Why Python folks should care:**
Think of the CUDA launch as `multiprocessing` on steroids: instead of one function running on a few CPU cores, *thousands* of tiny threads run the same body (`say_hello_kernel`) with different indices provided by the grid/block hardware. Once you grok that mapping, moving on to data-parallel math kernels is just replacing the `print` with arithmetic on arrays you've copied to the GPU.

### Build & Run

```bash
nvcc hello_cuda.cu -o build/hello_cuda
./build/hello_cuda
```

*Why the `build/` prefix?*  We keep all binaries in a single, `.gitignore`-ed directory so your Git history doesn't balloon with platform-specific executables. Drop the prefix if you'd rather keep outputs next to sources.

### Expected Output

```
Hello from block 0, thread 0
Hello from block 0, thread 1
Hello from block 0, thread 2
Hello from block 0, thread 3
```

> If you see all four lines—one per GPU thread—your compiler, runtime, and driver stack are wired up correctly on that side of the OS. You're now clear to move on to the meatier examples.

If you encounter the following warning: 

```bash
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
```

**Why this appears**

When you compile CUDA code, `nvcc` automatically creates code for many older GPU models, not just your current one.
NVIDIA plans to stop supporting very old GPUs (before compute capability 7.5) in future versions.
The warning is just telling you that your compiler is still including code for these older GPUs in your program.

---

**Two quick ways to silence it**

| What to do                                    | Command                                                        | Effect                                                                 |
| --------------------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **Compile only for your GPU** (best practice) | `nvcc -arch=sm_89 hello_cuda.cu -o hello_cuda`                 | Generates code solely for SM 8.9 → no deprecated targets → no warning. |
| **Keep all defaults but hide the message**    | `nvcc hello_cuda.cu -o hello_cuda -Wno-deprecated-gpu-targets` | Still emits old CUBINs, just mutes the warning.                        |
| **Compile only for your GPU** (best practice) | `nvcc -arch=sm_89 hello_cuda.cu -o build/hello_cuda`                 | Generates code solely for SM 8.9 → no deprecated targets → no warning. |
| **Keep all defaults but hide the message**    | `nvcc hello_cuda.cu -o build/hello_cuda -Wno-deprecated-gpu-targets` | Still emits old CUBINs, just mutes the warning.                        |

Pick the first approach for smaller binaries and faster compiles; use the second if you truly need those older GPU targets in the same executable.

`sm_89` is shorthand for **"Streaming Multiprocessor version 8.9."**

| Part          | Meaning                                                                                                                                                                   |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SM**        | *Streaming Multiprocessor* – the fundamental execution unit inside every NVIDIA GPU. Compiling "for an SM" means targeting its exact instruction set and register layout. |
| **8** (major) | Architecture generation. **8 × = the Ampere / Ada family** of GPUs.                                                                                                       |
| **9** (minor) | Specific implementation within that generation. Higher numbers typically mean newer/more advanced variants with additional features.                                       |

---

## 5 · Setting up CUDA in WSL

If you're using Windows natively with PowerShell, you can skip this section. However, if you prefer a Linux-like environment or are coming from macOS, setting up WSL (Windows Subsystem for Linux) provides a more familiar command-line experience and better integration with development tools like Cursor. 

[How to set WSL as the default shell when SSHing into a Windows system](../guides/troubleshooting/05-default-wsl-shell.md)

Assuming you've either set WSL as your default shell manually or followed the guide above, the next step is setting up CUDA within WSL. The following sections will help you troubleshoot common issues and get your CUDA environment properly configured.

### 1 · Why `nvcc` is Missing

The Windows installer you ran placed all user-mode CUDA files under
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\…` and wired them into Windows' `PATH`.

Inside WSL you get a *separate* root filesystem, so Linux can't see those binaries unless you mount and export them manually; the usual—and cleaner—solution is to install the **Linux-side toolkit packages** that contain `nvcc`.

---

### 2 · One-Minute Checklist

1. **GPU driver**: the Studio/Game-Ready driver you installed already contains the WSL kernel modules—no second driver goes inside WSL.
2. **WSL version**:

   ```powershell
   wsl -l -v          # column "VERSION" must read 2
   wsl --update       # pulls the latest kernel & GPU patches
   wsl --shutdown     # restart the subsystem so the new kernel loads
   ```
3. **Distribution**: commands below assume Ubuntu 22.04 or 24.04. Adjust the repo URL if you run another distro.

---

### 3 · Install the 12.9 Toolkit Inside WSL

```bash
# ----- inside your WSL shell -----
sudo apt update
sudo apt install build-essential gnupg curl -y

# 1) Add NVIDIA's package key & repo for CUDA 12.9
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
  | sudo tee /etc/apt/keyrings/nvidia.asc > /dev/null

echo "deb [signed-by=/etc/apt/keyrings/nvidia.asc] \
https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" \
| sudo tee /etc/apt/sources.list.d/cuda.list

# 2) Pull the user-mode toolkit (no driver)
sudo apt update
sudo apt install cuda-toolkit-12-9 -y      # grabs nvcc, libraries, samples

# 3) Wire PATH & LD_LIBRARY_PATH
echo 'export PATH=/usr/local/cuda-12.9/bin:$PATH' | sudo tee /etc/profile.d/cuda.sh
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH' \
  | sudo tee -a /etc/profile.d/cuda.sh
source /etc/profile.d/cuda.sh
```

*If `cuda-toolkit-12-9` isn't available yet for your distro, install the nearest minor release (e.g., `12-8`) or use the runfile installer. The driverless **"toolkit only"** packages are explicitly supported for WSL.* 

---

### 4 · Smoke Test

```bash
nvcc --version          # should print "Cuda compilation tools, release 12.9"
```

Nvidia stopped shipping samples with the toolkit. Just run the hello cuda example above to verify your setup.

---

### Common Pitfall Cheatsheet

| Symptom                                            | Fix                                                                                              |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `nvcc: command not found` even after install       | `source /etc/profile.d/cuda.sh` or reopen the shell so PATH updates take effect.                 |
| `cuda.h: No such file or directory` when compiling | Add `-I/usr/local/cuda/include` to your compile flags (or let CMake's `FindCUDAToolkit` do it).  |

**Troubleshooting nvcc not found**

To address the "nvcc command not found" error, check whether your driver and toolkit are correctly installed and ensure that `nvcc` is included in the system PATH. The solution might vary between systems like Windows and WSL. Here's a guide:

* Verify driver, CUDA toolkit.
* Ensure PATH is set in relevant files (`.profile`, `.bashrc`, `.zshrc`).
* Test execution with `hello_cuda.cu`.
* Consider differences between plain SSH, remote environments like Cursor, and login/non-login shells.

**Why `$PATH` sometimes "forgets" CUDA**

* **Login vs. non-login shells**  Files like `~/.profile` load only for login sessions; IDE terminals may spawn non-login shells.
* **Shell flavor**  `bash` reads `~/.bashrc`, `zsh` reads `~/.zshrc` *and* (on Ubuntu) chains through `/etc/profile → ~/.profile`.
* **System scripts overriding user files**  Some distros reset `PATH` late in `/etc/profile`, clobbering early exports in `~/.zshenv`. Putting your export **last** (`~/.profile`, `~/.zprofile`, or `/etc/profile.d`) wins every time.

```
export PATH=/usr/local/cuda-12.9/bin:$PATH
```

Remember: after editing environment files, open a **fresh** terminal or run `exec $SHELL -l` so the changes take effect.

---

## 6 · Quick CUDA <-> Metal cheat-sheet

| CUDA keyword / concept            | Rough Metal analog                                               | Why it matters later                                                 |
| --------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------- |
| `__global__` qualifier            | `kernel` function qualifier                                      | Same "runs on GPU, launched by CPU" contract.                        |
| `<<<grid, block>>>` launch config | `dispatchThreadgroups` (grid) + `threadsPerThreadgroup` (block)  | One-to-one mapping for launch geometry.                              |
| `threadIdx.x`, `blockIdx.x`       | `thread_position_in_threadgroup`, `threadgroup_position_in_grid` | Index math carries over unchanged.                                   |
| **Warp (32 threads)**             | **SIMD-group (32 threads on Apple GPU)**                         | Divergence & shuffle tricks translate almost directly.               |
| `__shared__` memory               | `threadgroup` memory                                             | Fast on-chip SRAM scoped to a threadgroup/block.                     |
| `__constant__` memory             | `constant` address space / argument buffer                       | Broadcast read-only scalars.                |
| `__syncthreads()`                 | `threadgroup_barrier(mem_flags)`                                 | Block-level sync; required before reading data another thread wrote. |
| `cudaMemcpyAsync + streams`       | `MTLCommandBuffer` + `blitEncoder`                               | Overlapping copies & compute for pipeline hiding.                    |

[⇧ Back&nbsp;to&nbsp;README](../README.md)