# Chapter 0 · Metal Primer Prologue — Setting Up a Compute-Only Metal Dev Environment

*This series is compute-only: no graphics, no shaders for rendering—just kernels that behave like CUDA `__global__` functions.*

*(Personal lab notebook - Last verified 2025‑05‑02)*

> 🎯 **Tested Rig**
>
> * macOS Sequoia 15.4.1 on a **80‑core Mac Studio M3 Ultra (512 GB)**
> * Terminal-only workflow (works in Cursor, iTerm, or plain Terminal)
> * Goal is headless **compute** kernels—no GUI rendering or Xcode projects required

This *Metal Guide* assumes you've already read the following mini-guides:

- [GPU Primer – Understanding GPUs for AI: CUDA vs Metal](../mini-guides/02-GPU-Primer.md)
- [The M-Series – How Apple Rewired Its Silicon Destiny](../mini-guides/03-The-M-Series-How-Apple-Rewired-Its-Silicon-Destiny.md)

*I highly recommend reading both the CUDA and Metal primers in parallel. This "parallel processing" approach (see what I did there?) will give you a more comprehensive understanding of GPU computing paradigms, even if you're primarily interested in just one platform. The comparative perspective highlights the core concepts that transcend specific APIs.*

---

## 1 · What **Metal Shading Language (MSL)** Really Is

Metal's compute language looks like "C++11 with attributes," but three extensions matter most:

| Surface-level Syntax / Keyword                                                      | What It Means in Practice                                                                                                              |
| ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `kernel` function qualifier                                                         | GPU entry point launched by CPU—CUDA `__global__` twin.                                          |
| **Attribute brackets** `[[thread_position_in_grid]]`, `[[threads_per_threadgroup]]` | Expose SIMT indices—Metal's `threadIdx`, `blockDim`.                                             |
| Address-space keywords `device`, `threadgroup`, `constant`                          | Map onto Apple GPU memory hierarchy (`device` = global, `threadgroup` = on-chip SRAM, `constant` = broadcast uniforms). |

Beyond the Metal-specific keywords and attributes, Metal Shading Language gives you access to standard C++11 features (like templates, classes, and STL algorithms). Additionally, Apple provides specialized extensions that are crucial for GPU programming:

1. **Packed vectors** (like `float4`, `half3`, etc.) – SIMD types that store several scalars in one register/memory slot. *Caveat:* on Apple GPUs prior to **A17/M4** any `half` vector is promoted to `float` internally, so size/perf gains appear only on the latest chips.

2. **Threadgroup barriers** - These are synchronization primitives (like `threadgroup_barrier()`) that ensure all threads in a threadgroup reach the same point before continuing execution. This is essential when threads need to share results through threadgroup memory, preventing race conditions where one thread might read data before another has finished writing it.

**Why "packed" vectors?**
The values are *packed together*—stored contiguously in one register or one 16-byte memory slot—so the hardware can move or operate on them as a single unit instead of four separate scalars. Think "one tightly wrapped package of 4 floats."

**Why the word "barrier"?**
It's a roadblock in the code: every thread must stop at the barrier and can't cross until all its companions arrive. Once everyone is lined up, the barrier lifts and all proceed together, guaranteeing no one races ahead and accesses half-written shared data.

The Metal Shading Language is deliberately designed to feel familiar to C++ programmers while exposing hardware-specific capabilities that enable high-performance GPU computing.

---

## 2 · Setting Up on macOS (M3 Ultra)

1. **Install full Xcode 15 or newer** from the Mac App Store.
   *Command-line tools alone omit the `metal` compiler binary.*
2. Accept the license and make Xcode's toolchain active:

   ```bash
   sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
   ```
3. **Verify the Metal toolchain**:

   ```bash
   xcrun -sdk macosx metal -v        # prints "Apple metal version …"
   xcrun -sdk macosx metallib -h     # help text means linker is visible
   ```

   Example output:

    ```
    Apple metal version 32023.619 (metalfe-32023.619)
    Target: air64-apple-darwin24.4.0
    Thread model: posix
    InstalledDir: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/metal/current/bin
    ```

    ```
    OVERVIEW: Metal Linker (and other random stuff)

    USAGE: metallib

    OPTIONS:
    --app-store-validate    Validate a MetalLib file meets necessary criteria for App Store
    ...
    ```

4. **Install Cursor or set up SSH connection (optional)** and enable *Remote-SSH* to this Mac if you prefer editing from another workstation; the PATH is already correct, so no extra setup is required.

    - [SSH Admin Guide](../guides/tools/ssh-admin.md)

We will assume a single-machine setup for the rest of this Metal primer series unless otherwise specified. This means you'll be developing and running Metal code on the same M3 Ultra system, rather than using a remote development workflow. This approach simplifies the setup process and eliminates potential networking complications while you're learning the fundamentals.

---

## 3 · Common Gotchas (C vets & Python newcomers)

| Habit                                          | Trip-Wire on Apple GPU                                                                                 | Quick Fix                                                                                                  |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| Assuming `printf` from GPU                     | There is **no** GPU `printf`; Metal kernels cannot write to stdout                                     | Debug with Xcode GPU Frame Capture or write results into a buffer the CPU reads back (see *hello_metal*). |
| Forgetting asynchronous command buffers        | `MTLCommandBuffer` executes later—CPU code races ahead                                                 | Call `commandBuffer.waitUntilCompleted()` before reading back results.                                     |
| Expecting unified virtual addresses ≈ CUDA UVM | CPU & GPU share **physical** DRAM, but you still map buffers explicitly and respect their storage mode | Use `.storageModeShared` for zero-copy host<→device buffers.                                               |
| Relying on garbage collection (Python folk)    | Metal objects are *reference-counted*; leaks persist across encoder loops                              | Wrap them in Swift/Objective-C ARC or `std::shared_ptr`.                                                   |
| Ignoring partial threadgroups                  | `dispatchThreadgroups` that don't cover the grid may leave "ragged edges"                              | Compute `threadgroupsPerGrid` with ceil-math just like CUDA grid/block arithmetic.                         |

    - [Apple Silicon Memory Copy Primer](04-Apple-Silicon-Memory-Copy.md)

---

## 4 · Quick **CUDA ↔ Metal** Cheat-Sheet

| CUDA concept             | Metal analog                                                     | Notes                              |
| ------------------------ | ---------------------------------------------------------------- | ---------------------------------- |
| `__global__`             | `kernel`                                                         | GPU entry point                    |
| `<<<grid, block>>>`      | `dispatchThreadgroups(grid, threadsPerThreadgroup)`              | Same two-level launch geometry     |
| `threadIdx` / `blockIdx` | `thread_position_in_threadgroup`, `threadgroup_position_in_grid` | SIMT indices                       |
| **Warp (32 threads)**    | **SIMD-group (32 threads)**                                      | Divergence & shuffle tricks align  |
| `__shared__` memory      | `threadgroup` memory                                             | On-chip SRAM for a threadgroup     |
| `__constant__` memory    | `constant` address space / argument buffer                       | For scalars & small LUTs           |
| `__syncthreads()`        | `threadgroup_barrier(mem_flags)`                                 | Local sync + optional memory fence |
| CUDA streams             | `MTLCommandBuffer` queues                                        | Async pipelines + events           |

---

## 5 · Hello Metal Example (command-line build)

### 5.1 · What We'll Do

*Each GPU thread writes its own indices into a buffer; the CPU prints them—our Metal stand-in for CUDA's `printf`.*

### 5.2 · Kernel (`hello.metal`)

[hello.metal](examples/hello.metal)
```cpp
#include <metal_stdlib>
using namespace metal;

kernel void hello(device uint2       *out    [[buffer(0)]],
                  uint  tid_in_tg    [[thread_index_in_threadgroup]],
                  uint  tg_id_in_g   [[threadgroup_position_in_grid]])
{
    const uint threadsPerTG = 4;                     // <-- constant
    uint gid = tg_id_in_g * threadsPerTG + tid_in_tg;
    out[gid] = uint2(tg_id_in_g, tid_in_tg);
}
```

*(For clarity, we hard-code threads_per_threadgroup in host code.)*

---

### 5.3 · Host driver in Swift — `hello_host.swift`

[hello_host.swift](examples/hello_host.swift)
```swift
import Metal
import Foundation

func main() throws {
    // ── 1 · Device & library ───────────────────────────────────
    guard let device = MTLCreateSystemDefaultDevice() else {
        fatalError("No Metal-capable GPU found")
    }
    let libURL   = URL(fileURLWithPath: "hello.metallib")
    let library  = try device.makeLibrary(URL: libURL)      // modern API
    let function = library.makeFunction(name: "hello")!
    let pipeline = try device.makeComputePipelineState(function: function)

    // ── 2 · Launch geometry ───────────────────────────────────
    let threadsPerTG  = 4
    let threadgroups  = 1
    let totalThreads  = threadsPerTG * threadgroups

    guard let outBuf = device.makeBuffer(length: totalThreads *
                                         MemoryLayout<SIMD2<UInt32>>.stride,
                                         options: .storageModeShared) else {
        fatalError("Buffer allocation failed")
    }

    // ── 3 · Encode & submit command buffer ────────────────────
    let queue          = device.makeCommandQueue()!
    let commandBuffer  = queue.makeCommandBuffer()!
    let encoder        = commandBuffer.makeComputeCommandEncoder()!

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(outBuf, offset: 0, index: 0)

    let tgSize  = MTLSize(width: threadsPerTG, height: 1, depth: 1)
    let grid    = MTLSize(width: threadgroups, height: 1, depth: 1)
    encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: tgSize)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    // ── 4 · Read results back on the CPU ──────────────────────
    let data = outBuf.contents()
                     .bindMemory(to: SIMD2<UInt32>.self, capacity: totalThreads)

    for i in 0..<totalThreads {
        print("Hello from threadgroup \(data[i].x), thread \(data[i].y)")
    }
}

do { try main() }
catch { print("Error: \(error)"); exit(1) }
```

---

### 5.4 · Build & Run (Terminal-only)

```bash
# 1. Compile the kernel to a Metal library
xcrun -sdk macosx metal -c hello.metal -o build/hello.air
xcrun -sdk macosx metallib build/hello.air -o build/hello.metallib

# 2. Compile + link the Swift host
xcrun -sdk macosx swiftc hello_host.swift \
      -framework Metal -framework Foundation -o build/hello_host

# 3. Execute (run from inside the build directory so the `.metallib` is discoverable)
cd build
./hello_host
```

*Path caveat:* the Swift host loads `hello.metallib` from **its current working directory**. Running the binary from inside `build/` keeps the executable and its library side-by-side. Prefer to stay in the repo root? Adjust the Swift code to `device.makeLibrary(URL: URL(fileURLWithPath: "build/hello.metallib"))` or copy the `.metallib` up one level.

### Expected Output

```
Hello from threadgroup 0, thread 0
Hello from threadgroup 0, thread 1
Hello from threadgroup 0, thread 2
Hello from threadgroup 0, thread 3
```

Exactly the same four lines you saw with CUDA's `printf`—proof that the Swift-only Metal stack is wired up and ready for heavier AI kernels.

---

### Why the Metal build looks "heavier" than the one-line CUDA compile

| Stage               | CUDA (`nvcc`)                                                                                                                 | Metal (`metal` → `metallib`)                                                                                    | What's really happening                                                                                                                       |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Front-end**       | `nvcc` parses **host *and* device** code in a single `.cu` file.                                                              | We run the ***Metal compiler only*** on the `.metal` source. Host Swift is compiled by `swiftc` (or Clang).     | Metal keeps the GPU shader tool-chain completely separate from the CPU language tool-chain.                                                   |
| **Intermediate**    | `nvcc` emits PTX/SASS and immediately bundles them into the final ELF/PE executable, hiding all the steps behind one command. | `metal -c` produces an **AIR** (Apple Intermediate Representation) object.                                      | Apple exposes this IR on purpose so you can archive, sign, or LTO-optimize it later.
| **Link**            | Same `nvcc` call links host objects + GPU cubins → a single executable.                                                       | `metallib` links one or more AIR files into a **`.metallib** GPU library.                                      | Metal lets you preload or hot-swap these libraries at runtime, and reuse them across multiple pipeline-state objects.                         |
| **Load at runtime** | Executable already contains the cubin; CUdriver patches if needed.                                                            | Host code calls `device.makeLibrary(URL:)` to load the `.metallib` and then builds a `MTLComputePipelineState`. | That late binding is what allows pre-compiled pipelines to be cached and shared across apps or shipped as updates without relinking the host. |

- PTX/SASS: PTX (Parallel Thread Execution) is NVIDIA's intermediate representation for GPU code, while SASS (Streaming Assembler) is the final machine code for NVIDIA GPUs. PTX provides a stable ISA that gets translated to hardware-specific SASS at runtime.
- ISA: Instruction Set Architecture - the set of instructions that a CPU can execute.
- ELF/PE: Standard executable file formats - ELF (Executable and Linkable Format) used on Linux/Unix systems, and PE (Portable Executable) used on Windows. CUDA embeds compiled GPU code within these CPU executable formats.
- LTO-optimize: Link Time Optimization - a compiler technique that performs optimization across multiple compilation units during the linking phase. For GPU code, this allows optimizations that span across kernel boundaries, potentially improving performance by eliminating redundant operations.

#### So, are the extra commands *really* mandatory?

* **Two-step flow (`metal -c` → `metallib`)** — What we showed. It's explicit, mirrors Apple's docs, and matches how serious projects cache and sign AIR objects before shipping.
* **One-step shortcut** — You *can* do:

  ```bash
  xcrun metal hello.metal -o hello.metallib
  ```

  …and skip the AIR file entirely. We spelled out the longer path so readers see where the pieces live and can reuse the AIR later for PSO caching or offline LTO.

#### Why NVIDIA hides the plumbing but Apple doesn't

* **CUDA** dates to 2006 when GPUs weren't expected to hot-reload code. Bundling PTX inside the executable was the simplest user model.
* **Metal** (2014-) added pipeline caching, dynamic libraries, binary archives, and code-signing requirements across iOS/tvOS/macOS. Exposing the separate IR + link step lets build systems cache and notarize GPU binaries the same way they treat CPU dylibs.  

In day-to-day hacking you can use the one-liner, but knowing the explicit two-step tool-chain will save you headaches when you:

* split kernels across multiple files,
* LTO-optimize or `-Osize` your AIR,
* ship precompiled PSOs to every node in your AI cluster.

- PSO: Pipeline State Object - a compiled representation of a compute pipeline that includes the compiled shader code, descriptor settings, and other configuration options.

That's why our Primer demonstrated the "long route" first—even though Metal can be just as concise as the CUDA command when you need it to be.

---

### One-liner (copy-paste into Terminal)

If you just want to **compile the kernel, build the Swift host, and launch the program in one shot**, chain the three commands with `&&` so the chain aborts on the first failure (mimics `set -e`).

```bash
set -e  # optional: stop script on first error
xcrun -sdk macosx metal hello.metal -o hello.metallib \
&& xcrun -sdk macosx swiftc hello_host.swift -framework Metal -framework Foundation -o hello_host \
&& ./hello_host
```

* What happens under the hood

  1. `metal … -o hello.metallib` ⟶ compiles **and** links the GPU code in a single pass (skips the `.air` intermediate).
  2. `swiftc … -o hello_host`   ⟶ builds the Swift driver that loads the library.
  3. `./hello_host`           ⟶ runs it; you'll see the four "Hello" lines if everything worked.

### Slightly cleaner: a tiny Makefile

[Makefile](examples/Makefile)
```make
# Makefile
METAL   = xcrun -sdk macosx metal
SWIFTC  = xcrun -sdk macosx swiftc

all: hello_host

hello.metallib: hello.metal
	$(METAL) $< -o $@

hello_host: hello_host.swift hello.metallib
	$(SWIFTC) $< -framework Metal -framework Foundation -o $@

run: hello_host
	./hello_host

clean:
	rm -f hello.metallib hello_host
```

```bash
make run     # builds everything (only if sources changed) and executes
make clean   # removes artifacts
```

### Why the CUDA-like "single command" is OK here

* The Metal front-end can both **compile** and **link** when you pass a `.metal` file directly to `metal -o foo.metallib`; no `.air` is required unless you need LTO or signing.
* The Swift host doesn't care **how** the `.metallib` was produced—only that the file exists at the path you hand to `device.makeLibrary(URL:)`.

For day-to-day experiments (especially in a cluster where you might rsync the directory and run remotely) the one-liner or Makefile keeps the workflow as terse as `nvcc hello.cu -o hello`. If you later need the full two-step AIR process—for pipeline caching or notarization—you can always expand the build back out.

---

### 5.5 · "Python-style" Pseudocode (*conceptual* mirror)

```python
def hello_kernel(tg_id, thread_id):
    buffer[ tg_id * THREADS_PER_TG + thread_id ] = (tg_id, thread_id)

THREADGROUPS = 1
THREADS_PER_TG = 4

for tg in range(THREADGROUPS):          # GPU launches all of this in parallel
    for t in range(THREADS_PER_TG):
        hello_kernel(tg, t)

# CPU prints after "command buffer" completes
for t in range(THREADS_PER_TG):
    print(f"Hello from threadgroup 0, thread {t}")
```

Same mapping table as the CUDA pseudocode—just substitute `threadgroup` for `block`.

---

## 6 · Fast-Path Troubleshooting

| Symptom                                                | Likely Cause                                              | Fix                                                                                           |
| ------------------------------------------------------ | --------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `metal: command not found`                             | Full Xcode not installed or `xcode-select` pointed at CLT | `xcode-select -s /Applications/Xcode.app/...`                                                 |
| "function not found" at runtime                        | Forgot to embed or path the `.metallib`                   | Use `[dev newDefaultLibraryWithFile:@"hello.metallib" error:nil]` or embed via `-sectcreate`. |
| Host prints zeros                                      | Command buffer committed but not waited for               | Add `waitUntilCompleted()` before reading the buffer.                                         |
| Build succeeds locally but fails via Cursor Remote-SSH | Remote shell lacks Xcode path                             | Export `DEVELOPER_DIR` or run `xcode-select` on the remote Mac.                               |
| One-liner fails halfway                               | Any step in the `&&` chain returned non-zero              | Check which sub-command failed; `&&` aborts the chain on first error.                         |

---

[⇧ Back&nbsp;to&nbsp;README](../README.md)