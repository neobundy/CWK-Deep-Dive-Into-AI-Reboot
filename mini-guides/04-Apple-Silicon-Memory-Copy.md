# Apple-Silicon Memory & Copy Primer  

*(Personal lab notebook — Last verified 2025‑05‑01)*

For deep dive Metal examples, refer back to Chapter 04:

[Local AI Infrastructure: Assessment, Implementation, and Strategic Roadmap](../guides/04-Introduction.md)

---

Let's have a bird's-eye view of what's happening here.

---

## 1. Host vs. GPU in a "Unified" World  

* **Host** = CPU side where macOS apps run.  
* **GPU** = Apple-GPU cores + their private caches.  
* **Unified Memory** means both sides *address* the same LPDDR5 pool, but the **path** they take (and the cache rules they follow) still matters.

---

## 2. Storage Modes—Your Three Doors  

* **`storageModeShared`**  
  * CPU-coherent. CPU and GPU see the same bytes.  
  * Coherency fabric keeps caches in sync → extra latency & bandwidth tax.  
  * Use for data the CPU must touch after the GPU writes (e.g., readbacks, dynamic buffers).  
* **`storageModePrivate`**  
  * GPU-only. CPU can't peek.  
  * No coherency overhead; full GPU bandwidth.  
  * Best for big weights, textures, anything the CPU doesn't need mid-frame.  
* **`storageModeMemoryless`**  
  * Tile memory for render passes (textures only). Lives entirely on-chip, vanishes after use.  
  * Great for depth buffers or transient G-buffers.

*What does coherent mean?* : Coherency ensures that when data is modified in one place (like CPU cache), all other copies of that data (like in GPU cache) are either updated or invalidated. In a coherent system, the CPU and GPU always see the same version of data, which prevents race conditions but adds overhead as the system must track and synchronize all copies across different caches and memory locations.

---

## 3. The SoC Fabric (a.k.a. Interconnect)  

* Think of it as the on-chip "highway" plus a built-in traffic cop.  
* Links CPU clusters, GPU, Neural Engine, SLC, and memory controllers.  
* Runs a **coherency protocol**: if one core writes a cache line, all stale copies elsewhere are invalidated automatically.  
* Fabric traffic = **on-chip**; it's separate from the LPDDR5 pins that go off-chip.

---

## 4. Blit Engines—Bulk Movers With Rules  

* **Blit** = BLT = *Bit-Block Transfer* → pronounced "blit."  
* Dedicated DMA hardware reachable via `MTLBlitCommandEncoder`.  
* Perfect for copies/fills/mipmaps, but:  
  * Limited number of lanes.  
  * Shares the fabric with everything else.  
  * On chips like M3 Ultra, a single blit runs on **one die at a time**, so bandwidth caps around 300 GB/s write-side.

The term "BLit" dates back to early graphics systems that moved rectangular blocks of bits (pixels) from one place in memory to another—often to scroll screens or copy sprites. Over time "blit" became shorthand for any bulk copy or fill operation handled by dedicated hardware.

"Blit" is a **spoken nickname for the original acronym** BLT — Block Transfer. Programmers needed a pronounceable term, so "BLT" became vocalized as "blit." The added 'i' makes it easier to say: BLT → blit (similar to how "SQL" becomes "sequel").

By the time Unix systems and early graphics workstations adopted the concept, "blit" was already the standard term.

Modern graphics APIs like Metal preserve this tradition with names like BlitCommandEncoder and BlitPass.

This historical evolution explains why we use "blit" today—it began as BLT (Block Transfer) and evolved into a pronounceable term that became industry standard.

In Metal you see this legacy in MTLBlitCommandEncoder, whose jobs are exactly those classic blit tasks: copy buffers or textures, fill regions with a value, generate mipmaps, etc.

---

## 5. Two Benchmarks, Two Paths

You already saw these in the main guide linked above.

Source links: [Blit copy `bandwidth.swift`](../guides/src/bandwidth.swift) · [GPU memcpy `dram_bandwidth.swift`](../guides/src/dram_bandwidth.swift)

```bash
# 1. CPU → GPU blit  (prints writes only)
swiftc bandwidth.swift -framework Metal -o bandwidth
./bandwidth 8 10      # 8 GiB, 10 s window

# 2. Pure-GPU memcpy (prints writes only)
swiftc gpu_memcpy_bandwidth.swift -framework Metal -o dram_bandwidth
./dram_bandwidth 8 10
```


* **`bandwidth.swift`**  
  * Shared → Private blit copy.  
  * Measures **host ↔ GPU transfer** ceiling (fabric + DMA lanes).  
  * M3 Ultra: ≈ 300 GB/s write-side ≈ 600 GB/s total.  
* **`dram_bandwidth.swift`**  
  * Private → Private compute copy.  
  * Measures **pure GPU DRAM bandwidth** per die.  
  * M3 Ultra: ≈ 300–350 GB/s write-side (per die) ≈ 600–700 GB/s total.

*Rule of thumb:* Real code rarely tops 60 % of Apple's headline 819 GB/s; two-thirds is already stellar.

---

## 6. When to Choose Which Path—Metal Decisions from Host Code

The concepts—`storageModeShared`, `storageModePrivate`, `storageModeMemoryless`, and "Blit"—are **Metal-specific API options**. You use them *from Swift* (or Objective-C) when you create Metal buffers or textures, but they're not part of the Swift language itself.

```swift
// 1) Stream data from CPU now, read it back later?
//    • Allocate the buffer with `.storageModeShared`  → CPU and GPU share it coherently
//    • Move large chunks with a Blit encoder         → dedicated DMA, less GPU stall
let cpuVisible = device.makeBuffer(length: size,
                                   options: .storageModeShared)!
blitEncoder.copy(from: cpuVisible,  // read on GPU
                 to:   gpuPrivate,  // write here
                 size: size)

// 2) Data lives its whole life on-GPU (weights, static textures)
//    • Allocate with `.storageModePrivate`           → no CPU coherency overhead
//    • Upload once (Blit or staging buffer), then keep it resident
let gpuPrivate = device.makeBuffer(length: size,
                                   options: .storageModePrivate)!

// 3) Purely transient per-tile render data (e.g., depth, G-buffer)
//    • Create texture with `.storageModeMemoryless`  → lives only in on-chip tile memory
//    • Consumes zero DRAM bandwidth; disappears when render pass ends
let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float,
                                                    width: width, height: height,
                                                    mipmapped: false)
desc.storageMode = .memoryless
let tileDepth = device.makeTexture(descriptor: desc)!
```

**Key points**

* `storageModeShared`, `storageModePrivate`, and `storageModeMemoryless` are fields on **Metal resources** (`MTLBuffer`, `MTLTexture`).  
  *You set them in Swift because you're using the Metal framework from Swift.*

* A **Blit encoder** (`MTLBlitCommandEncoder`) is also a Metal concept: it programs the GPU's DMA hardware to copy or fill resources efficiently.

* None of this changes Swift's normal memory model—these flags only affect how the GPU sees and moves data.

Bottom line: these are **Metal-specific knobs** you set from your host code—Swift, Objective-C, metal-cpp, or any language with Metal bindings—whenever you spin up GPU workloads.

🛠️ **Python-only coder’s reality check:**  
Metal sits a lot closer to the metal—literally. The shading language is C-like, and the API makes you manage memory the way C does:

* **You allocate every buffer, choose a `storageMode`, and decide when it moves.** Forget a step and you get a crash or silent corruption; there’s no garbage collector to save you.  
* **Pointers, byte counts, and sync points are on you.** Miss one fence and you’ll chase heisenbugs all night.  
* **The trade-off:** that extra paperwork buys you near-zero overhead and predictable performance. Once you master the `Shared`, `Private`, and `Memoryless` modes—and the blit / compute copy patterns—you gain precise control that high-level runtimes can’t match.

**Bottom line:** expect some C-style headaches up front, but once the pipes are tight you get speed and determinism no high-level runtime can match. That said, most of you will never have to swim this deep—Python and its libraries happily hide the plumbing. Be grateful for the abstraction; keep this knowledge in your back pocket for the rare day you need it.

---

## 7. Takeaways for Day-to-Day Development  

1. **Allocate wisely**—Shared for CPU talk, Private for everything else.  
2. **Expect ~50 % of spec** under real load; don't chase the perfect number.  
3. **Measure the path you care about**—fabric copy vs. in-place GPU copy are different ceilings.  
4. **Move on when intuition matches a quick benchmark**; the extra 5 % isn't worth the lost build time.

---

### 8. Personal Take  

If your goal is simply to *understand* how memory and copy paths work on Apple Silicon, you're already there.

Think in object-oriented terms: tuck the nitty-gritty behind an interface and only crack it open when a real use-case demands it. Chasing every last micro-detail is a fast way to grind progress to a halt.

The big picture for this repo is learning how to squeeze the most out of Apple Silicon's unified memory for local AI workloads. Low-level CUDA or Metal kernels can wait until that strategy is clear; they're implementation details, not the roadmap. 

I drill into low-level memory architecture here **only** when it sharpens our view of the system as a whole. The goal is just enough depth to guide real-world choices—no detours into minutiae that derail our main mission: building efficient, local AI infrastructure on Apple Silicon. 

Trust me—work through this mini-guide and you'll already be ahead of most people in the field.

---

[⇧ Back&nbsp;to&nbsp;README](../README.md)