# From Numbers to Insight: Decoding GPU Specs with Two Flagship Examples - M3 Ultra vs RTX 4090 

*(Personal lab notebook — Last verified 2025‑05‑04)*

*Quick-n-dirty field notes comparing Apple M3 Ultra vs NVIDIA RTX 4090 for AI workloads.*

## Important Note on Comparison Methodology

I've specifically chosen the **RTX 4090** (24 GB VRAM) for this cheatsheet because, right now, it's the most common "big iron" in consumer-level AI rigs. The venerable **RTX 3090** (also 24 GB) holds the runner-up slot. Yes, an RTX 5090 technically exists, but adoption is still niche, and true enterprise-class NVIDIA boards sit outside our scope. For context, I still have 3090- and 4090-based boxes humming away in my lab—and I'm in no rush to retire them.

Apple's **M3 Ultra** (80-core GPU, 512 GB unified memory) is a very different animal: a top-shelf prosumer chip rather than a data-center brute. Remember, real enterprise GPUs usually ship *without* fans—they're built for 24/7 duty inside chilled racks with dedicated air-handling.

So, don't treat this write-up as a head-to-head shoot-out. Each architecture follows its own design philosophy and set of trade-offs. My goal is to highlight those differences so you can read spec sheets with a bit more nuance—and pick the right tool for *your* workload.

Bottom line: asking "Which one is better?" is like comparing a Ferrari to a city bus. Context is everything—and you're the driver. (Seriously, you wouldn't take a Ferrari through stop-and-go Seoul Kangnam traffic every morning, would you?)

You get the point.

---

## 1 · Usual suspects of GPU performance metrics (But not enough to give a full picture)

1. **Memory bandwidth**
2. **Throughput (TFLOPS)**
3. **"New features"**—tensor/RT cores, data-type tricks, etc.

We'll weave the rest of this guide around those three yardsticks. Spoiler: by themselves they won't give you the full picture—they're just handy for quick head-math, nothing more.

---

## 2 · Mini-Guide: How to calculate memory bandwidth

**Simple formula for memory bandwidth:**

Memory Bandwidth (GB/s) = (Bus Width in bits ÷ 8) × Memory Clock Speed (MT/s) ÷ 1000

Where:
- Bus Width is measured in bits (divide by 8 to convert to bytes)
- Memory Clock Speed is measured in Mega Transfers per second (MT/s)
- The result is divided by 1000 to convert from MB/s to GB/s — we intentionally use **1,000** (rather than 1,024) for easy head-math; the numerical difference is only ≈2.4 %.

This formula works for both traditional GDDR memory and Apple's unified memory architecture, though the implementation details differ significantly.

**Simple example:**
Let's calculate the memory bandwidth for a hypothetical GPU with a 256-bit memory bus running at 8000 MT/s:

1. Convert bus width to bytes: 256 bits ÷ 8 = 32 bytes per transfer
2. Multiply by transfer rate: 32 bytes × 8000 MT/s = 256,000 MB/s
3. Convert to GB/s: 256,000 MB/s ÷ 1000 = 256 GB/s

Put differently, that GPU can sling about **256 GB** every second between memory and its cores—roughly the same as moving an entire 100 GB 4K movie in under half a heartbeat.


| Step                  | Formula                                         | M3 Ultra Example                                           | RTX 4090 Example                                                   |
| --------------------- | ----------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------ |
| 1. Gather bus width   | bits ➗ 8 = bytes/transfer                       | 1024-bit unified bus → 128 bytes                           | 384-bit GDDR6X bus → 48 bytes                                      |
| 2. Gather data-rate   | mega-transfers/s (MT/s)<br>or gigabits/s (Gbps) | LPDDR5-6400 → 6 400 MT/s                                   | 21 Gbps per pin                                                    |
| 3. Multiply & convert | (bytes × rate)/1 000 ≈ GB/s                     | 128 × 6400 = 819,200 MB/s ≈ **819 GB/s** | 48 × 21,000 = 1008,000 MB/s ≈ **1,008 GB/s** |

**Sidebar — MT/s vs Gbps**  
**MT/s** (mega-transfers per second) counts the number of data transfers regardless of bus width.  
**Gbps** (gigabits per second) is the raw bit rate per pin.  
For bandwidth math you can treat 21 Gbps per pin on GDDR6X as ≈21,000 MT/s on a one-bit lane; multiply by bus width and divide by eight to reach MB/s.

**Reflection.**
Even though Apple's bus is "only" LPDDR5, sheer width (1024 bits!) lets it play in the same-terabyte arena as GDDR6X. But the math also shows why NVIDIA can leap ahead with HBM on data-center cards—raise both width *and* speed.

TFLOPS = (1,280 × 16 × 1,400,000,000) ÷ 1,000,000,000,000 = 28.67 TFLOPS

*Apple marketing note: the "80-core GPU" label refers to 80 graphics-core clusters; each cluster houses 16 execution units, giving the 1,280 EUs used in this calculation.*

---

## 3 · Raw Throughput (TFLOPS) ≈ Engine Size

### How to calculate TFLOPS (Teraflops)

TFLOPS (Trillion Floating Point Operations Per Second) is a key metric for measuring raw GPU computational power. Here's how to calculate it:

**Basic TFLOPS Formula:**
TFLOPS = (Number of Compute Units × Operations per Clock × Clock Speed) ÷ 1,000,000,000,000

Where:
- Number of Compute Units: Total number of shader/stream processors or execution units
- Operations per Clock: How many floating-point operations each unit can perform per clock cycle
- Clock Speed: The GPU's frequency in Hz (cycles per second)
- Division by 1 trillion converts to teraflops

**Example Calculation:**
For an M3 Ultra with 80 GPU cores (1,280 execution units), running at 1.4 GHz:
1. 1,280 execution units
2. Each unit can perform 16 FP32 operations per clock (Apple's architecture)
3. Clock speed is 1.4 GHz (1,400,000,000 Hz)

TFLOPS = (1,280 × 16 × 1,400,000,000) ÷ 1,000,000,000,000 = 28.67 TFLOPS

**Different Precision Levels:**
- FP32 (single-precision): Standard calculation as above
- FP16 (half-precision): Often 2× the FP32 rate on modern GPUs with dedicated hardware
- INT8/INT4: Can be 4-8× faster on GPUs with specialized acceleration units

**Architecture Differences:**
- Apple Silicon: Generally maintains the same TFLOPS across precision levels (Apple does not expose a dedicated low-precision accelerator, so FP16/BF16 gains are limited. Apple doesn't publish per-precision ratios.)
- NVIDIA: Tensor cores provide dramatic acceleration for lower precision operations
- AMD: Matrix cores offer similar benefits for specialized workloads

Understanding TFLOPS helps compare raw computational potential, but remember that real-world performance depends on many other factors including memory bandwidth, architecture efficiency, and software optimization.

### Understanding Floating-Point Performance Units

Floating-point operations per second (FLOPS) is a measure of computer performance, specifically for calculations involving floating-point numbers. The units scale as follows:

- **FLOPS**: Floating-Point Operations Per Second (base unit)
- **KFLOPS**: Kilo FLOPS = 1,000 FLOPS
- **MFLOPS**: Mega FLOPS = 1,000,000 FLOPS
- **GFLOPS**: Giga FLOPS = 1,000,000,000 FLOPS
- **TFLOPS**: Tera FLOPS = 1,000,000,000,000 FLOPS
- **PFLOPS**: Peta FLOPS = 1,000,000,000,000,000 FLOPS

Modern consumer GPUs typically operate in the TFLOPS range, while high-end supercomputers reach into PFLOPS territory. For context, a GPU with 10 TFLOPS performance can theoretically perform 10 trillion floating-point calculations per second.

When comparing GPUs, it's important to note which precision level the FLOPS rating refers to:

- FP32 (single-precision) TFLOPS is the standard reference point
- FP16 (half-precision) TFLOPS is often 2× higher on the same hardware
- INT8/INT4 operations may be reported in TOPS (Tera Operations Per Second) rather than TFLOPS

The relationship is straightforward: 1 TFLOPS = 1,000 GFLOPS. When reviewing specifications, always check which precision level is being referenced, as marketing materials sometimes highlight the most favorable metric without clear context.

---

| Metric                  | M3 Ultra 80-core GPU                                                           | RTX 4090                                         |
| ----------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------ |
| FP32 (single-precision) | ≈ 28 TFLOPS (1.40 GHz, 1 280 EUs) | 82.6 TFLOPS @ stock clocks    |
| FP16/BF16               | Same as FP32 (no tensor cores)   | Official peak (no sparsity) is 330 TFLOPS with FP16 accumulate; 165 TFLOPS is the FP32-accumulate mode.    |
| FP8 / INT4              | GPU cores only (Neural Engine handles INT8, 36 TOPS) | ~330 TFLOPS observed, 1.3 PFLOPS theoretical.|

**Reflection.**

Think of Apple's GPU as a big naturally-aspirated V12 engine—it delivers steady, predictable torque across all precision levels. The architecture prioritizes consistent performance rather than specialized acceleration. Each compute unit handles all precisions equally well, giving you the same TFLOPS whether you're running FP32 or FP16 operations, but it's not as good at lower precision operations.

Think of **Apple's GPU** as a naturally-aspirated V12: smooth, predictable torque no matter the gear. It posts the same TFLOPS whether you're crunching FP32 or FP16, but with no dedicated low-precision hardware it loses steam once you drop below FP32.

**NVIDIA**, by contrast, bolts turbo-chargers onto its engine in the form of **Tensor Cores**. These matrix units light up at FP16/BF16/INT8/INT4 and hand the RTX 4090 a 2-4× throughput boost when you quantize—exactly what large-language-model inference craves.

That split personality explains why Apple Silicon feels well-rounded for general compute, while NVIDIA flexes hardest in pure AI inference once you embrace lower precisions.

**Key takeaway:**
- Specs in isolation lie. Core counts, bandwidth, and headline TFLOPS are just trees; the forest is how the pieces work together when you hit "run." Always judge the full system, not a single stat.

---

## 4 · Memory vs. VRAM: The Passenger Cabin

| Aspect      | M3 Ultra                                                                         | RTX 4090                        |
| ----------- | -------------------------------------------------------------------------------- | ------------------------------- |
| Capacity    | Up to **512 GB** unified, CPU & GPU share same pool  | Fixed **24 GB** GDDR6X          |
| Access path | On-package LPDDR5, 0 copy between CPU & GPU                                      | PCIe transfer to discrete card  |
| Latency     | Low (same silicon interposer)                                                    | Higher; batching/overlap needed |

**Why it matters.**
A 671 B-parameter 4-bit model fits entirely on the M3 Ultra's 512GB memory and runs under 200W; the same job on RTX 4090s requires model sharding across multiple cards due to the 24GB VRAM limitation.

Practically speaking, a **672-billion-parameter** model in 4-bit form (weights + KV cache) *can* shoehorn into 512 GB of unified memory—but that checkpoint is a research one-off, not something you can just `git clone`. And "fits" doesn't equal "feels snappy." From hard-earned experience, anything below **~10 tokens/sec** feels like molasses in an interactive chat. Even at 4-bit, monster models often fall short of that bar on consumer gear, so be ready to trade raw parameter count for real-world responsiveness.

---

## 5 · Power & Thermals: Luxury Sedan vs. Supercar

| Metric (max draw) | M3 Ultra Mac Studio                                                          | RTX 4090 board                                                                     |
| ----------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Power (W)         | **270 W** for the whole Mac Studio (CPU+GPU, SSD, fans) | **450 W** for the card alone, \~650 W with a high-end x86 tower |
| Noise & cooling   | Single blower under 40 dB                                                    | Triple-slot cooler, 50–55 dB typical                                               |
| Footprint         | 3.7 L desktop, one cable                                                     | Full ATX case, beefy PSU                                                           |

**Analogy.**  
The M3 Ultra is a luxury EV—whisper-quiet, sip-efficient, yet surprisingly punchy. The RTX 4090 is a track-tuned supercar: blistering speed, huge appetite for watts, and impossible to ignore.

**Real-world experience:**  
My Mac Studio purrs so softly I have to touch the aluminum to be sure it's alive. Fire up the same job on the 4090 tower and it goes full jet-engine, pumping enough heat to double as a space-heater during Seoul winters.

Run identical workloads and you'll feel the contrast instantly: the Mac chills in Zen-mode while the 4090 rig throws a metal concert—volume, heat, and all. Memory tells a similar story: you'd need a stack of 4090s to rival the M3 Ultra's unified pool.

That's the real trade-off: comfort and capacity versus raw, unfiltered horsepower.

(And yes, I'm still a Big Four headbanger—Metallica, Megadeth, Slayer, Anthrax on loop 🤘)

---

## 6 · Feature Dialects (the "new features" bullet)

| Category        | Apple M3 Ultra                                             | NVIDIA RTX 4090                                                         |
| --------------- | ---------------------------------------------------------- | ----------------------------------------------------------------------- |
| Matrix hardware | General-purpose ALUs, dynamic caching                      | 4th-gen Tensor Cores (FP8, BF16, INT8/4)                                |
| Ray tracing     | First-gen HW RT, Metal 3 API                               | 3rd-gen RT Cores, DXR, Vulkan RT                                        |
| AI coprocessor  | 32-core Neural Engine, 36 TOPS INT8 ([Wikipedia][1])       | Tensor Cores double as AI units (1.3 peta-INT4)                         |
| Software stack  | Metal + MLX / PyTorch-Metal; no CUDA                       | CUDA & cuDNN, Flash-Attention, huge ecosystem                           |
| Expandability   | USB-C/TB5, eGPU not supported; what you buy is what you get | RTX 4090 has no NVLink; NVLink is reserved for H100/Hopper class |

---

## 7 · Putting it together: choosing the right "class"

| Use-case                                      | Why M3 Ultra wins                   | Why 4090 wins                              |
| --------------------------------------------- | ----------------------------------- | ------------------------------------------ |
| **Ultra-large models that *fit* in 512 GB**   | Zero sharding; lower power; quieter | Needs multiple cards → higher cost & power |
| **Mixed-precision training / fine-tuning**    | Works, but slower—no Tensor cores   | FP8/BF16 Tensor Cores accelerate 2-4×      |
| **Realtime mobile-to-desktop dev**            | Same code path, unified memory      | Must juggle CPU↔GPU copies                 |
| **Gaming / real-time RT**                     | Playable, but not flagship          | DLSS 3, higher RT performance              |
| **Total cost of ownership (electric + HVAC)** | 200–270 W ceiling                   | 650 W+ rig + AC overhead                   |


###  Home-lab Power Caveat ⚡💸🔌

A single RTX 4090 rig under sustained AI load can draw 650W+. Two such boxes—or one big quad-GPU chassis—push well beyond 1.3 kW. On a standard U.S. 15-amp, 120 V branch circuit that's already flirting with the 1,800 W safety ceiling (80% of 1,500 W).

In Korea (220 V), a typical 20 A breaker trips around 4.4 kW. Two or three fully-loaded GPU towers plus monitors and peripherals can get you there faster than you'd think.

**Translation:** Before you scale out with multiple dedicated GPUs at home, budget not just for the electric bill but possibly for:

• A dedicated 20A (U.S.) or 30A (KR/EU) circuit  
• A beefier PSU per box  
• Heat extraction that follows all those extra kilowatts

Otherwise you'll meet your breaker long before you meet OOM errors!

On my side, the house runs on a 50A service and the lab gets its own 30A feed. That headroom lets several hungry rigs crunch away without nuking the breakers—double-check your panel before you splurge on fresh GPUs.

---

## 8 · OO Lens Wrap-Up

Think of **`GPU`** as a base *class* with three encapsulated fields: `bandwidth`, `flops`, `features`.

* **`AppleM3Ultra`** overrides **memory** with a *giant capacity* attribute and a modest `flops`.
* **`NvidiaRTX4090`** overrides **flops** & **features** (Tensor/RT cores) but keeps a smaller `memory` field.

Polymorphism at work: pick the *object* whose attributes align with the methods (workloads) you'll call the most.

---

### TL;DR Cheat Sheet

| Spec        | M3 Ultra              | RTX 4090                  |
| ----------- | --------------------- | ------------------------- |
| Memory BW   | \~ 819 GB/s           | 1,008 GB/s                |
| VRAM / RAM  | 512 GB UMA            | 24 GB GDDR6X              |
| FP32 TFLOPS | 28                    | 83                        |
| Peak power  | 270 W (entire box)    | 450 W (GPU only)          |
| Noise       | <40 dB                | \~55 dB                   |
| Key edge    | Capacity & efficiency | Raw speed & AI tensor ops |

**Noise disclaimer:** Decibel charts can be slippery—results swing wildly with mic placement and room acoustics. In real-world use I barely hear the M3 Ultra (<30 dB at 1 m), but a 4090 rig cranks up to 45–55 dB once the fans hit turbo. In a quiet office or during marathon jobs, that gap is impossible to ignore.

---

[⇧ Back&nbsp;to&nbsp;README](../README.md)