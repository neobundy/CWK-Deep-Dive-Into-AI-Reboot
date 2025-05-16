# One-Box Full AI Stack Deep Dive

- **Target machine:** Apple Silicon Mac Studio (M3 Ultra, 80-core GPU, 512 GB unified memory)
- **Core stack:** ggml • llama.cpp • llama-cpp-python • Ollama • Open WebUI
- **Goal:** Two parallel tracks—**Homebrew quick-install** for everyday use and **from-source builds** for hacking & profiling.

---

## Crucial Notes on the Stack

This toolchain is **production-oriented** by design and keeps the mental model as lean as possible. If a favorite framework isn't here, the reasons usually fall into one of four buckets:

1. **Maturity & Support** — we stick to projects that compile cleanly today, ship stable APIs, and are actively maintained.
2. **Architectural Fit** — every piece hangs off the same ggml → llama.cpp spine. Dropping in orthogonal runtimes would splinter that design.
3. **Cognitive Budget** — five moving parts is already plenty; bolting on extras would dilute focus without clear upside.
4. **Pragmatism over Tribalism** — there's zero brand loyalty at play. The moment another tool proves faster, stabler, or clearer, it'll earn a slot. For now, these five deliver the best return on effort.

In short: this setup is intentional and based on experience. If your needs or preferences differ, that's totally fine—adapt as you see fit. Everyone's workflow is unique.

**Quick sanity check for tech debates:** if the brightest teams across industry and academia haven't adopted a "hot new thing," it's probably not battle-tested yet. Let the collective expertise shake out the rough edges—when those minds pivot, you'll know it's safe to follow.

**Note:** The stack outlined here is fixed by design. Requests to swap out or debate components won't be entertained—this guide is focused on a proven, stable setup.

> ⚡️ **Lab Reality Check:**  
>  
> This repo is a raw brain-dump on how I squeeze every drop of AI performance out of a pair of Mac Studio M3 Ultras.  
> **Take what helps—skip what doesn't.**

---

## Roadmap

- [Chapter 1 · One-Box Full AI Stack: Dual-Track Setup Guide](01-setup.md)
- [Chapter 2 · Meet *ggml* — the Little Engine Behind Your AI Stack](02-gglm.md)
- [Chapter 3 · *llama.cpp* — Tiny-Ollama & the Bare-Metal Loop](03-llama-cpp.md)
- [Chapter 4 · *Go* in 60 Gentle Minutes — A Practical Primer for the Curious Engineer](04-go-in-60m.md)
- [Chapter 5 · *Ollama* on an M3 Ultra — From Prompt to Metal Shader](05-ollama.md)
- [Chapter 6 · Hacking Ollama in Go — A Zero-to-Hero Lab](06-ollama-hacking-in-go.md)
- [Chapter 7 · Extracting Models from Ollama — Building a CLI Feature in Go](07-ollama-hacking-advanced.md)
- [Chapter 8 · *llama-cpp-python* — The Python Interface to the Metal Engine](08-llama-cpp-python.md)
- [Chapter 9 · **Open WebUI** — A Browser Cockpit on Top of Ollama and More](09-open-webui.md)
- [Chapter 10 · Open WebUI Tools and Functions](10-open-webui-tools-and-functions.md)