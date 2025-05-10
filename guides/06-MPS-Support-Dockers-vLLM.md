# Why we're skipping Docker **and** vLLM for this build

*(Personal lab notebook — read if it helps; ignore if it doesn't. 🙂 Last verified 2025‑04‑23)*

## 1. Metal / MPS support isn't fully baked for vLLM  
- **CUDA‑centric codebase** – vLLM's highest‑throughput kernels are still written for CUDA. The project has an experimental Metal backend, but it (a) lags several releases behind, (b) lacks tensor‑parallel and SSE optimizations, and (c) has no official wheels on PyPI.  
- **Performance cliff** – on M‑series chips the Metal path currently runs **30–50 % slower** than Ollama's in‑house Metal kernels. Until vLLM upstream merges full MPS parity, you'd be trading simplicity for lower throughput.

## 2. Docker adds overhead—and hassle—on Apple silicon  
- **Two layers of isolation** – Ollama already sandboxes models; wrapping it in Docker mainly duplicates file-system and network layers while eating ~1 GiB of extra RAM.  
- **GPU passthrough friction** – Docker‑for‑Mac still forwards the Metal device via a user‑space shim. If the container image wasn't built with Apple's `libmps` present, the GPU falls back to CPU, negating Metal.  
- **Service lifecycle** – Homebrew's `brew services start/stop ollama` integrates with launchd, supports auto-restart, and logs to `/usr/local/var/log`. Docker needs a separate restart policy or `docker run --restart`, plus port management.

## 3. Ollama already solves the "single-box" use-case cleanly  
- **Native Apple-silicon binaries** – pull once, Metal kernels compile, no container rebuilds.  
- **Minimal attack surface** – a single user-space daemon on loopback, no exposed Docker socket.  
- **OpenWebUI speaks Ollama natively** – with the `--ollama-base-url` flag the UI autolists models and handles streaming; no translation layer or extra JSON overrides required.

## 4. Upgrade path remains open  
- The moment vLLM's Metal backend reaches feature-parity **and** publishes arm64 wheels, you can:  
  1. `pip install -U vllm` in a fresh conda env,  
  2. start it on port 8000,  
  3. add an **OpenAI-compatible** provider in OpenWebUI.  
  No Docker rebuilds, no service conflicts—just an extra process.

**Bottom line:** for today's Apple-silicon hardware, **Homebrew Ollama + pip-installed OpenWebUI** gives you maximum Metal performance with the least operational overhead. When vLLM catches up on MPS, we can swap it in without retooling the rest of the stack.

---

[⇧ Back&nbsp;to&nbsp;README](../README.md)