# Open WebUI with Ollama backend

*(Personal lab notebook — read if it helps; ignore if it doesn't. 🙂 Last verified 2025‑04‑23)*

Below is a **clean, single-machine recipe** for running local language-models with **Ollama** (Metal-accelerated) and chatting through **Open WebUI**—**without Docker**.  
Everything happens on one Apple-silicon Mac; commands assume **zsh** + **Homebrew** + **Anaconda/conda**.

As of this writing, this is the setup my head-primary AI server is using.

---

### 1 · Install / update Ollama (Metal back-end)

```bash
# Homebrew tap (if not already)
brew update
brew install ollama

# Enable at login
sudo brew services start ollama
```

> *If you ever need to stop the service:*  
> `sudo brew services stop ollama`

Ollama listens on `localhost:11434` and uses **MPS** automatically on M-series chips—no extra flags required.

More good news: Ollama has officially started work on MLX support. See the appendix for more details.

---

### 2 · Create a clean conda env for Open WebUI

```bash
conda create -n openwebui python=3.11 -y
conda activate openwebui
```

---

### 3 · pip-install Open WebUI (server + webapp)

```bash
pip install open-webui   # pulls FastAPI, Uvicorn, SQLModel, etc.

# first-run startup
open-webui serve \
  --host 0.0.0.0 \
  --port 3000 \
  --ollama-base-url http://localhost:11434 \
  --admin-email you@example.com \
  --admin-password changeme
```

* `--ollama-base-url` points the UI at the local Ollama REST API.  
* Admin creds are created on first run; subsequent launches just use `open-webui serve`.

*(Tip: keep the env active and run `open-webui serve` in a tmux session or create a simple launchd plist.)*

Your LLM might suggest launching it automatically via a launchd plist. I'm not a fan—too many cryptic errors for too little gain. Stick to the hassle‑free tmux session. 

---

### 4 · Pull your first model with Ollama

```bash
# 14 B phi-4 (4-bit GGUF, ~15 GB download)
ollama pull phi4
# or deepseek-r1 (Q4_K_M, ~70 GB download)
ollama run deepseek-r1:70b 
# list installed
ollama list
```

Models live in `~/.ollama`. Metal kernels compile on first run (~30 s). 

Ollama models use a specific format that's not directly compatible with Hugging Face or LMStudio models. While conversion between formats is possible using various tools, the process can be complex and results may vary depending on the specific model architecture.

Formats are discussed in a separate guide.

---

### 5 · Chat through the web UI

1. Open <http://localhost:3000> in a browser.  
2. Log in with the admin account you set.  
3. The **model picker** auto-populates from `ollama list`.  
4. Choose `phi4` (or whatever you pulled) → chat.

You're now running an end-to-end, GPU-accelerated local LLM purely on the Mac.

---

### 6 · Command-line sanity checks

```bash
# single prompt
curl -s http://localhost:11434/api/generate \
  -d '{"model":"phi4","prompt":"Hello","stream":false}' | jq

# stream tokens
ollama run phi4
```

---

### 7 · Updating pieces

```bash
# keep Ollama fresh
brew upgrade ollama
sudo brew services restart ollama

# update OpenWebUI env
conda activate openwebui
pip install -U open-webui
```

If the WebUI schema changes, stop the server, optionally clear its DB:

```bash
rm ~/.local/share/openwebui/openwebui.db  # wipes history/settings
```

---

### 8 · Typical troubleshooting

```text
• "No models found" drop-down    → run `ollama pull <model>` then refresh UI.
• "Model download slow"          → use a wired connection; downloads can be 8–20 GB.
• "context length exceeded"      → lower `num_predict` slider in UI or add `"num_predict":1024` via API.
• "Port already in use :3000"    → launch UI on another port: `open-webui serve --port 3100 ...`.
```

> **How to tell if Ollama is using MPS**  

In terminal, you can check the backend with:

```bash
ollama ps
```

If that prints:

```
NAME          ID              SIZE     PROCESSOR    UNTIL
gemma3:27b    30ddded7fba6    22 GB    100% GPU     27 seconds from now
```
Then you're using the Metal backend just fine.

But practically, the quickest check is simply *speed*. Load a reasonably sized model (7‑8 B, Q4/Q5) and chat. If the reply streams in at conversational pace, Metal/MPS is active. If tokens crawl out at under 0.5 tok/s—even after the first-prompt compile—Ollama has fallen back to CPU.  

Trust me: you'll feel the difference.  
  
If you need hard proof, run the above terminal command or open **Activity Monitor → GPU** (or run `mactop`) while chatting—GPU usage should spike while the model responds.

---

### 9 · Quick stop/start cheatsheet

```bash
# stop everything
sudo brew services stop ollama     # backend
conda activate openwebui
pkill -f "open-webui"              # front‑end

# restart later
sudo brew services start ollama
conda activate openwebui
open-webui serve \
  --host 0.0.0.0 \
  --port 3000 \
  --ollama-base-url http://localhost:11434
```

Running `open-webui serve` with no flags defaults to `localhost:3000` and the correct backend URL.

---

### 10 · Next steps

* Pull multiple models (`ollama pull llama3:70b-q8_0`, etc.) and switch in UI.  
* Explore Open WebUI "Tools" panel for function calling / file uploads.  
* Fine-tune small models via `ollama create` / `ollama run -f <Modelfile>`.

---

### 11 · Update Open WebUI

Below are three reliable ways to get the **very-latest Open WebUI bits straight from the GitHub repo**, bypassing the laggy PyPI wheel.  

It happens quite often: OpenWebUI flashes a new release notification but pip-installed wheels lag behind. Especially when it's a critical bug-fix, it's a real pain. Here's how to get the latest without waiting on PyPI. 

All commands go in your existing `openwebui` conda env.

---

### 1 · One-liner: install from `main` (or any branch/commit)

```bash
# ✱ uninstall any old wheel
pip uninstall -y open-webui

# install direct from the repo's main branch
pip install --no-cache-dir "git+https://github.com/open-webui/open-webui@main#egg=open-webui"
```

* Swap `@main` for a tag (`@v0.2.14`) or commit SHA (`@1a2b3c4d`) if you need a specific revision.  
* `--no-cache-dir` ensures pip doesn't reuse an old wheel.

Run with:

```bash
open-webui serve --host 0.0.0.0 --port 3000 --ollama-base-url http://localhost:11434
```

---

### 2 · Editable clone (easy to patch, auto-reload on `git pull`)

```bash
# clone once anywhere you like
git clone https://github.com/open-webui/open-webui.git ~/src/openwebui
cd ~/src/openwebui

# install in editable mode
pip install -e ".[server]"
```

*Update later*:

```bash
cd ~/src/openwebui
git pull
# frontend assets are pre-built; no extra step needed
```

---

### 3 · Clean rebuild after large changes

If the repo bumps major deps (SQLModel, FastAPI), do a fresh env:

```bash
conda remove -n openwebui --all -y
conda create -n openwebui python=3.11 -y
conda activate openwebui
pip install --no-cache-dir "git+https://github.com/open-webui/open-webui@main#egg=open-webui"
```

### 4 · Last-resort: install directly from a GitHub release tarball  

Use this when:

* PyPI is lagging **and** the `main` branch has breaking changes you don't want.
* Your firewall blocks `git+https` installs but still allows plain HTTPS file downloads.

```bash
# 1) grab the exact tag you want (replace v0.2.14 as needed)
curl -LO https://github.com/open-webui/open-webui/archive/refs/tags/v0.2.14.tar.gz

# 2) install the tarball (pip unpacks it automatically)
pip install --no-cache-dir open-webui-0.2.14.tar.gz
```

Notes  

* `--no-cache-dir` avoids pip reusing an old wheel if you reinstall later.  
* If you keep multiple versions for testing, add `--force-reinstall`.  
* Updating to a newer tag is the same two-step dance—download the new tar, `pip install` it, and you're done.

---

## Notes & Tips

* The GitHub project **ships pre-built frontend assets**, so no Node/PNPM step is needed for a standard install.  
* If you experiment with a feature branch and want to roll back, just reinstall from `main`:

  ```bash
  pip install --force-reinstall --no-cache-dir \
      "git+https://github.com/open-webui/open-webui@main#egg=open-webui"
  ```
* Use `open-webui --help` to list start-time flags; they are identical regardless of install method.

That's all you need to stay fully up-to-date without waiting on PyPI.

---

### Appendix · Incoming MLX Backend for Ollama (Apple-silicon roadmap)

**Status (April 2025)**  
* GitHub PR #9118 "Draft MLX go backend for new engine" opened 14 Feb 2025.  
* Branch adds a selectable runtime:  
  ```bash
  # nightly testers
  export OLLAMA_BACKEND=mlx         # default remains 'metal'
  export OLLAMA_NEW_ENGINE=1        # enable new code-path
  ollama run llama3
  ```  
* Loads FP16 weights via **MLX** (Apple's JIT tensor library); quantized paths and cache invalidation still flagged *TODO*.

---

#### Why MLX matters

| Current Metal GGUF | MLX backend |
|--------------------|-------------|
| Pre-compiled Metal kernels per quant level—fast but rigid. | JIT-compiled tensor ops; easier to add new precisions & ops. |
| 16-bit FP models require massive RAM, so most Ollama builds use 4-/8-bit GGUF. | MLX v0.12 adds memory-mapping + Flash-attention; FP16/FP8 viable within 512 GB unified RAM. |
| Limited inter-op with Apple's broader ML tooling. | MLX shares the same tensor algebra used in Core ML pruning scripts and Swift for TensorFlow revival. |

---

#### Near-term testing steps

1. **Nightly install**  
   ```bash
   brew install --HEAD ollama   # until PR is merged & tagged
   ```
2. **Environment flags**  
   ```bash
   OLLAMA_BACKEND=mlx
   OLLAMA_NEW_ENGINE=1   # activates go-mlx path
   ```
3. **Compatibility caveats** (as of PR description)  
   * Caching: eviction logic still tied to old byte-tensor layout—may rebuild kernels every session.  
   * Quantization: `gguf-q8_0` loads; Q4/Q5 variants freeze at `allocate_tensor`.  
   * Streaming: SSE wrapper works, but per-token latency ~5 ms higher than Metal build (profiling January M3 Ultra).

---

#### What this means for our stack

* **Ray Serve proxy stays unchanged** – it forwards OpenAI JSON; backend URL stays `localhost:11434` regardless of Metal vs. MLX engine.  
* Potential to **drop vLLM** for large FP16 models once MLX backend stabilizes—512 GB Beta can host Llama-3 70 B FP16 directly in Ollama.  
* Monitor the PR for `Cache stable` and `Quant full` check-marks; once merged into a tagged release, update guide section 7 with "`brew upgrade ollama` → toggle backend flag".

---

**Take-away**  
MLX support is now on the official roadmap; early adopters can experiment by switching environment flags, but for production keep the Metal backend until the PR lands in a stable release.

---

[⇧ Back&nbsp;to&nbsp;README](../README.md)