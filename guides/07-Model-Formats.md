# Model Formats 101 — A Video‑Container Analogy

*(Personal lab notebook — read if it helps; ignore if it doesn't. 🙂 Last verified 2025‑04‑23)*

**Goal:** demystify common LLM weight formats (GGUF, Safetensors, PyTorch .pt, etc.) by comparing them to familiar video‑file concepts (codecs vs. containers). Focus is on fundamentals useful for daily ops: what lives in the file, why formats exist, how runtimes consume them.

---

## 1. Core Analogy at a Glance

| Video world                | LLM world                              | Purpose                                    |
|----------------------------|----------------------------------------|--------------------------------------------|
| **Codec** (H.264, VP9, AV1)| **Tensor encoding** (FP16, INT4, GPTQ) | How raw data is numerically compressed.    |
| **Container** (MKV, MP4)   | **Model format** (GGUF, Safetensors)   | Wrapper that bundles encoded data **plus** metadata required for decoding/use. |
| **Media player** (VLC)     | **Runtime** (Ollama, llama.cpp, vLLM)  | Software that parses the container and feeds decoded frames/tokens to the GPU/CPU. |

> Just as MP4 can carry H.264 **or** HEVC, GGUF can carry FP16 tensors **or** INT4 GPTQ tensors. The runtime must support both the container *and* the codec within.

---

## 2. Why Containers Exist

1. **Self‑describing metadata**  – shape, vocab size, quant type → no external config files.
2. **Future‑proofing**           – new quantizations can be added without changing runtime APIs (just parse header tags).
3. **Multiplexing**              – single file may include LoRA deltas, tokenizer, or even multiple weight shards.
4. **Integrity checks**          – header includes training hash / checksum; runtime can verify before loading.

---

## 3. Walk‑through of Three Widely‑Used Formats

### 3.1 PyTorch `.pt` / `.bin`

* **Container**: flat pickled Python dict.
* **Pros**: universal across research repos; saves optimizer states.
* **Cons**: unsafe loading, endianness issues, large (raw FP16/32). **Not** stream‑friendly.

### 3.2 Safetensors `.safetensors`

* Think "MP4 fixed-size header instead of pickle."
* **Key features**: memory‑map ability, alignment guarantees, SHA256 integrity.  
  ```bash
  safetensors-cli show model.safetensors
  ```
* Used by Hugging Face for FP16/8‑bit weights.

### 3.3 GGUF `.gguf`

* Designed for **quantized** llama.cpp ecosystem; text header (`GGUFv3\n`).
* Bundles: tensors, tokenizer JSON, quant scheme, optional LoRA.
* **Stream reading** via mmap; INT4/8 kernels load fast on Metal/CUDA.

---

## 4. Header Anatomy (GGUF example)

```text
GGUFv3\n
key: arch            → "llama"
key: vocab_size      → 32000
key: tensor_data     → offset 0x003e8900
key: qtype           → Q4_K_M
…
```
Runtimes read the header → allocate correct tensor shapes → memcpy or mmap the payload.

---

## 5. Runtime Flow: Loading vs. Decoding

1. **Parse container header** (like MKV EBML).  
2. **Check codec support** (e.g., INT4 GPTQ → do we have kernels?).  
3. **Memory‑map or copy** tensor blocks to GPU/CPU RAM.  
4. **Run inference kernels** (Metal, CUDA, MLX).  

If any step fails, you might see errors like *"unsupported quantization type Q2_K"* (analogous to trying to play AV1 on a player without that codec).

---

## 6. Practical Ops Questions

| Question | Video analogy | Model answer |
|----------|---------------|--------------|
| **Can I change quantization without retraining?** | Re‑encode H.264 → HEVC | Yes: quantize offline — GPTQ/AWQ → GGUF **or** `mlx_lm.convert -q` → MLX checkpoint. |
| **Can I merge models?** | Remux two audio tracks into MKV | Use LoRA merge or weight diff → write new container with both tensor sets. |
| **Why is my model huge?** | Raw .yuv vs. compressed .mp4 | It's FP32/PT; convert to FP16 or INT4 & save as Safetensors / GGUF. |

You can convert **and** quantize a Hugging Face model for Apple‑silicon in one shot with

```bash
python -m mlx_lm.convert --hf-path <model_repo> -q   # writes to ./mlx_model by default
```

• `-q` defaults to 4‑bit group‑quant (g=64).  
• Use `--q-bits` / `--q-group-size` to pick other schemes.  
• The output is an MLX model folder containing NumPy .npz weight shards plus `config.json` & `tokenizer.json`.

Run it with:

```bash
python -m mlx_lm.generate --model ./mlx_model --prompt "Hello"
```

Limitations
1. The resulting checkpoint is **MLX‑only**; it can't be loaded by llama.cpp/Ollama.  
2. MLX‑LM reloads weights on every invocation (no long‑lived server cache like Ollama). cwkLLMTextForge defaults to ollama backend exactly for this reason unless I have to use MLX models, in which case I switch to the MLX backend using `mlx-lm.generate` under the hood.
3. HF files are still cached under `~/.cache/huggingface`, so subsequent runs avoid re‑download.

---

## 7. Migration‑Ready Checklist

```bash
# 1. Pick target container → GGUF (quantized) or Safetensors (FP16)
# 2. Pick codec/quant  → Q4_K_M, GPTQ, AWQ, etc.
# 3. Convert once (Python script or llama.cpp quantize)
# 4. Verify runtime > supports that codec
# 5. Test with: /v1/models list + chat prompt
```

---

## 8. Future trends

* **MLX weights** – Apple's MLX library saves checkpoints as standard NumPy archives (`.npz`) via `mlx.core.savez`. The format exists today but isn't yet widespread.
* **Block‑wise streaming** – GGUF v4 draft discusses interleaved chunks for faster cold starts (think progressive MP4).

---

## 9. Take‑home

* Model *format* = container; quantization = codec.  
* Choosing the right pair impacts load time, memory, and runtime compatibility more than the model architecture itself.  
* Understanding headers & metadata lets you debug "unsupported quant" errors quickly—just like checking a video's codec before blaming the player.

---

# Model Sourcing Guide — Hugging Face Hub vs Ollama Registry vs LM Studio

**Focus:** where the weights come from, how they're packaged, where they land on disk, and how to mirror / purge / update them.

---

## 1. High‑level comparison

| Aspect | **Hugging Face Hub** | **Ollama Registry** | **LM Studio Catalog** |
|--------|----------------------|---------------------|-----------------------|
| Primary URL | `https://huggingface.co/<org>/<model>` | `https://registry.ollama.ai/library/<name>` | Desktop app pulls `https://models.lmstudio.ai/<id>` (often mirrors HF) |
| Transport | HTTPS (git‑LFS or `hf://` REST) | Custom HTTP JSON + layer blobs | Signed JSON manifest + direct blob (S3) |
| File types | Safetensors (`.safetensors`), PyTorch (`.bin`), GGUF | GGUF (quantised) | GGUF + JSON sidecar |
| Local cache (macOS) | `~/.cache/huggingface/hub` | `~/.ollama` | `~/.cache/lm-studio/models` (*current public build 0.4.x*) |
| Versioning | Branches, tags, `revision=` | Semver tag (`llama3:70b-q8_0`) | Folders named `<model>@<sha256>`; UI shows date |
| Private / gated models | Token via `huggingface-cli login` | None yet (planned) | Not supported |
| Offline reuse | Copy `HF_HOME` dir; set env var | Copy `.ollama` folder | Copy/symlink `.cache/lm-studio/models` |

---

## 2. Hugging Face Hub essentials

### 2.1 Install & login
```bash
pip install huggingface-hub
huggingface-cli login
```

### 2.2 Programmatic download
```bash
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2 \
  --local-dir ~/Models/mistral-7b
```

### 2.3 Move cache
```bash
export HF_HOME=/Volumes/Models/hf-cache
```

### 2.4 Purge old revisions
```bash
huggingface-cli cache list | head
huggingface-cli cache purge --yes
```

### 2.5 Automatic download via pipelines
Many libraries (*transformers*, **MLX‑LM**, PEFT, etc.) will **auto‑download** a model the first time you instantiate a pipeline with a Hugging Face model ID. No manual `download` call is needed.

```python
from transformers import pipeline

# first call: downloads weights to ~/.cache/huggingface
pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2", device="mps")
print(pipe("Hello", max_new_tokens=10))
```

Or using MLX‑LM:

```bash
pip install mlx-lm
mlx-lm run mistralai/Mistral-7B-Instruct-v0.2 --prompt "Hello" --max-tokens 10
```
Both commands resolve `mistralai/Mistral-7B-Instruct-v0.2`, stream the shards into your HF cache, then reuse them on subsequent runs.

If you want these auto‑downloads to land on a bigger drive, just set `HF_HOME` before launching the script:

```bash
export HF_HOME=/Volumes/Models/hf-cache
python my_script.py
```
---

## 3. Ollama registry quick facts

### 3.1 Pull & list
```bash
ollama pull llama3:70b-q8_0
ollama list
```

### 3.2 Disk layout
```
~/.ollama/
 ├─ llama3/70b-q8_0/
 ├─ phi4/14b-q4_k_m/
 └─ blobs/                   # dedup content‑addressed layers
```

### 3.3 Mirror to another Mac
```bash
rsync -av ~/.ollama/ user@alpha:~/.ollama/
```

### 3.4 Remove models
```bash
ollama rm llama3:70b-q8_0
```

---

## 4. LM Studio catalog (desktop app)

### 4.1 Where models live (macOS)
```
~/.cache/lm-studio/models/
  └─ TheBloke/
      └─ Llama-3-8B-Instruct/
          └─ llama-3-8b-instruct.Q4_K_M.gguf
```
*(Older docs referenced `~/Library/Application Support/LM Studio/models`; current builds migrated to `~/.cache` for cross‑platform parity.)*

### 4.2 Manual management
```bash
# list
ls ~/.cache/lm-studio/models

# purge a model
rm -rf ~/.cache/lm-studio/models/TheBloke/Llama-3-8B-Instruct
```

### 4.3 CLI (experimental)
```bash
lmstudio models list
lmstudio models download mistral-7b-instruct
```
If CLI subcommands are missing, you're on an older app build—update via app's menu.

---

## 5. Cross‑loading scenarios

| Goal | Approach |
|------|----------|
| Use HF safetensors in **Ollama** | Quantise → GGUF (`convert.py` or llama.cpp) then `ollama create` with Modelfile. |
| Serve Ollama GGUF in **vLLM** | Not supported — vLLM expects safetensors / fp16. |
| Single cache for all Macs | Syncthing or rsync `HF_HOME`, `.ollama`, `.cache/lm-studio/models` to a shared SSD. |

---

## 6. Disk mapping suggestion (16 TB SSD)

```bash
# move caches once, symlink permanently
mv ~/.cache/huggingface /Volumes/Models/hf-cache
ln -s /Volumes/Models/hf-cache ~/.cache/huggingface

mv ~/.ollama /Volumes/Models/ollama
ln -s /Volumes/Models/ollama ~/.ollama

mv ~/.cache/lm-studio/models /Volumes/Models/lmstudio
ln -s /Volumes/Models/lmstudio ~/.cache/lm-studio/models
```

---

## 7. Troubleshooting quick map

| Symptom | Source | Fix |
|---------|--------|-----|
| **403 on HF download** | gated model | `huggingface-cli login` & accept license. |
| **`checksum mismatch` in Ollama** | partial layer download | `ollama pull --force <model>` |
| **LM Studio stuck at 0 %** | network filtering | turn off "Use HTTPS" in settings or VPN. |

---

## 8. Key takeaways

* **Hugging Face**: richest formats + auth.
* **Ollama**: pull‑and‑go GGUF; great Metal performance.
* **LM Studio**: GUI that mirrors HF weights under `~/.cache/lm-studio`.
* Centralise caches on your big SSD, rsync between Macs, and avoid repeated 70 GB downloads.

---

# Why Some Hugging Face Models Can't Be Reliably Converted for Every Backend  
*(and why that's perfectly normal)*  

---

## 1 · The video analogy in one minute

| Video stack | LLM stack |
|-------------|-----------|
| **Codec** (H.265, AV1) – how the frames are encoded | **Quant/architecture** (FP16, Q4_K_M, GQA, MoE) – how tensors are stored |
| **Container** (MP4, MKV) – wrapper that holds codec + metadata | **File format** (Safetensors, GGUF, PT) – wrapper for weights + tokenizer |
| **Player** (VLC, QuickTime) – decodes & displays | **Runtime** (Ollama, LM Studio, llama.cpp, vLLM) – loads & runs tensors |

A "universal video player" still glitches when a brand-new codec appears; it needs time to add **decoders**.  
Likewise, a "universal conversion script" breaks whenever a new model architecture, tensor layout, or quant method lands on Hugging Face before runtimes support it.

---

## 2 · Concrete reasons conversions fail

| Class | How it bites you |
|-------|------------------|
| **Architecture drift** | Ollama & LM Studio only load *llama-family* tensors. Pull a Falcon, Qwen-2, or MPT repo → converter errors ("missing key `rotary_emb.inv_freq`"). |
| **Quantisation spec churn** | GGUF v3 added `Q4_K_M`; v4 may add `BLOCK_K`. If your converter builds v4 but Ollama's embedded llama.cpp only knows v3, runtime exits: "unsupported qtype." |
| **Tokenizer packaging** | GGUF must embed the tokenizer; many HF repos ship only `tokenizer.json`. Converter skips it → model loads but generates gibberish. |
| **Sharded safetensors quirks** | 60-shard checkpoints: if one LFS blob 404s, the converter creates an incomplete GGUF that silently segfaults at runtime. |
| **Runtime/kernel lag** | New grouped-query attention (Llama-3) needs fresh Metal/CUDA kernels. Until Ollama or LM Studio update, the model loads on CPU or crashes. |
| **Modelfile metadata** (Ollama only) | Script converts tensors but doesn't create a chat template → "unknown template" during generation. |

---

## 3 · Why a truly *universal* script is a moving target

To succeed 100 % of the time a script would have to:

1. **Detect architecture** → choose the right converter for llama, Falcon, Phi-3, etc.  
2. **Handle every quant** → GPTQ, AWQ, BLOCK-K, whatever comes next.  
3. **Embed correct tokenizer** or auto-build it.  
4. **Regenerate Modelfiles** for Ollama with the proper system prompt.  
5. **Track GGUF spec bumps** and re-encode older files when runtimes catch up.  

That's effectively re-implementing the combined maintenance work of llama.cpp **and** every runtime that embeds it.

---

## 4 · Practical workflow that *does* work

1. **Download from Hugging Face once** (keep the raw safetensors).  
   ```bash
   huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2 --local-dir ~/hf/mistral-7b
   ```
2. **Use the converter tied to the runtime commit you'll run**.  
   ```bash
   git clone https://github.com/ggerganov/llama.cpp -b 6c4d… ~/llama.cpp
   python convert.py --gguf --outtype q4_km ...
   ```
3. **Validate immediately**:  
   ```bash
   ollama create test -f Modelfile && ollama run test -p "Hello"
   ```
4. **Store both copies**:  
   - `/Models/hf/` – source fp16 shards  
   - `/Models/gguf/` – converted qtypes  
   Re-convert later when specs or runtimes advance.

---

## 5 · When you *must* share one cache across tools

* **Ollama ↔ LM Studio**  
  *Both* understand llama-family GGUF v3+; you can rsync `.ollama` into `~/.cache/lm-studio/models` to avoid re-downloads.  
* **vLLM ↔ Transformers**  
  Share the raw HF cache (`HF_HOME`) because both expect safetensors.

Anything else (e.g., GGUF → vLLM) is a lossy transcode—like trying to play a Blu-ray .m2ts file in a DVD player.

---

## 6 · Bottom line

There **will always be models that can't be reliably converted** the day they drop, for the same reason a brand-new AV1 Level 6.1 video won't play on last year's smart TV: the runtime hasn't shipped the decoder yet.  
Accept the lag, keep raw weights, validate conversions, and reconvert when your runtimes catch up—that's the bandwidth-friendly path that never leaves you stuck.

--- 

## Appendix: Bonus scripts you may try.

The following scripts mostly work with older models. Even newer models with little architecture drift may work. Use them as a source of ideas, not as drop-in replacements. 

- [HF model downloader and tester - huggingface_model_downloader](src/huggingface_model_downloader.py)
- [HF to LMStudio Converter(CLI) - huggingface_2_lmstudio.py](src/huggingface_2_lmstudio.py)
- [HF to LMStudio Converter(Streamlit) - huggingface_2_lmstudio_webui.py](src/huggingface_2_lmstudio_webui.py)
- [HF to GGUF Converter(Streamlit) - huggingface_2_gguf_webui.py](src/huggingface_2_gguf_webui.py)

If you're not sure how to deal with missing dependencies, these scripts are probably not for you. They require Python knowledge and familiarity with package management. Before using them, ensure you have the necessary libraries installed (transformers, streamlit, etc.) and understand how to troubleshoot import errors.

You can use models from LM Studio with Ollama, as both support GGUF format. For Hugging Face models:

1. **Convert HF models to GGUF first** using llama.cpp's converter:
   ```bash
   python ~/llama.cpp/convert_hf_to_gguf.py --outfile model.gguf --outtype q4_k_m ~/path/to/hf/model
   ```

2. **Create an Ollama Modelfile** pointing to the GGUF:
   ```
   FROM /path/to/your/model.gguf
   TEMPLATE "{{.System}}\n\n{{.Prompt}}"
   PARAMETER stop "assistant:"
   ```

3. **Import into Ollama**:
   ```bash
   ollama create mymodel -f Modelfile
   ```

Direct HF→Ollama conversion isn't supported, but this GGUF intermediary approach works reliably. 

---
## Ollama Modelfile & Chat-Template Guide

This appendix explains **why chat templates matter**, how to embed them in an Ollama *Modelfile*, and shows two concrete examples (Phi-4 + QwQ) you can adapt for any llama-family model.

---

## 1. Why chat templates?

* **Instruction vs. raw completion** — Most reasoning models rely on a *prompt scaffold* that labels speaker turns (`user`, `assistant`) and injects special tokens (`<|eot_id|>`). Without the correct wrapper, the model treats every request as a continuation of training data and outputs junk.
* **Consistent stop tokens** — Templates define `stop` parameters so Ollama knows where to cut the response stream.
* **System prompts at the source** — Embedding your system role in the Modelfile means UIs (OpenWebUI, LM Studio) don't need to resend it every call.

Ollama ships default templates for its official models, but **community‑converted GGUFs won't have one**. You supply it in a Modelfile.

---

## 2. Anatomy of a simple Modelfile (Phi-4 example)

```plaintext
# Build a new model named "pippa" based on the upstream phi4 GGUF

FROM phi4              # base weights already pulled via `ollama pull phi4`

SYSTEM """
Your name is Pippa. You are a helpful assistant.
"""

TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""

PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
```

Build & test:
```bash
ollama create pippa -f Modelfile
ollama run pippa -p "Hello!"
```

OpenWebUI will now auto-list `pippa` and inherit the system prompt.

---

## 3. Complex template with tool-calling (QwQ-32B GPT-Q)

Some reasoning models (Qwen/QwQ) expect ChatGPT-style `<|im_start|>` / `<|im_end|>` markers and XML tags for function calls. A shortened Modelfile:

```plaintext
FROM ~/.cache/lm-studio/models/Qwen/QwQ-32B-GGUF/QwQ-32B-q8_0.gguf

TEMPLATE """{{- if or .System .Tools }}<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools
… (full tool JSON / instructions) …
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
  … (user / assistant / tool handling) …
{{- end }}"""

SYSTEM You are a helpful, harmless assistant.
PARAMETER stop <|im_start|>
PARAMETER stop <|im_end|>
```

Build:
```bash
ollama create qwq -f Modelfile
```
Use in OpenWebUI with **Tools** feature enabled—model will emit <tool_call> chunks compatible with function-calling spec.

---

## 4. Tips & gotchas

| Issue | Cause | Fix |
|-------|-------|-----|
| Model rambles with no stops | missing `PARAMETER stop` tokens | add all boundary tokens used in template. |
| OpenWebUI ignores system prompt | UI overrides with its own; uncheck "Send system prompt" or keep it brief in Modelfile. |
| Mixed language output | training data bias; embed language directive in `SYSTEM` section. |
| Template breaks on vLLM | vLLM ignores Modelfile—apply template at client layer instead. |

---

## 5. Workflow cheat-sheet

```bash
# 1. Pull or convert GGUF
ollama pull phi4           # or quantize your own

# 2. Write Modelfile with correct TEMPLATE + stop tokens
# 3. Build
ollama create mymodel -f Modelfile

# 4. Chat
ollama run mymodel -p "Hello"
```

Once baked, the model behaves consistently across Ollama CLI, Ray Serve proxy, and OpenWebUI.

It's usually safest to think of chat templates as **architecture-specific** (or at least "model-family-specific").

### Why grouping by architecture works

| Model family | Common template style | Typical boundary tokens |
|--------------|----------------------|-------------------------|
| **Llama / Mistral / Phi-2 / Phi-4** | `system` / `user` / `assistant` blocks wrapped with `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>` | `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>` |
| **Qwen / QwQ** | ChatGPT-like `<|im_start|>` / `<|im_end|>` markers, plus XML blocks for tool calls | `<|im_start|>`, `<|im_end|>` |
| **Gemma** | Smaller variant of Llama template, but different stop tokens (`<|eot|>`) | `<|eot|>` |
| **Falcon / MPT** | Plain role prefixes (`User:\n`, `Assistant:\n`) with double-newline stop | `\n\n` |

Runtime (Ollama or llama.cpp) doesn't auto-detect which template a GGUF needs; you have to supply one that matches the model's training prompts. If you use the wrong template:

* The model may treat your entire prompt as **assistant** text and refuse to answer.  
* Stop tokens won't be recognised, so the stream never ends or truncates mid-sentence.  
* Tool-calling models won't generate the required XML wrappers.

### Practical rule

> **When you convert a model for Ollama, copy the template block shown on that model's page on <https://ollama.ai> and paste it into your Modelfile.**

If the exact model isn't listed, grab the template from another model in the *same family* (e.g., take `mistral`'s template for any custom Llama-2 quant). Then update:

1. **`FROM`** line to point at your converted GGUF.  
2. **`PARAMETER stop`** lines to match the tokens present in the template.

Do that, rebuild with `ollama create`, and your converted model will behave exactly like its upstream counterpart.

---

[⇧ Back&nbsp;to&nbsp;README](../README.md)