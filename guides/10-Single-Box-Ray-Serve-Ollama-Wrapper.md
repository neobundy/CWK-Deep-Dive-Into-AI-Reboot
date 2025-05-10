# Ray‑Serve Ollama Proxy Guide

*(Personal lab notebook — read if it helps; ignore if it doesn't. 🙂 Last verified 2025‑04‑23)*


> Turn **Ollama** into a full OpenAI‑compatible backend using **Ray Serve**, so that any OpenAI‑style client (OpenWebUI, LangChain, etc.) can talk to your local models even while vLLM's Metal support is incomplete.

---

## 1. Why skip vLLM and Docker here?

* **vLLM Metal gap** – CUDA kernels still outperform the experimental Metal path; streaming/tool‑calling features only work reliably on CUDA. Waiting for upstream parity.
* **Ollama already offers fast Metal inference** with baked‑in quantized GGUF; no need to re‑quantize.
* **Docker overhead** unnecessary on a single‑box Apple‑silicon setup; Homebrew service + conda envs keep things simpler.

---

## 2. High‑level flow

```text
Client (OpenWebUI, curl)  ─►  Ray Serve /ray_ollama.py  ─►  Ollama REST (localhost:11434)
```

* Ray Serve exposes the **same paths** `/v1/chat/completions`, `/v1/models`.
* Incoming OpenAI JSON is translated → Ollama `/api/chat` or `/api/models`.
* Results re‑wrapped so the client sees standard OpenAI schema.

---

## 3. Prerequisites

```bash
brew install ollama
sudo brew services start ollama           # ensures Metal backend

conda create -n ollama_proxy python=3.12 -y
conda activate ollama_proxy
pip install ray[serve] fastapi uvicorn requests
```

---

## 4. Save the proxy code

Create `ray_ollama.py`.

Note that even the latest o3 model can't reliably craft a wrapper that works in one shot. Pippa and I refined this script multiple times checking the logs and the output. The JSON response format can be finnicky and should exactly match the output of the OpenAI API. 

If you work with a capable SOTA model for creating a wrapper, it will know what to do. Just don't expect it can do it alone in one go. It would need your help and guidance.

Also note that OpenWebUI is very actively developed. If your LLM generated the wrapper using an older screenshot, the element IDs or API path assumptions may already be outdated. Feed the model a fresh UI screenshot whenever it references missing widgets. Frankly, the UI can feel like a moving target.

```python
"""
ray_ollama.py  –  Ray Serve proxy that turns Ollama into an OpenAI‑compatible backend.

Launch:
    ray start --head
    serve run ray_ollama:app --route-prefix /

Test:
    curl -s http://127.0.0.1:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"qwq:latest","messages":[{"role":"user","content":"Hello"}],"stream":false}' | jq
"""

import os, json, requests, time
from ray import serve
from starlette.responses import JSONResponse, StreamingResponse

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")

@serve.deployment(
    name="ollama_router",
    ray_actor_options={"num_cpus": 4},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)
class OllamaRouter:
    def __init__(self):
        self.chat   = f"{OLLAMA_URL}/api/chat"
        self.models = f"{OLLAMA_URL}/api/models"

    async def __call__(self, request):
        path = request.scope["path"]

        # ─────────────────────────────────────  /v1/chat/completions  ──
        if path.endswith("/chat/completions"):
            body = await request.json()

            # force stream=false if client didn't specify
            stream = body.get("stream", False)
            resp = requests.post(self.chat, json=body, timeout=600, stream=True)

            # ----- non‑stream --------------------------------------------------
            if not stream:
                first_line = next(resp.iter_lines()).decode()
                try:
                    ol = json.loads(first_line)
                except json.JSONDecodeError as e:
                    return JSONResponse({"error": f"Bad Ollama JSON: {e}"}, status_code=500)

                out = {
                    "id": f"chatcmpl-{ol.get('created_at','')}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": ol["model"],
                    "choices": [{
                        "index": 0,
                        "message": ol["message"],
                        "finish_reason": ol.get("done_reason", "stop")
                    }],
                    "usage": {
                        "prompt_tokens": ol.get("prompt_eval_count", 0),
                        "completion_tokens": ol.get("eval_count", 0),
                        "total_tokens": ol.get("prompt_eval_count", 0) + ol.get("eval_count", 0)
                    }
                }
                return JSONResponse(out)

            # ----- stream=true  (Server‑Sent Events) --------------------------
            def sse_events():
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        ol = json.loads(line.decode())
                        delta = ol.get("message", {}).get("content", "")
                        chunk = {
                            "choices": [{
                                "delta": {"content": delta},
                                "index": 0,
                                "finish_reason": None
                            }],
                            "object": "chat.completion.chunk"
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    except json.JSONDecodeError:
                        continue
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse_events(), media_type="text/event-stream")

        # ─────────────────────────────────────  /models  &  /v1/models  ──
        if path in ("/models", "/v1/models"):
            # /api/models may stream NDJSON
            lines = requests.get(self.models, timeout=10).text.strip().splitlines()
            models = []
            for ln in lines:
                try:
                    obj = json.loads(ln)
                    if "models" in obj:          # single JSON blob
                        models = obj["models"]
                        break
                    elif "name" in obj:          # NDJSON line
                        models.append(obj)
                except json.JSONDecodeError:
                    continue

            data = [{"id": m["name"], "object": "model"} for m in models]
            return JSONResponse({"object": "list", "data": data})

        # ─────────────────────────────────────  fallback 404  ───────────
        return JSONResponse({"error": "Not found"}, status_code=404)

# Application object for serve run
app = OllamaRouter.bind()
```
---

## 5. Launch sequence

```bash
# 1. start Ray (single box)
ray start --head --port=6379 --dashboard-port=8265

# 2. deploy the proxy
serve run ray_ollama:app --route-prefix /
```
*Ray Serve prints:* `Uvicorn running on http://127.0.0.1:8000`.

---

## 6. Pull a model and test

```bash
ollama pull phi3

# non‑stream test
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy" \
  -d '{"model":"phi3","messages":[{"role":"user","content":"Hello"}],"stream":false,"max_tokens":64}' | jq
```
Expected JSON reply with `choices[0].message.content`.

* Streaming test: set `"stream": true` – output will be SSE.

---

## 7. Integrate with OpenWebUI

```bash
# assume openwebui env is active
open-webui serve \
  --host 0.0.0.0 --port 3000 \
  --openai-base-url http://localhost:8000 \
  --openai-api-key dummy
```
* UI → Settings → Connections → Add OpenAI provider pointing to `http://localhost:8000`.
* Model list auto‑populates because proxy rewrites `/v1/models` from Ollama's `/api/models`.

---

## 8. Autoscaling knobs

`ray_ollama.py` sets:
```python
@serve.deployment(
    ray_actor_options={"num_cpus": 4},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)
```
* Increase `num_cpus` to throttle per‑replica concurrency.
* Raise `max_replicas` for higher QPS (each replica still calls the single Ollama backend, so gains are modest).

---

## 9. Troubleshooting reference

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| **404 Not Found** at `/v1/chat/completions` | Serve not bound to `/` | ensure `--route-prefix /` in `serve run`. |
| **400 Bad JSON from Ollama** | Proxy hit NDJSON quirks | Verify Ollama returns valid JSON lines; upgrade Ollama. |
| **CORS errors in browser** | Ray Serve default headers | In OpenWebUI use server-side fetch (no browser CORS) or add CORS middleware to proxy. |
| **High latency** | Ollama compiles Metal kernels on first request | Send a dummy prompt after pull; subsequent calls are fast. |

---

## 10. Future swap‑over to vLLM

1. `pip install -U vllm` in a *separate* env.
2. Launch vLLM on port 8000; stop the proxy (`serve delete ollama_router`).
3. Update OpenWebUI provider Base‑URL → done.

---

[⇧ Back&nbsp;to&nbsp;README](../README.md)