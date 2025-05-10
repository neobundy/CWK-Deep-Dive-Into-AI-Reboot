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
