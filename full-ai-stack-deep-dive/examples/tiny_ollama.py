# tiny_ollama.py – runner + OpenAI facade on port 12434
import os, subprocess, uvicorn, httpx
from fastapi import FastAPI, Request
from sse_starlette.sse import EventSourceResponse

RUNNER = subprocess.Popen(["python", "tiny_runner.py"])  # spawns on 12435
app = FastAPI()

@app.post("/v1/chat/completions")
async def chat(req: Request):
    data = await req.json()
    prompt = "".join(f"{m['role']}: {m['content']}" for m in data["messages"])
    payload = {"prompt": prompt, "stream": True, "n_predict": data.get("max_tokens", 128)}
    async def gen():
        async with httpx.AsyncClient(timeout=None) as cli, cli.stream("POST", "http://127.0.0.1:12435/completion", json=payload) as r:
            async for ln in r.aiter_lines():
                if ln.startswith("data: "):
                    tok = ln[6:]
                    yield {"event": "data", "data": f'{tok}'}
    return EventSourceResponse(gen())

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=12434)