#!/usr/bin/env python3
"""
tiny_chat.py — Gemma-aware REPL for llama-server’s /completion endpoint.

• Assumes the runner is listening on 127.0.0.1:12435
  (override with TINY_PORT env var).
• Automatically wraps your prompt in the <start_of_turn> tags Gemma expects.
• Streams tokens live and prints them as they arrive.
"""

import os, sys, json, httpx

PORT = int(os.getenv("TINY_PORT", 12435))
URL  = f"http://127.0.0.1:{PORT}/completion"

TEMPLATE_USER  = "<start_of_turn>user\n{}\n<end_of_turn>\n"
TEMPLATE_MODEL = "<start_of_turn>model\n"

def stream(prompt: str, n_predict: int = 256):
    payload = {
        "prompt": TEMPLATE_USER.format(prompt) + TEMPLATE_MODEL,
        "stream": True,
        "n_predict": n_predict,
    }
    with httpx.stream("POST", URL, json=payload, timeout=None) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw.startswith("data: "):
                continue
            if raw.strip() == "data: [DONE]":
                break
            token_json = raw[6:]
            yield json.loads(token_json).get("content", "")

def repl():
    print("Connected to runner on", URL)
    while True:
        try:
            user = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            return
        print("assistant> ", end="", flush=True)
        for tok in stream(user):
            sys.stdout.write(tok)
            sys.stdout.flush()
        print()  # newline after each complete answer

if __name__ == "__main__":
    repl()
