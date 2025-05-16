#!/usr/bin/env python3
"""
Interactive Ollama chat (CLI-style) with upfront model warm-up
and a tqdm spinner while loading.

• Pass -m / --model, or choose interactively.
• Shows a live progress bar while the model is pulled into RAM/VRAM.
• Exits on  exit, quit, or /bye  — just like the Ollama CLI.

Requires ollama-python ≥ 0.2.0 and (optionally) tqdm.
"""

from __future__ import annotations
import sys
import time
import argparse
import threading
from ollama import chat, list as list_models

try:
    from tqdm import tqdm
except ImportError:  # graceful fallback
    tqdm = None  # type: ignore

EXIT_WORDS = {"exit", "quit", "/bye"}


# --------------------------------------------------------------------------- #
# Helper: list local models and let user choose                               #
# --------------------------------------------------------------------------- #
def choose_model() -> str:
    models = (list_models()).models or []
    if not models:
        sys.exit("No local models found. Pull one first:  ollama pull <name>")

    print("Available models:\n")
    for idx, m in enumerate(models, 1):
        mb = m.size.real / 1024 / 1024
        print(f"  [{idx}] {m.model:<30}  {mb:>8.1f} MB")

    while True:
        choice = input("\nPick a model (name or number): ").strip()
        if not choice:
            print("Type something.")
            continue
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(models):
                return models[idx - 1].model
            print("Invalid number.")
            continue
        for m in models:
            if m.model == choice:
                return m.model
        print("Model not in list.")


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Streamed Ollama chat.")
    p.add_argument("-m", "--model", help="Model name/tag (e.g. llama3.2)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Helper: warm the model, showing tqdm bar if available                       #
# --------------------------------------------------------------------------- #
def warm_up(model_name: str) -> None:
    if tqdm:
        stop_flag = threading.Event()

        def _progress():
            with tqdm(
                desc=f"Loading {model_name}",
                bar_format="{l_bar}{bar} ⏳",
                total=1000,  # arbitrary large number, we’ll close early
                colour=None,
                leave=False,
            ) as bar:
                while not stop_flag.is_set():
                    bar.update(1)
                    time.sleep(0.02)  # 50 FPS for a smooth bar

        t = threading.Thread(target=_progress, daemon=True)
        t.start()
    else:
        print(f"\n⏳ Loading **{model_name}** …", flush=True)

    t0 = time.perf_counter()

    # Dummy request with num_predict=0 forces the load without output.
    _ = chat(
        model=model_name,
        messages=[{"role": "user", "content": "warm-up"}],
        stream=False,
        options={"num_predict": 0},
    )

    elapsed = time.perf_counter() - t0

    if tqdm:
        stop_flag.set()
        t.join()
        # Ensure we leave the progress bar line clean
        print(f"\r✅ {model_name} ready in {elapsed:.1f}s.{' ' * 20}")
    else:
        print(f"✅ Model ready in {elapsed:.1f}s.\n")


# --------------------------------------------------------------------------- #
# Main loop                                                                   #
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_cli()
    model_name = args.model or choose_model()
    warm_up(model_name)

    print("\nType your messages below — exit, quit, or /bye to leave.\n")
    messages: list[dict[str, str]] = []

    while True:
        try:
            user_text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGood-bye!")
            break

        if user_text.lower() in EXIT_WORDS:
            print("See you next time!")
            break

        messages.append({"role": "user", "content": user_text})

        stream = chat(model=model_name, messages=messages, stream=True)

        reply_parts: list[str] = []
        print()  # blank line before assistant types

        for chunk in stream:
            part = chunk["message"]["content"]
            reply_parts.append(part)
            sys.stdout.write(part)
            sys.stdout.flush()

        print()  # newline after assistant finishes
        messages.append({"role": "assistant", "content": "".join(reply_parts)})


if __name__ == "__main__":
    main()
