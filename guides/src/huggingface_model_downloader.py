#!/usr/bin/env python
"""
huggingface_model_downloader.py  (Metal-only version)
-----------------------------------------------------
Downloads (if needed) and smoke-tests any HF causal-LM on macOS / Apple Silicon.
The script places the entire model on the **M-series GPU (mps)** if possible,
then falls back to CPU FP16.

Example
-------
python huggingface_model_downloader.py Qwen/QwQ-32B \
        "Explain quantum tunneling in one sentence."
"""

from __future__ import annotations
import os, sys, argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------
# Prompt templates (add more as you need)
# ---------------------------------------------------------------------
TEMPLATES = {
    "llama": (
        "<|im_start|>system<|im_sep|>\n{sys}\n<|im_end|>\n"
        "<|im_start|>user<|im_sep|>\n{usr}\n<|im_end|>\n"
        "<|im_start|>assistant<|im_sep|>"
    ),
    "qwen": (
        "<|im_start|>system\n{sys}\n<|im_end|>\n"
        "<|im_start|>user\n{usr}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "gemma": "{sys}\n\n### USER:\n{usr}\n\n### ASSISTANT:",
    "phi":   "{sys}\nUser: {usr}\nAssistant:"
}

DEFAULT_SYSTEM = "You are Pippa, the user's teenage daughter."
DEFAULT_USER   = "hi, pippa!"
MAX_NEW        = 256      # new tokens to generate

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def pick_template(repo_name: str) -> str:
    n = repo_name.lower()
    if "qwen"  in n: return TEMPLATES["qwen"]
    if "gemma" in n: return TEMPLATES["gemma"]
    if "phi"   in n: return TEMPLATES["phi"]
    return TEMPLATES["llama"]

def model_size_gb(model) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3

def load_model(repo: str):
    """Always try MPS first; otherwise fallback to CPU (FP16)."""
    args = {"trust_remote_code": True}

    if torch.backends.mps.is_available():
        print("› Loading on Apple Silicon GPU (mps)")
        args.update(device_map={"": "mps"}, torch_dtype=torch.float16)
    else:
        print("› Apple GPU not available → CPU FP16")
        args.update(device_map="cpu", torch_dtype=torch.float16)

    return AutoModelForCausalLM.from_pretrained(repo, **args)

def run_chat(repo: str, user_msg: str):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = load_model(repo)
    model.eval()

    dtype = next(model.parameters()).dtype
    print(f"\nModel dtype  : {dtype}")
    print(f"Size in mem  : {model_size_gb(model):.2f} GB")
    print(f"Param count  : {sum(p.numel() for p in model.parameters())/1e9:.2f} B")

    prompt = pick_template(repo).format(sys=DEFAULT_SYSTEM, usr=user_msg)
    print("\nPrompt:\n" + prompt)

    inputs = tokenizer(prompt, return_tensors="pt")
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    full = tokenizer.decode(out[0], skip_special_tokens=True)
    reply = full.split(user_msg)[-1].strip()
    print("\nAssistant:\n" + reply)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download + test an HF causal-LM on Metal")
    parser.add_argument("model_id", help="HF repo ID or local snapshot path")
    parser.add_argument(
        "prompt",
        nargs="?",
        default=DEFAULT_USER,
        help="User prompt (default: 'hi, pippa!')",
    )
    args = parser.parse_args()

    run_chat(args.model_id, args.prompt)
