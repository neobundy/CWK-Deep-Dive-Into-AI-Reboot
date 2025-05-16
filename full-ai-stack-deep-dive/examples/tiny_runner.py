#!/usr/bin/env python3
"""
tiny_runner.py — launch llama-server with a single model.

Usage examples
--------------
$ python tiny_runner.py
      # loads ~/.tiny_ollama/gemma-3-27B-it-QAT-Q4_0.gguf on port 12435

$ python tiny_runner.py -m llama3-mini.gguf -p 13000 -t llama-3
      # custom model, port, and chat template
"""
import argparse, pathlib, subprocess, os, sys, shlex

HOME       = pathlib.Path.home()
DEFAULT_MD = HOME / ".tiny_ollama" / "gemma-3-27B-it-QAT-Q4_0.gguf"
DEFAULT_PT = 12435

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model",          default=str(DEFAULT_MD))
parser.add_argument("-p", "--port",  type=int,default=DEFAULT_PT)
parser.add_argument("-t", "--chat-template",  default="gemma",
                   help="pass the name if your model expects one; "
                        "set to '' to omit")
parser.add_argument("--gpu-layers", default="100")
args = parser.parse_args()

cmd = [
    "llama-server", "-m", args.model,
    "--host", "127.0.0.1", "--port", str(args.port),
    "--n-gpu-layers", args.gpu_layers
]
if args.chat_template:
    cmd += ["--chat-template", args.chat_template]

print("launching:", " ".join(shlex.quote(c) for c in cmd), flush=True)
proc = subprocess.Popen(cmd)

try:
    proc.wait()
except KeyboardInterrupt:
    print("\nshutting down …")
    proc.terminate()
    proc.wait()
    sys.exit(0)
