import os
import subprocess
from pathlib import Path

import streamlit as st

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------


def list_hf_models(cache_dir: Path) -> list[Path]:
    """Return HF model dirs found at the top level of the cache."""
    return sorted(p for p in cache_dir.iterdir() if p.name.startswith("models--"))


def latest_snapshot(model_dir: Path) -> Path | None:
    """Return the snapshot dir pointed to by refs/main (or newest)."""
    ref_main = model_dir / "refs" / "main"
    if ref_main.exists():
        sha = ref_main.read_text().strip()
        snap = model_dir / "snapshots" / sha
        return snap if snap.exists() else None

    snaps = sorted(
        (model_dir / "snapshots").iterdir(), key=lambda p: p.stat().st_mtime
    )
    return snaps[-1] if snaps else None


def shards_present(snapshot: Path) -> bool:
    """True if at least one *.safetensors file exists in the snapshot."""
    return any(p.suffix == ".safetensors" for p in snapshot.iterdir())


def convert_to_gguf(snapshot: Path, outfile: Path, outtype: str, llama_cpp: Path) -> str:
    """Run convert_hf_to_gguf.py and return a status message."""
    script = llama_cpp / "convert_hf_to_gguf.py"
    if not script.exists():
        return f"convert_hf_to_gguf.py not found in {llama_cpp}"

    outfile.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        str(script),
        str(snapshot),
        "--outfile",
        str(outfile),
        "--outtype",
        outtype,
    ]

    try:
        subprocess.run(cmd, check=True)
        return f"✅  GGUF saved to\n{outfile}"
    except subprocess.CalledProcessError as exc:
        return f"❌  Converter exited with code {exc.returncode}"


# ------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------

st.title("HF → GGUF Converter (local cache)")

home = Path.home()
HF_CACHE = home / ".cache" / "huggingface" / "hub"

st.sidebar.header("Paths")

# llama.cpp repo location (editable)
default_llama_cpp = home / "llama.cpp"
llama_cpp_dir = Path(
    st.sidebar.text_input("Path to llama.cpp repo", str(default_llama_cpp))
).expanduser()

st.sidebar.text_input("HF cache (read-only)", str(HF_CACHE), disabled=True)

# GGUF dtype
dtype = st.sidebar.selectbox(
    "GGUF dtype",
    ["f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"],
    index=1,
)

# Output base directory (editable)
out_base_default = home / "GGUF"
out_base = Path(st.sidebar.text_input("Output directory", str(out_base_default))).expanduser()

# ------------------------------------------------------------------

model_dirs = list_hf_models(HF_CACHE)
if not model_dirs:
    st.error("No models in the HF cache.")
    st.stop()

model_dir = Path(st.selectbox("Choose cached model", model_dirs))

snapshot = latest_snapshot(model_dir)
if snapshot is None:
    st.error("No snapshot found for this model.")
    st.stop()

if not shards_present(snapshot):
    st.warning(
        "Snapshot contains no *.safetensors weight shards.\n"
        "Run the model once (or download shards) before converting."
    )
    st.stop()

# Compose output filename
model_name = model_dir.name.replace("models--", "").replace("--", "/")
org, name = model_name.split("/")
outfile = out_base / org / f"{name}-GGUF" / f"{name}-{dtype}.gguf"

st.success(f"Snapshot: {snapshot.name}")
st.text_input("GGUF will be written to", str(outfile), disabled=True)

if st.button("Convert to GGUF"):
    with st.spinner("Converting …"):
        message = convert_to_gguf(snapshot, outfile, dtype, llama_cpp_dir)
    if message.startswith("✅"):
        st.success(message)
    else:
        st.error(message)
