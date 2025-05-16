# Mini Troubleshooting Guide: Installing `sentencepiece` on Apple Silicon (M3 Ultra, Clean Setup)

## Problem
When attempting to `pip install` packages like `mlx-lm`, installation can fail at the `sentencepiece` dependency due to missing C++ build tools on a fresh Mac setup.

Typical error:
```plaintext
Failed to build sentencepiece
```

This happens because `pip` tries to compile `sentencepiece` from source, which requires:
- Clang (C++ compiler)
- CMake
- Proper developer headers

Fresh macOS installs often lack these by default.

---

## Quick Solution (Best Practice)
**Use Conda to install a pre-built binary of `sentencepiece` from conda-forge instead of building from source.**

### Step-by-Step

1. **Install `sentencepiece` via Conda:**

```bash
conda install -c conda-forge sentencepiece
```

2. **Then install your target package (e.g., `mlx-lm`) via pip:**

```bash
pip install mlx-lm
```

✅ `sentencepiece` will be already available, so `pip` won't try to build it again.

---

## Why This Works
| | pip build | conda-forge binary |
|:---|:---|:---|
| Requires compiler | Yes (clang++, cmake) | No |
| Risk of failure | High on clean Macs | Very low |
| Install speed | Slow | Fast |
| Resource usage | High | Minimal |

Conda-forge provides pre-compiled, architecture-optimized binaries—no C++ build required.

---

## General Rule for Apple Silicon Users
When a Python package has C++ components (e.g., `sentencepiece`, `faiss`, `xformers`):
1. Search for a Conda package on `conda-forge` first.
2. Install it via Conda.
3. Then proceed with your `pip install`.

This saves time, avoids errors, and keeps your environment clean.

```bash
# Good habit template:
conda install -c conda-forge <package>
pip install <main-package>
```

---

## TL;DR
- `sentencepiece` fails to build? Install it via Conda first.
- Always prefer Conda-forge binaries for C++ heavy packages on Mac.

Stay smooth, skip the compiler headaches! 🚀

---

[⇧ Back&nbsp;to&nbsp;README](../../README.md)