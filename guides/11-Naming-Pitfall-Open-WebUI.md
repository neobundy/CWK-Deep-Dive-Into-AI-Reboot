# Naming Pitfall — **open‑webui** vs. **openwebui**  
*(Personal lab notebook — read if it helps; ignore if it doesn't. 🙂 Last verified 2025‑04‑23)*

> TL;DR  The **only** officially supported package and CLI name going forward is `open-webui`. Historical docs, gists, and blog posts may show `openwebui` (no dash) or other permutations. Treat them as synonyms, but **always use the dashed form in new scripts**.

As mentioned earlier, your LLM will often hallucinate outdated details about **Open WebUI**. When that happens, give it a fresh screenshot of the current interface so it can realign with any renamed widgets or layout changes.

As noted previously, OpenWebUI is very actively developed. Not just naming changes but also UI changes can trip you due to the fast pace of development. You might see multiple updates in a single week, for instance. 

And trust me—your fingers will instinctively type **openwebui** due to 'openai' muscle memory; mentally remap that muscle memory to **open‑webui** or be prepared for endless typos.

---

## 1 · Quick reference table

| Context | Correct today | Legacy variant(s) | Notes |
|---------|---------------|-------------------|-------|
| PyPI package | `open-webui` | `openwebui` | Wheels were republished under dashed name in v0.2.14. |
| Git install | `git+https://github.com/open-webui/open-webui@…#egg=open-webui` | `…#egg=openwebui[server]` | Replace egg fragment with `open-webui`; extras tag `[server]` is now implicit. |
| CLI binary | `open-webui` | `openwebui`, `open-webui serve` vs `openwebui start` | Sub‑command renamed: `serve` (dashed) supersedes `start`. |
| Process grep | `pkill -f open-webui` | `pkill -f openwebui` | Adjust monitoring scripts. |

---

## 2 · Why the rename happened

1. **Namespace collision** – PyPI already had unrelated packages named `openwebui`; adding a dash avoided confusion.
2. **Unix CLI convention** – Multi‑word commands typically use hyphens (`terraform`, `docker‑compose`).
3. **Extras cleanup** – The old `[server]` extras tag is now bundled; plain `pip install open-webui` pulls all you need.

---

## 3 · Practical migration tips

### 3.1 Shell alias for muscle‑memory
```zsh
# ~/.zshrc
aalias openwebui='open-webui'
```
Reload with `source ~/.zshrc` — legacy commands keep working locally.

### 3.2 Requirements.txt pin
```text
open-webui>=0.2.14
```
Explicit pin prevents pip from resolving the stale `openwebui` name.

### 3.3 One‑liner search/replace (ripgrep + sed)
```bash
rg -l 'openwebui start' guides | xargs sed -i '' 's/openwebui start/open-webui serve/g'
```
Run in repo root whenever you paste older snippets.

---

## 4 · CI lint (optional)
Add a tiny GitHub Action to fail the build if `openwebui` sneaks back in:
```yaml
name: Naming guard
on: [push, pull_request]
jobs:
  grep:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: |
          if grep -R --line-number --quiet 'openwebui' guides README.md; then
            echo "Legacy command detected — use open-webui" && exit 1
          fi
```

---

## 5 · Common Q&A

| Question | Answer |
|----------|--------|
| *My older env still has `openwebui --help`* | Uninstall with `pip uninstall openwebui` then reinstall `open-webui`. |
| *Can I still use `[server]` extras?* | Not necessary; `open-webui` bundles server deps by default. |
| *Docker image tags?* | Docker Hub moved to `open-webui:<tag>`; `openwebui:<tag>` is an unofficial mirror. |

---

### Bottom line

When in doubt, **dash it**: `open-webui`.

---

[⇧ Back to README](../README.md)
