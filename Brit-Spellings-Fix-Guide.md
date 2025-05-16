# Banishing the British Accent
*A lightning guide for purging UK spellings (colour, optimise…) and locking in U.S. variants across the repo.*

---

## 1 · Install Codespell
```bash
pip install codespell
```
Why **Codespell**? It's lightweight, script‑friendly, and plugs into **pre‑commit** & CI with zero friction.

---

## 2 · Build our en‑US Dictionary
Create **`scripts/us‑dialect.txt`** (symlinked, so every repo shares it).
Each line is **british→american** – use `->` *or* `=>`, both are valid.

```text
colour      -> color
colours     -> colors
optimise    -> optimize
optimised   -> optimized
optimising  -> optimizing
optimisation-> optimization
organise    -> organize
organised   -> organized
organising  -> organizing
organisation-> organization
tokenise    -> tokenize
tokenised   -> tokenized
initialise  -> initialize
initialised -> initialized
favourite   -> favorite
behaviour   -> behavior
labour      -> labor
```

> **Heads‑up on flags**  
> * `-D/--dictionary`  → *replace* using the mapping file above.  
> * `-I/--ignore-words`→ *silence* specific tokens (no replacement).  
> * `-L/--ignore-words-list` → inline comma‑separated ignore list.

---

## 3 · Dry‑Run (see, don't touch)
```bash
codespell \
  --builtin clear,informal \
  --dictionary scripts/us-dialect.txt \
  --ignore-words scripts/codespell.ignore \
  --check-filenames --check-hidden \
  -S .git,.specstory, Brit-Spellings-Fix-Guide.md \
  -q 3
```
*What those switches do*
| Flag | Purpose |
|------|---------|
| `--builtin clear,informal` | drop British variants from built‑in dicts |
| `--dictionary` | load our mapping file for auto‑fix suggestions |
| `--check-filenames` | lint file names too |
| `--check-hidden` | include dot‑files |
| `-S` | skip paths (comma‑separated) |
| `-q 3` | quiet except real hits |

> **Heads-up** Running `codespell` on *this* guide will only show the real typos (e.g. the ANE/`re-use` demo). All the obvious British spellings you see in the dictionary table or the quoted passage are either inside fenced code blocks **or** wrapped with the `codespell:disable … enable` pragma, so the linter intentionally skips them. In your own docs those words *will* be flagged unless you use the same escape hatches.

---

## 4 · Auto‑Fix in Place
```bash
codespell -w \
  --builtin clear,informal \
  --dictionary scripts/us-dialect.txt \
  --ignore-words scripts/codespell.ignore \
  --check-filenames --check-hidden \
  -S .git,.specstory, Brit-Spellings-Fix-Guide.md \
  -q 3
```
`-w` writes corrections. Stage ➜ commit ➜ accent gone.

---

## 5 · pre‑commit Gatekeeper
Add **`.pre-commit-config.yaml`**:
```yaml
repos:
  - repo: https://github.com/codespell-project/codespell
    rev: v2.4.0  # check latest
    hooks:
      - id: codespell
        args: [
          --builtin, clear,informal,
          --dictionary, scripts/us-dialect.txt,
          --check-filenames,
          --check-hidden,
          --skip, .git,.specstory, Brit-Spellings-Fix-Guide.md
        ]
```
Then wire it up:
```bash
pip install pre-commit
pre-commit install
```
Push with a stray *optimise*? Hook fails instantly.

---

## 6 · CI Spell‑check (GitHub Actions example)
```yaml
# .github/workflows/spellcheck.yml
name: Spellcheck
on: [push, pull_request]

jobs:
  codespell:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install codespell
      - run: codespell --dictionary scripts/us-dialect.txt -S .git,.specstory -q 3
```
CI lights red if any UK spellings sneak through.

---

## 7 · Ignoring *Legit* Tokens
Some acronyms (e.g. **ANE**) or stylistic picks (e.g. *re-use*) aren't mistakes.

*Global whitelist*
```text
# scripts/codespell.ignore
ANE
re-use
```
Run with:
```bash
codespell --ignore-words scripts/codespell.ignore ...
```

*Inline pragma*
```md
<!-- codespell:disable -->
Quoted passage with colour, organisation, behaviour…
<!-- codespell:enable -->
```

---

## 8 · Quick Grep & Replace
```bash
rg -i '\boptimise\b' --hidden -g '!*.git*'

# macOS BSD‑sed bulk fix
sed -i '' -e 's/colour/color/g' $(rg -l colour)
```

---

### FAQ
**Q: Why two files?**  
Use *dictionary* for words you want converted, *ignore‑words* for words you want completely skipped.

**Q: Will this rename variables?**  
It might. Always eyeball the diff if you have identifiers like `colourScale` in the code.

**Q: Can I keep British spelling in docs?**  
Yes—prefix the word with `!` in the dictionary or wrap the section with `codespell:disable/enable` comments.

---

Happy *optimizing*—never *optimising*. Keep your diffs red‑white‑and‑blue! 🇺🇸

---

Appendix. `brit-spell-fix.sh`

```bash
#!/usr/bin/env zsh
# brit-spell-fix.sh – Run Codespell with our US-dialect rules
#
# Default: dry-run (no writes). Add --fix to actually apply fixes (-w).
# Usage:
#   ./brit-spell-fix.sh [--fix] [paths…]
#   ./brit-spell-fix.sh -h | --help
#
# Examples:
#   # Just see potential corrections (entire repo)
#   ./brit-spell-fix.sh
#
#   # Actually rewrite files in-place
#   ./brit-spell-fix.sh --fix
#
#   # Limit the check to docs/ and src/
#   ./brit-spell-fix.sh docs src
#
set -o errexit -o pipefail

function usage() {
  cat <<'EOF'
Banishing the British Accent – wrapper for Codespell

This helper runs Codespell with the exact arguments recommended in
Brit-Spellings-Fix-Guide.md.

  -h, --help   Show this help text and exit
      --fix    Write changes in-place (-w)
      --       Treat all following args as positional paths for Codespell

Any additional positional arguments are forwarded to Codespell so you can
limit the check to specific files or directories.  If no paths are supplied
Codespell scans the current working directory tree.
EOF
}

# --- Parse options ---------------------------------------------------------
WRITE_FLAG=""   # empty means dry-run
ARGS=()
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;
    --fix)
      WRITE_FLAG="-w"
      ;;
    --)
      shift
      ARGS+=("$@")
      break
      ;;
    *)
      ARGS+=("$arg")
      ;;
  esac
  shift
done

# --- Determine project root (the directory where this script resides) -----
SCRIPT_DIR="$(cd "${0:a:h}" && pwd)"
# Change to the repo root (assumed to be the directory where this script lives)
cd "$SCRIPT_DIR"

# --- Codespell command -----------------------------------------------------
CODESPELL_ARGS=(
  --builtin clear,informal
  --dictionary scripts/us-dialect.txt
  --ignore-words scripts/codespell.ignore
  --check-filenames
  --check-hidden
  -S .git,.specstory,Brit-Spellings-Fix-Guide.md 
  -q 3
)

# Append write flag if requested
if [[ -n "$WRITE_FLAG" ]]; then
  CODESPELL_ARGS=("-w" ${CODESPELL_ARGS[@]})
fi

# Append any user-supplied paths (or fallback to .)
if [[ ${#ARGS[@]} -gt 0 ]]; then
  CODESPELL_ARGS+=(${ARGS[@]})
else
  CODESPELL_ARGS+=(.)
fi

echo "Running: codespell ${CODESPELL_ARGS[@]}" >&2
exec codespell ${CODESPELL_ARGS[@]} 
```