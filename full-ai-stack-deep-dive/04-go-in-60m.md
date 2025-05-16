# Chapter 4 · Go in 60 Gentle Minutes — A Practical Primer for the Curious Engineer

*(Personal lab notebook – last verified 2025‑05‑12)*

> **Why are we here?**
> You don't have to be a Go guru, but the Ollama wrapper we'll open in Chapter 5 is written in Go. A bit of fluency now will spare you hours of head‑scratching later.
> **Promise** — By the end of this chapter you'll be comfortable finding your way around Ollama's source—enough to tweak a flag and rebuild—but true mastery still comes from practice.

![Go](images/04-go-title.png)

---

## 1 · What's with Go, anyway?

Go (often called "Golang") was created at Google to build fast, reliable server software without the headaches of C++ or the runtime baggage of Java. The language is ruthlessly pragmatic—minimal syntax, near‑instant compilation, and a batteries‑included standard library.

### 1.1 Why Go matters in this AI stack

* **Simplicity & speed** — Readable code that compiles to a single static binary. No dependency hell, no "works on my machine" drama.
* **Concurrency, built‑in** — Goroutines and channels make it trivial to juggle many tasks at once—perfect for servers dispatching model jobs.
* **First‑class tooling** — Formatter, linter, test runner, and dependency manager are all built‑in and consistent across platforms.
* **Deployment super‑powers** — One binary runs anywhere—Linux, macOS, Windows, ARM, x86—so many cloud‑native tools (Docker, Kubernetes, Terraform, and yes, Ollama) are written in Go.

---

### 1.2 Ollama at a glance *(assume you're at the repo root)*

#### 1.2.1 CLI executable

1. **Entry point** — `main.go` at the repo root. ([view on GitHub](https://github.com/ollama/ollama/blob/main/main.go))
2. **Build scripts** — Platform helpers live in `/scripts` (`build_darwin.sh`, `build_linux.sh`, etc.).
3. **Compilation** — Go's toolchain (with **CGO**) builds Go and C/C++ bits, including patched `llama.cpp`.
4. **GPU switches** — Platform‑specific flags enable Metal, CUDA, or CPU paths.
5. **Assets bundled** — Templates and static files are embedded into the final binary.
6. **Sub‑commands** live under `/cmd`.

#### 1.2.2 Server

1. Ships inside the same binary—start it with `ollama serve`.
2. HTTP routes live in `/server/routes.go`.
3. Requests/types are defined in `/api`; model execution flows through `/llm` and `/runner`, which call into `llama.cpp`.

#### 1.2.3 Desktop app

1. UI code is in `/app`.
2. `/app/main.go` boots the UI and embeds the server process.
3. **macOS** uses an Electron wrapper in `/macapp` (see `macapp/src/index.ts`).
4. **Windows** manages the server with OS‑specific lifecycle files such as `/app/lifecycle/server_windows.go` 
5. Platform builds package everything appropriately (.app bundle, Windows installer, etc.).

We'll tour that code in Chapter 5; for now, keep in mind that a pinch of Go knowledge unlocks the whole toolbox.

We installed the Go toolchain back in Chapter 1:

```bash
brew install go cmake
xcode-select --install   # Metal SDK
```

With those in place you can clone Ollama, build from source, and follow the examples ahead.

Ready? Let's dive in—gently.

---

## 2 · A Quick Orientation — What Makes Go *Go, Go, Go!?*

* **Compiled like C, batteries like Python** — One `go build` spits out a self-contained binary. No virtual envs, no shared-object hunts.
* **First-class concurrency** — Goroutines and channels stand in for threads, mutex gymnastics, and callback pyramids.
* **A "boring" standard library** — Networking, JSON, HTTP, templates, testing—ready-to-use, zero external deps.

Those three traits explain why Ollama's team chose Go for the control plane while leaving tensor math to C/C++.

**Goroutines** – Lightweight coroutines you launch with the keyword `go`. They're cheaper than OS threads, so spinning up thousands is routine.

**Mutexes** – The `sync` package offers classic mutual-exclusion locks to guard shared data and prevent race conditions.

**Callbacks** – Go favors channels and goroutines for async work, but you can still pass functions around when you need custom hooks.

Together, these features deliver powerful concurrency without tangled callback stacks or heavy thread management.

---

### 2.1 · When *Not* to Use Go

Go excels at servers, CLIs, and cloud plumbing, but it's not a silver bullet. Reach for something else if:

* **You need a polished GUI** – Go's desktop toolkits lag behind Swift, C#, or even Python's Qt/Tkinter.
* **You're writing high-performance kernels** – For SIMD-tuned math, C/C++ or Rust will outrun Go, which is why Ollama offloads tensor math to `llama.cpp`.
* **You want rapid scripting** – Go is compiled and statically typed; Python, Ruby, or JavaScript feel nimbler for one-off scripts.
* **You rely on heavy metaprogramming** – Generics arrived in Go 1.18, but they're intentionally restrained; Rust or C++ offers deeper type-level wizardry.
* **You're building mobile apps** – Go cross-compiles, yet native Swift/Kotlin—or Flutter/React Native—usually yields smoother results.
* **You must live inside a legacy runtime** – Plugging straight into Java, .NET, or Node.js is simpler if you stay in those ecosystems.

Right tool, right job. Go is your friend for infrastructure; elsewhere, pick the specialist and move on.

---

## 3 · Your First Go Project — Step by Step

We'll build and run a ten-line program, pausing after **every** command to note what changed on disk.

### 3.1 · Create a folder and initialize a module

```bash
mkdir hello-go && cd hello-go      # a blank folder
go mod init demo                   # ① creates go.mod
```

| What appears on disk                         | Why it matters                                                                                                                     |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **go.mod** with the first line `module demo` | Go now treats this directory as one cohesive module. Every future `go build` or `go run` reads **go.mod** to resolve dependencies. |

```text
module demo

go 1.24.3
```

**Gentle tip — module names**
Throw-away code? `demo` is fine.
Planning to publish? Use a fully-qualified path such as `github.com/cwk/ollama-hacks`; that string later becomes your import path.

---

### 3.2 · Write and run the tiniest program

Create **main.go**:

```go
package main            // every executable starts in package main
import "fmt"            // fmt = formatted I/O helpers

func main() {
    fmt.Println("Hello, Go world!")
}
```

Run it:

```bash
go run main.go          # ② compile & execute
```

| Step | What happens under the hood                                                                                                                             |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ②    | The compiler reads **go.mod**, sees only standard-library imports, builds a temporary binary, runs it, then deletes the binary. The folder stays clean. |

*Heads-up — typos happen*
Mistype the import (`"fm"`) and you'll see `cannot find package "fm"` with an exact file-and-line pointer. Fix and rerun.

---

### 3.3 · Tidy up after edits

Whenever you add or remove imports:

```bash
go mod tidy            # sync go.mod & go.sum with reality
```

`go mod tidy` drops unused dependencies and pins exact versions for new ones—the Go equivalent of "`pip freeze` + cleanup" in one command.

**What is `go.sum`?**

When you use Go modules, your project directory will often contain two files: `go.mod` and `go.sum`.

- **`go.mod`** records the module's name and the specific versions of dependencies your code directly requires.
- **`go.sum`** is an automatically maintained checksum file. It lists cryptographic hashes for every version of every dependency your project (and its dependencies) use.

**Why does `go.sum` exist?**

- It ensures the integrity and security of your builds. When you (or anyone else) run `go mod tidy`, `go build`, or `go get`, Go checks that the downloaded modules match the expected hashes in `go.sum`.
- If a dependency is tampered with upstream, the hash won't match, and Go will refuse to build—protecting you from supply chain attacks.

**Should you commit `go.sum` to version control?**

- **Yes.** Always commit both `go.mod` and `go.sum`. This guarantees that anyone cloning your repo gets the exact same, verified dependencies.

**What if you delete `go.sum`?**

- Go will regenerate it the next time you run a module command, but you might lose the guarantee that everyone is building against the same dependency tree and checksums.

**Summary Table**

| File      | Purpose                                                      | Commit to VCS? |
|-----------|--------------------------------------------------------------|:-------------:|
| `go.mod`  | Declares your module name and direct dependencies            |      ✅       |
| `go.sum`  | Stores checksums for all dependencies (direct & transitive)  |      ✅       |

*In short: `go.sum` is your project's tamper-evidence seal for dependencies. Don't ignore it!*

---

## 4 · A Functional Go Program

Time to do a bit more than print *Hello*. We'll write a tiny tool that:

1. Accepts a `-name` flag.
2. Greets that name (or falls back to "world").
3. Returns a clean status code so you can chain it in shell scripts later.
4. Prints the greeting to standard output.

### 4.1 · Add a new file

Create **main.go** (overwriting the previous one is fine):

```go
package main

import (
    "flag" // standard-lib command-line flags
    "fmt"
    "os"
)

// greet returns the string we want to print.
// Keeping it as a function makes unit-testing painless.
func greet(who string) string {
    if who == "" {
        who = "world"
    }
    return fmt.Sprintf("Hello, %s!", who)
}

func main() {
    // ① define a string flag called -name
    nameFlag := flag.String("name", "", "name to greet")
    flag.Parse() // ② parse os.Args and populate nameFlag

    // ③ do the work
    msg := greet(*nameFlag)
    fmt.Println(msg)

    // ④ exit 0 on success (explicit for clarity)
    os.Exit(0)
}
```

### 4.2 · Create a module and run it

```bash
go mod init demo
```

You'll see a new `go.mod` appear. That tells Go "everything in this directory belongs to one project."

```bash
go run .                 # default: no flag
# → Hello, world!

go run . -name Pippa     # with flag
# → Hello, Pippa!
```

### 4.3 · What just happened?

| Step | Under the hood                                                                                              |
| ---- | ----------------------------------------------------------------------------------------------------------- |
| ①    | `flag.String` registers a `-name` option with a default `""`.                                               |
| ②    | `flag.Parse()` reads `os.Args`, fills `nameFlag`, and strips parsed flags so `flag.Args()` holds leftovers. |
| ③    | We call `greet`, a plain function that returns a string—easy to test.                                       |
| ④    | `os.Exit(0)` makes success explicit; returning a non-zero code (e.g., `os.Exit(1)`) signals failure.        |

Small pieces, but together they cover most patterns you'll meet in Ollama's CLI. Next up: reading environment variables and spawning subprocesses—the bread-and-butter of its control plane.

---

## 5 · Environment Variables & Subprocesses — The Control-Plane Workhorses

Ollama's CLI juggles paths, models, and GPU flags by reading environment variables and then firing off helper binaries like `llama-server`. Let's build a postcard-sized demo that does both tasks.

### 5.1 · The Goal

* Read `OLLAMA_MODEL_PATH` (or fall back to `~/models/default.gguf`).
* Launch a dummy worker process (`echo`) that prints the chosen path.
* Propagate any error codes so your shell scripts behave predictably.

### 5.2 · The Code

Create **main.go**:

```go
package main

import (
    "fmt"
    "os"
    "os/exec"
    "path/filepath"
)

func main() {
    // ① Pick up an env var with a fallback
    modelPath := os.Getenv("OLLAMA_MODEL_PATH")
    if modelPath == "" {
        modelPath = filepath.Join(os.Getenv("HOME"), "models/default.gguf")
    }

    fmt.Println("Using model:", modelPath)

    // ② Prepare the subprocess
    cmd := exec.Command("echo", "Spawning a worker for", modelPath)
    cmd.Stdout = os.Stdout   // forward child stdout to our stdout
    cmd.Stderr = os.Stderr   // same for stderr

    // ③ Run and propagate exit status
    if err := cmd.Run(); err != nil {
        // exec failed or child returned non-zero
        fmt.Fprintln(os.Stderr, "worker error:", err)
        // If it's an ExitError, use its code; otherwise exit 1
        if exitErr, ok := err.(*exec.ExitError); ok {
            os.Exit(exitErr.ExitCode())
        }
        os.Exit(1)
    }
}
```

### 5.3 · Create a module and run it

```bash
go mod init demo
```

```bash
# Default path (env var unset)
go run .
# Using model: /Users/<you>/models/default.gguf
# Spawning worker for /Users/<you>/models/default.gguf

# Override with an env var
OLLAMA_MODEL_PATH=/tmp/custom.gguf go run .
# Using model: /tmp/custom.gguf
# Spawning worker for /tmp/custom.gguf
```

### 5.4 · Why this matters

| Line | Purpose                                                                                                              |
| ---- | -------------------------------------------------------------------------------------------------------------------- |
| ①    | `os.Getenv` lets config flow in from the outside—no recompiles for path tweaks.                                      |
| ②    | `exec.Command` is Go's thin, cross-platform wrapper around fork/exec. You map stdout/stderr, so logs stay unified.   |
| ③    | By forwarding the child's exit code, your wrapper behaves just like the tool it spawns—vital for scripted pipelines. |

That's essentially what Ollama's control plane does at scale: read a few env switches (`OLLAMA_ORCHESTRATE_GPU`, `OLLAMA_CACHE_DIR`, …) and spin up the right `llama.cpp` variant with those settings. Now you've seen the core trick in 40 lines. Onward!

>**Sidebar · "Script" vs. Executable — Two Ways to Run Go**

* **`go run .`**
  Compiles the code to a temporary file, runs it, then deletes the binary. Perfect for quick experiments or notes—no installation fuss, even if you haven't tidied the module yet.

* **`go build -o mytool`**
  Produces a stand-alone binary you can copy anywhere. Subsequent runs are instant because the compile step is already done. Need a different platform? Prepend two variables and cross-compile in one line:
  `GOOS=linux GOARCH=arm64 go build -o mytool`

**Rule of thumb:** Reach for `go run` while you're still poking at ideas; switch to `go build` when the tool needs a permanent home in `/usr/local/bin` or on someone else's machine.

---

## 6 · A Friendly First Goroutine

```go
package main
import (
    "fmt"
    "time"
)

func main() {
    done := make(chan bool)          // channel signals when work is done

    go func() {                      // **go** keyword launches a goroutine
        time.Sleep(time.Second)
        fmt.Println("token 1")
        done <- true                 // send signal
    }()

    <-done                            // main blocks here until signal arrives
}
```

Run with `go run goroutine.go`. You'll see **token 1** after one second—no threads, no mutexes, just plain code.

*Reflection* – Ollama launches each runner with a goroutine, then cancels it via a channel when the HTTP request ends. Same primitives, bigger scale.

---

## 7 · Context — Graceful Cancellation in One Idiom

> **Pain-point first:**
> Long-running AI calls can chew GPU memory for minutes. If the browser tab closes or the mobile app drops off Wi-Fi, you want the backend to stop work *immediately*—not keep burning tokens (and watts) in a zombie process.

Go solves that with a tiny, pervasive construct called **`context.Context`**.

---

### 7.1 · The Canonical Pattern

```go
import (
    "context"
    "net/http"
    "time"
)
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()  // ① guarantee resources are released

req, _ := http.NewRequestWithContext(ctx, "GET", url, nil)
resp, err := http.DefaultClient.Do(req)  // ② auto-cancels after 5 s
```

#### Line-by-line

1. **Create a cancellable context**
   `context.WithTimeout` returns two things:
   *a derived context* (`ctx`) and *a cancel function* (`cancel`).
   After 5 seconds—or earlier if you call `cancel()`—`ctx` is marked "done."

2. **Pass the context downstream**
   Instead of `http.NewRequest`, you use `NewRequestWithContext`, embedding `ctx`. Every library that receives the request can call `ctx.Err()` to see if it's been cancelled and bail out early.

3. **Tidy up with `defer`**
   The `defer cancel()` line means: "Whether this function returns normally, errors, or panics, make sure pending timers and goroutines tied to this context are released." It's one of Go's best safety nets.

---

### 7.2 · Why Ollama Leans on `Context`

* **Fast aborts on client disconnects** – Each `/v1/chat/completions` handler attaches the request's context to the worker that feeds tokens back to the socket. If the user closes the connection, Go's HTTP server cancels the context. The runner sees `ctx.Err() == context.Canceled`, frees KV-cache, and exits within microseconds. No orphaned GPU kernels.
* **Time-boxed speculative tasks** – When Ollama spawns background jobs—embedding generation, cache refresh, speculative decoding—the parent hands down a context with a deadline. Miss it, and the job tears itself down cleanly instead of lingering half-finished.
* **Uniform plumbing** – Whether the work unit is a Metal shader, a CUDA kernel, or plain CPU math, the cancellation signal flows through the same narrow interface: `ctx.Done()` channel plus a quick error check. Less boilerplate, fewer bespoke flags.

---

### 7.3 · Common Gotchas (and fixes)

* **Forgetting `defer cancel()`** – Without it, timer resources leak if the function returns before the deadline. Always pair `WithTimeout` or `WithCancel` with a `defer cancel()`.
* **Ignoring `ctx.Err()` in worker loops** – Inside a long `for` loop, sprinkle `select { case <-ctx.Done(): return ctx.Err(); default: /* work */ }` to break out fast.
* **Storing Context in struct fields** – Treat contexts like request-scoped values, not global state. Pass them as arguments; don't stash them for later.

---

**Checkpoint:** You now know how a two-line idiom (`ctx, cancel := …; defer cancel()`) lets Go programs shed work instantly when the outside world says "stop." That single pattern powers Ollama's ability to keep GPUs busy only while the user is still listening.

---

## 8 · Re‑writing our Python *tiny_runner* in Go (≤30 LoC)

```go
// tiny_runner.go – minimal launcher for llama-server
package main
import (
    "flag"
    "os"
    "os/exec"
)

func main() {
    model := flag.String("model", os.Getenv("HOME")+"/.tiny_ollama/gemma-3-27B-it-QAT-Q4_0.gguf", "gguf path")
    port  := flag.String("port", "12435", "listen port")
    flag.Parse()

    cmd := exec.Command("llama-server",
        "-m", *model,
        "--host", "127.0.0.1", "--port", *port,
        "--n-gpu-layers", "100",
        "--chat-template", "gemma",
    )
    cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
    cmd.Run()                        // blocks until llama-server exits
}
```

Compile once:

```bash
go mod init tiny_runner      
go build tiny_runner.go      # outputs ./tiny_runner
```

Run:

```bash
./tiny_runner -model ~/.tiny_ollama/gemma-3-27B-it-QAT-Q4_0.gguf -port 12435
```

*Checkpoint* – You just replaced the Python launcher with a 3 MB static binary. No interpreter needed on the target machine.

---

## 9 · Checkpoint

You've just covered the essentials: modules, flags, env vars, subprocesses, and graceful cancellation. With that toolkit, Ollama's Go wrapper will read like approachable prose, not black magic. But, deep dives will still take patience.

Remember—your mission isn't to rewrite Ollama; it's simply to *understand* how its pieces snap together.

Take a breath, stretch your fingers, and get ready to "pet" our Ollama in the next chapter. 🐑💻

---

[⇧ Back to README](../README.md)
