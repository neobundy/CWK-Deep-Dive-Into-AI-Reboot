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
    cmd := exec.Command("echo", "Spawning worker for", modelPath)
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