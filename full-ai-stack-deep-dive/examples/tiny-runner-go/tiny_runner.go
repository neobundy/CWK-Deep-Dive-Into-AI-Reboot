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