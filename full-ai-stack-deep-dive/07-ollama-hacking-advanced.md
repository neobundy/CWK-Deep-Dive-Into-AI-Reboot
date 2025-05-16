# Chapter 7 · Extracting Models from Ollama — Building a CLI Feature in Go

*(Personal lab notebook — last verified 2025-05-13)*

> **Why this chapter?**
> Because downloading a 15-GB model only to have it locked inside Ollama's SHA-256 vault is frustrating. This lab shows you how to pry those weights loose: we'll bolt an `extract` command onto the Ollama CLI that turns any model into a portable GGUF file—complete with progress bars, integrity checks, and zero-copy speed. Along the way you'll reverse-map the blob store, trace the Modelfile graph, and craft a production-ready Go tool that moves multi-gig files without breaking a sweat.

*(Applies to the commit streamed on 2025-05-13; paths will drift over time.)*

---

## 0 · TL;DR — What You'll Build

1. A new **`extract`** command added to the Ollama CLI
2. Ability to convert any Ollama model (e.g., `phi4:latest`) to a standalone GGUF file
3. Real-time progress reporting for large file operations
4. A single custom binary named **`cwk-ollama`** that includes this feature

Again, our custom binary is a full-featured Ollama binary that includes the `extract` command. 

```bash
~/cwk-ollama
Usage:
  ollama [flags]
  ollama [command]

Available Commands:
  serve       Start ollama
  create      Create a model from a Modelfile
  show        Show information for a model
  run         Run a model
  stop        Stop a running model
  pull        Pull a model from a registry
  push        Push a model to a registry
  list        List models
  ps          List running models
  cp          Copy a model
  rm          Remove a model
  extract     Extract a model to a standalone GGUF file # <--- our custom command
  help        Help about any command

Flags:
  -h, --help      help for ollama
  -v, --version   Show version information

Use "ollama [command] --help" for more information about a command.
```

You can see the help for our custom command by running `~/cwk-ollama extract -h`.

```bash
~/cwk-ollama extract -h # <--- extra help for our custom command 
Extract a model to a standalone GGUF file

Usage:
  ollama extract [model] [destination] [flags]

Flags:
  -h, --help   help for extract
```

Example usage:
```bash
~/cwk-ollama extract phi4:latest ~/Downloads
Extracting phi4:latest (8.4 GB) to /Users/wankyuchoi/Downloads/phi4.gguf...
Extracting 100% |██████████████████████████████████████████████████████████████████████████████████████| (9.1/9.1 GB, 2.8 GB/s)
Successfully extracted phi4:latest to /Users/wankyuchoi/Downloads/phi4.gguf

llama-run /Users/wankyuchoi/Downloads/phi4.gguf -p "Tell me a joke in 10 words" -no-cnv
Why don't skeletons fight each other? They don't have the guts! 😄
```

Phi4 extracted and ready to use with other tools as verified by llama-run.

DISCLAIMER: Some recent models with new architectures may still fail when used with other backends. For example, when loading
  with llama.cpp or other tools, you might encounter errors like:
  load_tensors: loading model tensors, this can take a while... (mmap = true)
  llama_model_load: error loading model: done_getting_tensors: wrong number of tensors; expected 1247, got 808
  llama_model_load_from_file_impl: failed to load model

  This is typically due to version mismatches between Ollama's custom llama.cpp fork and other implementations. Older or more
  standard models should work fine across tools.

For instance, as of this writing `gemma3:27b-it-qat` can be extracted but llama-run and LM Studio fail to load it with the error: 

```
Error: Failed to load model: done_getting_tensors: wrong number of tensors; expected 1247, got 808
```

When a model fails to load in external tools—even after a successful extraction from Ollama—it's typically due to one or more of the following reasons:

1. **Custom Model Architecture Support**  
   Ollama uses a customized fork of `llama.cpp` with additional features and support for new model architectures. They often add support for the latest models (like Gemma3) before those changes are merged into the main `llama.cpp` branch or adopted by other tools.

2. **Tensor Structure Differences**  
   For example, if an external tool expects 1247 tensors but only finds 808, it means:
   - The model's structure includes elements that standard `llama.cpp` doesn't recognize.
   - Ollama's fork can handle these extra tensors, but the standard version cannot.

3. **GGUF Version/Format Differences**  
   Even though the file is in GGUF format, there may be subtle differences in how tensors are organized, named, or structured. Ollama may use custom extensions or tweaks to the GGUF format.

4. **Model-Specific Patches**  
   Ollama frequently applies patches to their `llama.cpp` fork to support unique requirements of certain models. These patches are often not present in the standard versions.

**Summary:**  
This is why newer or more complex models (like Gemma3) might work flawlessly in Ollama but fail in other tools—at least until those tools update their codebases to support the same model architectures and features.

**Note:** The vision capability in the Gemma3 model is almost certainly the cause of the tensor count mismatch. Multi-modal models like Gemma3 with vision capabilities have additional tensors to handle. When extracting multi-modal models, you should be aware that these models may only work with tools that explicitly support the same vision architecture implemented in Ollama's customized runtime.

---

## 1 · The Problem: Models Trapped in Ollama's Storage

Ollama stores downloaded models in a content-addressable blob system using SHA256 hashes as filenames. This design is excellent for deduplication and versioning but makes it difficult for users to directly access models for use with other tools like LM Studio, llama.cpp, or for creating backups.

When you download a model with Ollama, it gets stored here:
```
~/.ollama/models/blobs/sha256-[long-hash-value]
```

There's no direct way to know which hash corresponds to which model, making manual extraction nearly impossible.

---

## 2 · Understanding Ollama's Model Storage Architecture

Before building our feature, let's understand how Ollama organizes models:

1. **Model Naming and Resolution**:
   - User-friendly names like `phi4:latest` or `gemma3:27b-it-qat`
   - These names are resolved through manifests to actual file paths

2. **Storage Structure**:
   - `~/.ollama/models/` is the base directory (configurable via `OLLAMA_MODELS`)
   - `manifests/` contains metadata about models
   - `blobs/` contains the actual model files named by their SHA256 digest

3. **Modelfile References**:
   - Each model has a Modelfile (accessible via `ollama show --modelfile`)
   - The `FROM` directive in the Modelfile points to the actual model blob

We'll need to navigate this system to find the right file and extract it.

---

## 3 · Setting Up Our Development Environment

First, let's clone the repo and prepare our workspace:

```bash
git clone https://github.com/ollama/ollama.git
cd ollama

# Create a starting point
git commit --allow-empty -m "CWK: Starting point for extract feature"
```

Feel free to use any naming convention that suits you. Personally, I append my initials—C.W.K (Wankyu Choi, or Creative Works of Knowledge)—as a suffix for easy identification.

Get in the habit of committing your work early and often—after every meaningful change, make a commit. This practice makes it much easier to debug issues, since you can always roll back to a known good state if something breaks. If you're collaborating with AI, be explicit: ask it to commit after each step. Frequent, incremental commits are a widely recognized best practice and will save you time and frustration in the long run.

---

## 4 · Implementing the Extract Command

### 4.1 Create the Command Registration

First, let's create a file `cmd/cmd_extract.go` to register our new command:

```go
package cmd

import (
    "fmt"
    "os"
    "path/filepath"
    
    "github.com/spf13/cobra"
)

// validateExtractArgs validates the arguments for the extract command
func validateExtractArgs(cmd *cobra.Command, args []string) error {
    destPath := args[1]

    // Check if destination is a directory
    fi, err := os.Stat(destPath)
    if err != nil {
        if !os.IsNotExist(err) {
            return err
        }
        // Destination doesn't exist, check if parent directory exists
        parent := filepath.Dir(destPath)
        if _, err := os.Stat(parent); err != nil {
            return fmt.Errorf("destination parent directory %s does not exist", parent)
        }
    } else if !fi.IsDir() {
        return fmt.Errorf("%s is not a directory", destPath)
    }

    return nil
}

// CreateExtractCommand creates the extract command
func CreateExtractCommand() *cobra.Command {
    extractCmd := &cobra.Command{
        Use:     "extract [model] [destination]",
        Short:   "Extract a model to a standalone GGUF file",
        Args:    cobra.ExactArgs(2),
        PreRunE: validateExtractArgs,
        RunE:    ExtractHandler,
    }

    return extractCmd
}
```

Commit our first piece:

```bash
git add cmd/cmd_extract.go
git commit -m "CWK: Add extract command registration"
```

### 4.2 Implement Core Extraction Logic

Now, let's create `cmd/extract.go` with the main functionality:

```go
// Package cmd implements Ollama's CLI commands
package cmd

// ExtractCmd implements the "ollama extract" command which extracts models
// from Ollama's content-addressable blob storage into standalone GGUF files.
//
// This allows models downloaded via Ollama to be used with other LLM tools
// and simplifies model file handling by bypassing the SHA256 hash jungle.

import (
    "context"
    "fmt"
    "io"
    "os"
    "path/filepath"
    "strings"

    "github.com/schollz/progressbar/v3"
    "github.com/spf13/cobra"

    "github.com/ollama/ollama/api"
)

func ExtractHandler(cmd *cobra.Command, args []string) error {
    // Get arguments
    modelName := args[0]
    destDir := args[1]

    // Handle model name normalization for consistency with Ollama
    // If no tag is specified, Ollama assumes ":latest"
    if !strings.Contains(modelName, ":") {
        modelName = modelName + ":latest"
    }

    // Create client
    client, err := api.ClientFromEnvironment()
    if err != nil {
        return err
    }

    // Find the model in the list of available models
    models, err := client.List(context.Background())
    if err != nil {
        return fmt.Errorf("failed to list models: %w", err)
    }

    var targetModel *api.ListModelResponse
    for _, model := range models.Models {
        // Handle exact match first
        if model.Name == modelName {
            targetModel = &model
            break
        }

        // If user provided name without tag, match with the default ":latest" tag
        // This is already handled by our normalization above, but keep as a fallback
        if model.Name == modelName+":latest" {
            targetModel = &model
            break
        }
    }

    if targetModel == nil {
        return fmt.Errorf("model %s not found - available models:\n%s", 
            modelName, 
            formatAvailableModels(models.Models))
    }

    // Get the modelfile to extract the model path
    showReq := &api.ShowRequest{Name: modelName}
    modelInfo, err := client.Show(cmd.Context(), showReq)
    if err != nil {
        return fmt.Errorf("failed to get model details: %w", err)
    }

    // Try first to get the model path from the Modelfile
    modelPath := extractModelPathFromModelfile(modelInfo.Modelfile)
    
    // If not found in Modelfile, try to find it from the blob storage
    if modelPath == "" {
        var findErr error
        modelPath, findErr = findModelBlob(targetModel)
        if findErr != nil {
            return fmt.Errorf("couldn't locate model file: %w", findErr)
        }
    }
    
    // Verify the model path exists
    if _, err := os.Stat(modelPath); err != nil {
        return fmt.Errorf("model file not found at path: %s", modelPath)
    }

    // Generate clean output filename from model name
    // For models like "gemma3:27b-it-qat", create "gemma3-27b-it-qat.gguf"
    outputName := formatOutputFilename(targetModel.Name)
    outputPath := filepath.Join(destDir, outputName)

    // Copy the file with progress bar
    fmt.Printf("Extracting %s (%s) to %s...\n", 
        targetModel.Name, 
        formatBytes(targetModel.Size),
        outputPath)
        
    if err := extractModelFile(modelPath, outputPath); err != nil {
        return fmt.Errorf("extraction failed: %w", err)
    }

    fmt.Printf("Successfully extracted %s to %s\n", targetModel.Name, outputPath)
    return nil
}

// extractModelPathFromModelfile extracts the model path from the FROM line in the Modelfile
func extractModelPathFromModelfile(modelfile string) string {
    lines := strings.Split(modelfile, "\n")
    for _, line := range lines {
        line = strings.TrimSpace(line)
        if strings.HasPrefix(line, "FROM ") {
            parts := strings.SplitN(line, "FROM ", 2)
            if len(parts) == 2 {
                // Extract the path - could be a file path or another model name
                pathOrModel := strings.TrimSpace(parts[1])
                if strings.HasPrefix(pathOrModel, "/") {
                    // It's a file path, return it
                    return pathOrModel
                }
            }
        }
    }
    return ""
}

// findModelBlob locates the main GGUF model file in the blob directory
func findModelBlob(modelInfo *api.ListModelResponse) (string, error) {
    // Get OLLAMA_MODELS env var or use default ~/.ollama/models
    modelsDir := os.Getenv("OLLAMA_MODELS")
    if modelsDir == "" {
        homeDir, err := os.UserHomeDir()
        if err != nil {
            return "", err
        }
        modelsDir = filepath.Join(homeDir, ".ollama", "models")
    }
    
    // Get the digest from the ListModelResponse
    digest := modelInfo.Digest
    
    if digest == "" {
        return "", fmt.Errorf("model digest not found in model info")
    }
    
    // The proper way to construct the model file path
    // Ollama uses 'sha256-[digest]' format for file names on disk
    // with the full 64 character digest
    blobsDir := filepath.Join(modelsDir, "blobs")
    digestPath := "sha256-" + digest
    modelPath := filepath.Join(blobsDir, digestPath)
    
    // Check if the file exists at the expected path
    if _, err := os.Stat(modelPath); err == nil {
        return modelPath, nil
    }
    
    // If direct path doesn't work, try searching for files with the first part of the digest
    files, err := os.ReadDir(blobsDir)
    if err != nil {
        return "", fmt.Errorf("failed to read blobs directory: %w", err)
    }
    
    for _, file := range files {
        if !file.IsDir() && strings.Contains(file.Name(), digest[:12]) {
            modelPath := filepath.Join(blobsDir, file.Name())
            return modelPath, nil
        }
    }
    
    return "", fmt.Errorf("model file not found for digest: %s", digest)
}

// extractModelFile copies the model file to the destination with progress reporting
func extractModelFile(srcPath, destPath string) error {
    // Open source file
    source, err := os.Open(srcPath)
    if err != nil {
        return err
    }
    defer source.Close()

    // Get file size for progress tracking
    fileInfo, err := source.Stat()
    if err != nil {
        return err
    }
    totalSize := fileInfo.Size()

    // Create destination file
    destination, err := os.Create(destPath)
    if err != nil {
        return err
    }
    defer destination.Close()

    // Create a progress bar
    bar := progressbar.DefaultBytes(
        totalSize,
        "Extracting",
    )

    // Copy the file with progress updates
    buffer := make([]byte, 4*1024*1024) // 4MB buffer
    for {
        n, err := source.Read(buffer)
        if err != nil && err != io.EOF {
            return err
        }
        if n == 0 {
            break
        }

        // Write to destination
        if _, err := destination.Write(buffer[:n]); err != nil {
            return err
        }

        // Update progress bar
        if err := bar.Add(n); err != nil {
            return err
        }
    }

    return nil
}

// Format the output filename from model name
func formatOutputFilename(modelName string) string {
    // Replace colon with hyphen: "gemma3:27b-it-qat" -> "gemma3-27b-it-qat"
    name := strings.ReplaceAll(modelName, ":", "-")
    
    // Remove ":latest" suffix if present
    name = strings.TrimSuffix(name, "-latest")
    
    return name + ".gguf"
}

// Format available models as a string for error messages
func formatAvailableModels(models []api.ListModelResponse) string {
    var sb strings.Builder
    for _, model := range models {
        sb.WriteString(fmt.Sprintf("  - %s\n", model.Name))
    }
    return sb.String()
}

// Format bytes to human-readable string
func formatBytes(bytes int64) string {
    const unit = 1024
    if bytes < unit {
        return fmt.Sprintf("%d B", bytes)
    }
    div, exp := int64(unit), 0
    for n := bytes / unit; n >= unit; n /= unit {
        div *= unit
        exp++
    }
    return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}
```

Commit our implementation:

```bash
git add cmd/extract.go
git commit -m "CWK: Implement model extraction logic"
```

### 4.3 Register the Command in Ollama's CLI

Now we need to update `cmd/cmd.go` to register our new command with the CLI:

```go
// Inside the NewCLI function, find the rootCmd.AddCommand block and add our command
rootCmd.AddCommand(
    serveCmd,
    createCmd,
    showCmd,
    runCmd,
    stopCmd,
    pullCmd,
    pushCmd,
    listCmd,
    psCmd,
    copyCmd,
    deleteCmd,
    runnerCmd,
    CreateExtractCommand(), // Add this line
)
```

Commit this change:

```bash
git add cmd/cmd.go
git commit -m "CWK: Register extract command in Ollama CLI"
```

### 4.4 Create a Custom Binary

To keep our hack separate from the standard Ollama binary, let's create a custom binary with our own version prefix:

```go
// cmd/runner/cwk-ollama.go
package main

import (
    "context"
    "fmt"
    "os"

    "github.com/ollama/ollama/cmd"
    "github.com/ollama/ollama/version"
)

func init() {
    // Set a custom version prefix for our build
    version.Version = "cwk-" + version.Version
}

func main() {
    // Create the CLI
    rootCmd := cmd.NewCLI()

    // Execute the command
    if err := rootCmd.ExecuteContext(context.Background()); err != nil {
        fmt.Fprintf(os.Stderr, "Error: %s\n", err)
        os.Exit(1)
    }
}
```

Commit our custom binary source:

```bash
git add cmd/runner/cwk-ollama.go
git commit -m "CWK: Add custom binary with version prefix"
```

---

## 5 · Building and Testing

Now let's build our custom binary:

```bash
# Build the custom binary
go build -o bin/cwk-ollama cmd/runner/cwk-ollama.go
chmod +x bin/cwk-ollama
```

To test our feature:

```bash
# Make sure the Ollama server is running
ollama serve &

# List available models
./bin/cwk-ollama list

# Extract a model (replace phi4:latest with any model you have installed)
./bin/cwk-ollama extract phi4:latest ~/Downloads
```

Our custom binary operates independently of the Ollama server—you can run the standard Ollama server as usual, and use the custom binary solely for model extraction. Since our changes don't modify the server component, both binaries work side by side without conflict.

You should see output like:

```
Extracting your_model_name:latest (8.4 GB) to /Users/username/Downloads/your_model_name.gguf...
Extracting 100% |██████████████████████████████| (9.1/9.1 GB, 3.8 GB/s)
Successfully extracted your_model_name:latest to /Users/username/Downloads/your_model_name.gguf
```

---

## 6 · Design Decisions and Technical Insights

### 6.1 Model Path Resolution Strategy

We implemented a dual-approach strategy for finding the model file:

1. **Primary Strategy - Modelfile Extraction**:
   - Get the Modelfile via `ShowRequest`
   - Extract the path from the `FROM` directive
   - This is the most direct and reliable path

2. **Fallback Strategy - Blob Search**:
   - If the Modelfile approach fails, use the model digest
   - Look for the file in the blobs directory
   - Try exact match with 'sha256-' prefix
   - If still not found, search for files containing the first 12 chars of the digest

This makes our solution robust against different Ollama versions and configurations.

### 6.2 Large File I/O Considerations

Model files are large (often 4-20GB), which requires special handling:

1. **Progress Reporting**:
   - We use the `progressbar` library to provide real-time feedback
   - Shows completion percentage, data rate, and ETA

2. **Efficient Buffering**:
   - We use a 4MB buffer size to balance memory usage and performance
   - Too small would cause excessive syscalls; too large wastes memory

3. **Resource Management**:
   - We use `defer` to ensure files are closed properly even on error
   - This prevents file descriptor leaks

### 6.3 Error Handling

Our implementation includes comprehensive error handling:

1. **Input Validation**:
   - Destination directory existence
   - Model name resolution

2. **Descriptive Error Messages**:
   - When a model isn't found, we list available models
   - File path errors include the exact problematic path

3. **Error Wrapping**:
   - We use Go 1.13+ error wrapping (`fmt.Errorf("...%w", err)`)
   - This preserves the error chain for better debugging

---

## 7 · Real-World Applications

This feature solves real pain points for Ollama users:

1. **Interoperability**:
   - Use models downloaded with Ollama in other applications (LM Studio, llama.cpp)
   - No need to download models multiple times

2. **Backup**:
   - Create backups of models in a standard format
   - Store them wherever you want (external drives, cloud storage)

3. **Sharing**:
   - Share models with others without requiring them to use Ollama
   - Simplifies collaboration

4. **Fine-tuning**:
   - Use the extracted models as a base for fine-tuning with other tools
   - Enables more advanced model adaptation workflows

---

## 8 · Beyond the Extract Command

Once you understand how to extract models, you could expand this feature in several directions:

1. **Model Conversion**:
   - Add options to convert between different quantization levels
   - Implement GGUF → GGML conversion or other formats

2. **Metadata Extraction**:
   - Extract tokenizer, configuration, and other model metadata
   - Create complete model packages for different frameworks

3. **Model Analysis**:
   - Add options to analyze model architectures
   - Display tensor sizes, layer configurations, etc.

---

## 9 · Learning From This Implementation

By implementing this feature, we've learned several key concepts:

1. **CLI Extension**: How to extend existing command-line applications
2. **API Integration**: How to interact with Ollama's internal APIs
3. **File I/O**: Efficient large file operations with progress reporting
4. **Resource Management**: Proper handling of file handles and memory
5. **Error Handling**: Comprehensive error handling for a better user experience

These skills are transferable to other Go projects and will help you build more robust CLI applications.

---

## 10 · Checkpoint

- ✔ Added `extract` command to Ollama CLI and compiled **cwk-ollama** binary
- ✔ Implemented dual-path model resolution (Modelfile → blob fallback)
- ✔ Streamed multi-GB copy with 4 MB buffer + live progress bar
- ✔ Wrapped every syscall in clear, user-friendly errors
- ✔ Verified end-to-end: `phi4:latest` ➜ `phi4.gguf` ➜ `llama-run`

Models are no longer prisoners—go set them free, then meet me in Chapter 8. 🚀

---

[⇧ Back to README](../README.md)
