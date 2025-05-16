#!/bin/zsh

# Upgrade ollama
brew upgrade ollama

# Launch with required environment variables
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KEEP_ALIVE=-1 OLLAMA_CONTEXT_LENGTH=128000 ollama serve
