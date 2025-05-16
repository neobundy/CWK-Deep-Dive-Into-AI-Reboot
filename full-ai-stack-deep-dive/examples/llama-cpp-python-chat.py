#!/usr/bin/env python3
"""
Interactive CLI Chat Example for llama-cpp-python
Supports conversation history and streaming responses
"""

import os
import sys
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_model():
    """Download model from HuggingFace if not already cached."""
    print("📦 Checking for model...")
    model_path = hf_hub_download(
        repo_id="Qwen/Qwen3-32B-GGUF",
        filename="Qwen3-32B-Q4_K_M.gguf"
    )
    print(f"✅ Model ready: {os.path.basename(model_path)}")
    return model_path

def main():
    """Main interactive chat loop."""
    clear_screen()
    print("\n🔄 Initializing LLM, please wait...\n")
    
    # Initialize model
    model_path = get_model()
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,        # Context window size
        n_gpu_layers=-1,   # All layers on GPU
        verbose=False      # Set to True for debugging
    )
    
    # Set up chat history
    messages = [
        {"role": "system", "content": "You are a helpful assistant with a sense of humor."}
    ]
    
    clear_screen()
    print("\n🤖 Welcome to Interactive LLM Chat")
    print("    Type 'exit' to quit, 'clear' to reset conversation\n")
    
    # Main interaction loop
    while True:
        # Get user input
        user_input = input("\n🧑 You: ")
        
        # Handle special commands
        if user_input.lower() in ('exit', 'quit'):
            print("\n👋 Goodbye!\n")
            break
        elif user_input.lower() == 'clear':
            messages = [
                {"role": "system", "content": "You are a helpful assistant with a sense of humor."}
            ]
            clear_screen()
            print("\n🧹 Conversation history cleared\n")
            continue
        elif not user_input.strip():
            continue
            
        # Add user message to history
        messages.append({"role": "user", "content": user_input})
        
        # Prepare full prompt from message history
        # Convert chat history to prompt format that works with completion API
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"<system>\n{msg['content']}\n</system>\n\n"
            elif msg["role"] == "user":
                prompt += f"Human: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n"
        
        # Add final assistant prompt
        prompt += "Assistant: "
        
        print("\n🤖 Assistant: ", end="", flush=True)
        
        # Generate streaming response
        full_response = ""
        for chunk in llm(
            prompt,
            max_tokens=2048,
            stop=["Human:", "\nHuman:"],
            echo=False,
            stream=True
        ):
            text_chunk = chunk["choices"][0]["text"]
            print(text_chunk, end="", flush=True)
            full_response += text_chunk
        
        # Add assistant response to history
        messages.append({"role": "assistant", "content": full_response})
        print("\n")  # Add newline after response

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!\n")
        sys.exit(0)