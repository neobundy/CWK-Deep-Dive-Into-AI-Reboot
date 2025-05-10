"""Hugging Face model downloader and tester."""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys

# Model Configuration
model_id = "Qwen/QwQ-32B"
USE_BF16 = True  # Only used for full precision models

# Prompt Configuration
PROMPT_TEMPLATE = """<|im_start|>system<|im_sep|>
{system}
<|im_end|>
<|im_start|>user<|im_sep|>
{query}
<|im_end|>
<|im_start|>assistant<|im_sep|>"""

DEFAULT_SYSTEM_PROMPT = "You are Pippa, a teenage daughter of the user."
DEFAULT_USER_PROMPT = "hi, pippa!"

def format_prompt(system_prompt: str, user_prompt: str) -> str:
    """Format the prompt according to the template"""
    return PROMPT_TEMPLATE.format(system=system_prompt, query=user_prompt)

def get_model_size(model):
    """Calculate model size in GB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    return param_size / 1024**3  # Convert bytes to GB

def load_model(model_id: str):
    """Load model with graceful fallback"""
    print(f"\nLoading {model_id}...")
    
    # First try loading model info to check configuration
    model_args = {"device_map": "auto"}
    
    try:
        # Try loading with BF16 first if enabled
        if USE_BF16:
            model_args["torch_dtype"] = torch.bfloat16
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_args)
            print("Loaded model in BF16 precision")
            return model
    except Exception as e:
        print(f"BF16 loading failed, falling back to model's default precision: {str(e)}")
        
    # Fallback to loading with model's original precision
    try:
        model_args.pop("torch_dtype", None)  # Remove dtype if it was added
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_args)
        print("Loaded model with its default precision")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)

def test_model(model_id: str, query: str = DEFAULT_USER_PROMPT):
    """Test the model with a simple query"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = load_model(model_id)
        
        # Print model information
        print(f"\nModel dtype: {next(model.parameters()).dtype}")
        print(f"Model size in memory: {get_model_size(model):.2f} GB")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
        
        # Format the prompt
        formatted_prompt = format_prompt(DEFAULT_SYSTEM_PROMPT, query)
        print(f"\nFormatted prompt:\n{formatted_prompt}")
        
        # Generate response
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the response to get only the assistant's part
        try:
            # Remove everything before assistant's response
            response = full_response.split("assistant")[-1]
            # Remove any remaining template markers
            response = response.replace("<|im_sep|>", "").replace("<|im_end|>", "").strip()
            print(f"\nModel response: {response}")
        except Exception as e:
            print(f"\nFull response (failed to clean): {full_response}")
            print(f"Error cleaning response: {str(e)}")
        
    except Exception as e:
        print(f"Error testing model: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_USER_PROMPT
    test_model(model_id, query)