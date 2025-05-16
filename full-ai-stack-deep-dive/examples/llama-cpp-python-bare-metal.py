import os
import llama_cpp
import ctypes

# ⚠️ CAUTION: This example demonstrates low-level llama.cpp API calls.
# It may cause a bus error on some systems due to memory access patterns or API changes.
# This is intended as a learning example to understand the API, not for production use.


# Expand home directory path
model_path = os.path.expanduser("~/.tiny_ollama/model.gguf")

llama_cpp.llama_backend_init(False)        # ① global runtime init

# ② initialize model parameters (not context parameters)
model_params = llama_cpp.llama_model_default_params()
model_params.n_ctx = 8192  # max tokens to allocate space for

model = llama_cpp.llama_load_model_from_file(
    model_path.encode('utf-8'),  # use full path, properly encoded
    model_params
)

# Get vocabulary from model - needed for tokenization
vocab = llama_cpp.llama_model_get_vocab(model)

# ③ initialize context parameters
ctx_params = llama_cpp.llama_context_default_params()
ctx_params.n_ctx = model_params.n_ctx
ctx_params.n_batch = 512  # tokens to process in parallel
ctx_params.n_threads = 8  # CPU threads to use during computation

ctx = llama_cpp.llama_new_context_with_model(model, ctx_params)

# ④ tokenize prompt
max_tokens = ctx_params.n_ctx
tokens = (llama_cpp.llama_token * int(max_tokens))()
prompt = b"Q: Name the planets in the Solar System? A: "

# Call tokenize with all required parameters (7 arguments)
n_tokens = llama_cpp.llama_tokenize(
    vocab,                   # vocabulary pointer
    prompt,                  # prompt bytes
    len(prompt),             # prompt length
    tokens,                  # output tokens array
    max_tokens,              # max tokens
    ctypes.c_bool(True),     # add_special
    ctypes.c_bool(False)     # parse_special
)

# ⑤ inference loop - prepare batch
batch = llama_cpp.llama_batch_init(n_tokens, 0, 1)
batch.n_tokens = n_tokens  # Explicitly set the token count in the batch
for i in range(n_tokens):
    batch.token[i] = tokens[i]
    batch.pos[i] = i
    batch.n_seq_id[i] = 1
    batch.seq_id[i][0] = 0
batch.logits[batch.n_tokens - 1] = 1  # compute logits only for the last token

# ⑥ run the computation for prompt
if llama_cpp.llama_decode(ctx, batch) != 0:
    print("Error: Failed to decode initial batch")
    exit(1)

# Process one token at a time for simplicity
print("\nSequence: ", end="", flush=True)

# Track context position
n_past = n_tokens
n_remaining = max_tokens - n_past
token_id = 0

# Get EOS token ID once for later comparison
eos_token_id = llama_cpp.llama_token_eos(model)

# Main token generation loop
for i in range(30):  # Generate up to 30 new tokens
    # Get logits from last prediction
    logits = llama_cpp.llama_get_logits(ctx)
    n_vocab = llama_cpp.llama_n_vocab(model)  # Get vocab size from model, not ctx
    
    # Simple greedy sampling - find token with highest probability
    max_logit = -float('inf')
    token_id = 0
    
    # We'll examine a limited number of tokens for efficiency
    top_k = 100
    for j in range(min(top_k, n_vocab)):
        if logits[j] > max_logit:
            max_logit = logits[j]
            token_id = j
    
    # Check if we got the EOS token
    if token_id == eos_token_id:
        print("\n[EOS reached]")
        break
    
    # Convert token to text and print
    buf_size = 32  # Buffer size for token text - some tokens might be longer
    buf = ctypes.create_string_buffer(buf_size)
    
    # Use the updated llama_token_to_piece API
    n_chars = llama_cpp.llama_token_to_piece(
        model,            # Model pointer
        token_id,         # Token ID
        0,                # Token ID for previous token (0 means beginning of text)
        buf,              # Output buffer
        buf_size,         # Buffer size
        False             # Special handling flag
    )
    
    if n_chars > 0:
        # Convert the bytes to a Python string and print
        token_str = buf.raw[:n_chars]
        print(token_str.decode('utf-8', errors='replace'), end='', flush=True)
    
    # Prepare new batch with just this token
    new_batch = llama_cpp.llama_batch_init(1, 0, 1)
    new_batch.n_tokens = 1  # Explicitly set token count
    new_batch.token[0] = token_id
    new_batch.pos[0] = n_past
    new_batch.n_seq_id[0] = 1
    new_batch.seq_id[0][0] = 0
    new_batch.logits[0] = 1  # We want logits for this token
    
    # Update context with new token
    if llama_cpp.llama_decode(ctx, new_batch) != 0:
        print("\nError: Failed to decode token")
        break
    
    # Update position counter
    n_past += 1
    
    # Free batch
    llama_cpp.llama_batch_free(new_batch)
    
    # Safety check for context window
    if n_past >= max_tokens - 4:
        print("\n[Context window full]")
        break

# ⑦ clean up resources
llama_cpp.llama_batch_free(batch)
llama_cpp.llama_free(ctx)
llama_cpp.llama_free_model(model)
llama_cpp.llama_backend_free()  # global shutdown
print("\n")