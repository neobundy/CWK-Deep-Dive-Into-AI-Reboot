from llama_cpp import Llama
from llama_cpp import Llama

llm = Llama(
    model_path="/Users/wankyuchoi/.ollama/models/blobs/sha256-0bc207fe2eecc3385d4099d7a04ac6e37adc0c2f02e95be84cd7d7e4f441b54e",
    n_gpu_layers=-1,          # off-load all layers to Metal
    verbose=False             # ← suppress perf + metadata spam
)

resp = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Hello, Mac! What's your name?"}],
    stream=False        # True → stream tokens
)

print(resp["choices"][0]["message"]["content"].strip())