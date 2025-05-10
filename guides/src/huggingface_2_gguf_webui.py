import os
import streamlit as st

def list_huggingface_models(cache_dir):
    """Lists available models in the Hugging Face cache directory."""
    models = []
    for root, dirs, _ in os.walk(cache_dir):
        for dir_name in dirs:
            if dir_name.startswith("models--"):
                models.append(os.path.join(root, dir_name))
        break  # Only look at the top level
    return models

def get_latest_snapshot(model_path):
    """Gets the latest snapshot path for a model."""
    snapshots_path = os.path.join(model_path, "snapshots")
    if not os.path.exists(snapshots_path):
        return None
    
    snapshot_subdirs = sorted(os.listdir(snapshots_path))
    if not snapshot_subdirs:
        return None
    
    return os.path.join(snapshots_path, snapshot_subdirs[-1])

def convert_to_gguf(model_path, output_path, model_type):
    """Converts a model to GGUF format using llama.cpp."""
    llama_cpp_path = os.path.expanduser("~/Dropbox/Python-Projects/llama.cpp")
    convert_script = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
    
    if not os.path.exists(convert_script):
        return f"Error: llama.cpp convert script not found at {convert_script}"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Build and execute the conversion command
    cmd = f"python {convert_script} --outfile {output_path} --outtype {model_type} {model_path}"
    result = os.system(cmd)
    
    if result == 0:
        return f"Successfully converted model to: {output_path}"
    else:
        return f"Error converting model. Command exit code: {result}"

# Streamlit UI
st.title("Hugging Face to GGUF Model Converter")

# User paths
home_dir = os.path.expanduser("~")
hf_cache_dir = os.path.join(home_dir, ".cache/huggingface/hub")
lmstudio_base_dir = os.path.join(home_dir, ".cache/lm-studio/models")

# Sidebar configuration
st.sidebar.header("Paths")
st.sidebar.text_input("Hugging Face Cache Directory", hf_cache_dir, disabled=True)
st.sidebar.text_input("LM Studio Base Directory", lmstudio_base_dir, disabled=True)

# Model type selection
model_types = ["f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"]
selected_type = st.sidebar.selectbox(
    "Select GGUF Model Type",
    model_types,
    index=model_types.index("q8_0"),  # Default to q8_0
    help="q8_0 is recommended for balance of quality and size"
)

# List models
models = list_huggingface_models(hf_cache_dir)
if not models:
    st.error("No models found in the Hugging Face cache.")
else:
    selected_model = st.selectbox("Select a model to convert", models)
    
    if selected_model:
        # Get latest snapshot
        snapshot_path = get_latest_snapshot(selected_model)
        if snapshot_path:
            st.success(f"Found latest snapshot: {os.path.basename(snapshot_path)}")
            
            # Show output path
            model_name = os.path.basename(selected_model).replace("models--", "").replace("--", "/")
            org_name, model_simple_name = model_name.split("/")  # Split into org and model name
            output_path = os.path.join(
                lmstudio_base_dir,
                org_name,
                f"{model_simple_name}-GGUF",
                f"{model_simple_name}-{selected_type}.gguf"
            )
            st.text_input("Output Path", output_path, disabled=True)
            
            # Convert button
            if st.button("Convert to GGUF"):
                with st.spinner("Converting model to GGUF format..."):
                    result = convert_to_gguf(snapshot_path, output_path, selected_type)
                    if "Error" in result:
                        st.error(result)
                    else:
                        st.success(result)
        else:
            st.error("No snapshot found for selected model.") 