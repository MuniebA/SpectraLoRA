import torch
import os
import sys
import importlib.util
from huggingface_hub import hf_hub_download
from .layers import SpectraLoRALayer
from .config import SpectraConfig

"""
SpectraLoRA Model Loader (Phase 3, Step 5)
------------------------------------------
Handles the complexity of loading the custom Prithvi-100M architecture.
Since Prithvi is not a standard Hugging Face AutoModel, we must:
1. Download the architecture code (Prithvi.py) dynamically.
2. Import it as a Python module.
3. Load the checkpoint weights manually.
"""

def load_prithvi_model(
    repo_id: str = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M", 
    filename: str = "Prithvi_100M.pt",
    config_path: str = "Prithvi_100M_config.yaml"
):
    """
    Downloads and reconstructs the Prithvi Vision Transformer.
    
    Args:
        repo_id (str): Hugging Face repository ID.
        filename (str): The weights filename.
    
    Returns:
        model (nn.Module): The loaded Prithvi model, ready for surgery.
    """
    print(f"🌍 SpectraLoRA: Downloading Prithvi weights from {repo_id}...")
    
    # 1. Download the Weights
    weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
    
    # 2. Download the Architecture Definition
    # Note: Prithvi repos often keep the class in a file named 'Prithvi.py' or similar.
    # We will try to download the architecture file. 
    # If this fails in the future, we fallback to a local version.
    try:
        model_code_path = hf_hub_download(repo_id=repo_id, filename="Prithvi.py")
    except Exception:
        # Fallback: Sometimes the file is named differently or we use a local copy
        print("⚠️ Warning: Could not download Prithvi.py from HF. Using local fallback if available.")
        # For this code snippet, we assume the user might have 'prithvi_vit.py' locally 
        # or we implement a minimal ViT wrapper. 
        # For robustness, we will assume standard ViT loading for the snippet:
        pass

    # 3. Dynamic Import (The "Hack" to load remote code)
    # We load the downloaded python file as a module named 'prithvi_module'
    spec = importlib.util.spec_from_file_location("prithvi_module", model_code_path)
    prithvi_module = importlib.util.module_from_spec(spec)
    sys.modules["prithvi_module"] = prithvi_module
    spec.loader.exec_module(prithvi_module)
    
    # 4. Instantiate the Model
    # Prithvi is usually a 'MaskedAutoencoderViT'. 
    # We need to match the config parameters (img_size, num_frames, etc.)
    print("🏗️ Constructing Model Architecture...")
    
    # These parameters match the standard Prithvi-100M config
    model = prithvi_module.MaskedAutoencoderViT(
        img_size=224,
        patch_size=16,
        in_chans=6,            # Blue, Green, Red, NIR, SWIR1, SWIR2
        embed_dim=768,         # ViT-Base
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=torch.nn.LayerNorm
    )

    # 5. Load State Dict
    print("⚖️ Loading Weights...")
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    # Prithvi checkpoints sometimes wrap weights in 'model' or 'state_dict' keys
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        
    # Load and allow mismatch (because we might ignore the decoder for segmentation)
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"✅ Model Loaded. Missing keys (expected for fine-tuning): {len(msg.missing_keys)}")
    
    return model

"""
This function will take this loaded model, find the attention layers, and replace them with our SpectraLoRALayer
"""
def inject_spectra_lora(model: torch.nn.Module, config: SpectraConfig = SpectraConfig):
    """
    The Surgeon Function (Phase 3, Step 6).
    Walks through the Prithvi model and replaces specific Linear layers 
    with our physics-aware SpectraLoRALayer.
    
    Args:
        model (nn.Module): The loaded Prithvi Vision Transformer.
        config (SpectraConfig): Configuration for the adapters.
    
    Returns:
        model (nn.Module): The modified model, ready for training.
    """
    print("✨ SpectraLoRA: Beginning surgery on the model...")
    
    # Counter to track how many layers we modified
    layers_modified = 0
    
    # We iterate through all named modules in the network
    # The Prithvi ViT structure usually names attention blocks like:
    # blocks.0.attn.qkv (Query-Key-Value) OR blocks.0.attn.q_bias, etc.
    # We specifically target the ATTENTION LAYERS.
    
    # Note: Prithvi/ViT implementation often combines QKV into one layer.
    # Or separate Q, K, V layers. We need to be robust.
    # For standard ViT (timm style), it's often 'attn.qkv' or 'attn.proj'.
    
    for name, module in model.named_children():
        # Recursive helper to dig deep into the network
        _recursive_injection(model, name, module, config, counter_ref=[0])
        
    print(f"✅ Surgery Complete. Injected SpectraLoRA into {layers_modified} layers.")
    return model

def _recursive_injection(parent_module, child_name, child_module, config, counter_ref):
    """
    Helper function to recursively search for Linear layers to replace.
    """
    
    # 1. Base Case: If the module has children, keep digging
    if len(list(child_module.children())) > 0:
        for name, grandchild in child_module.named_children():
            _recursive_injection(child_module, name, grandchild, config, counter_ref)
            
    # 2. Check if this module is a Target Layer
    # We generally target 'qkv' (Query-Key-Value) or 'proj' (Output Projection) in Attention.
    # We want to catch strings like "blocks.0.attn.qkv"
    elif isinstance(child_module, torch.nn.Linear):
        if "attn" in child_name or "qkv" in child_name or "proj" in child_name:
            
            # 3. THE SWAP (The "Surgery")
            print(f"   -> Injecting LoRA into: {child_name}")
            
            # Create our custom layer, wrapping the original one
            # This automatically freezes the original weights inside __init__
            new_layer = SpectraLoRALayer(child_module, config)
            
            # Replace it in the parent
            setattr(parent_module, child_name, new_layer)
            
            counter_ref[0] += 1