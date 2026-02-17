import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

#works
# Ensure we can import the library from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spectra_lora import (
    load_prithvi_model, 
    inject_spectra_lora, 
    SpectraConfig, 
    get_spectral_fingerprint,
    SpectraLoRALayer
)

# Re-use the context patcher from train.py logic
# (In a real scenario, this helper should be in utils.py)
class SpectraContext:
    current_z = None

def patch_model_for_context(model):
    print("🔧 Patching model for inference...")
    for module in model.modules():
        if isinstance(module, SpectraLoRALayer):
            original_forward = module.forward
            def patched_forward(x):
                z = SpectraContext.current_z
                if z is None:
                    # Fallback for safety if context is missing
                    return original_forward(x, torch.zeros(x.shape[0], 5).to(x.device))
                return original_forward(x, z)
            module.forward = patched_forward
    return model

def run_demo():
    # 1. Force CPU Mode (Safe for your laptop)
    device = torch.device("cpu")
    print(f"💻 Running on {device} (Optimized for your hardware)...")

    # 2. Load Model
    # We use the standard config.
    print("⏳ Loading Prithvi-100M (This might take 30s)...")
    try:
        model = load_prithvi_model()
        model = inject_spectra_lora(model)
        model = patch_model_for_context(model)
        model.to(device)
        model.eval() # Important: Switch to Eval mode!
    except Exception as e:
        print(f"\n❌ Critical Error Loading Model: {e}")
        print("Tip: Ensure you have internet access to download weights from Hugging Face.")
        return

    # 3. Create a Synthetic "Satellite Image"
    # We will create a fake "Forest" patch to test the Physics Engine.
    # Dimensions: (1, 6, 224, 224) -> Batch, Bands, H, W
    print("🧪 Generating synthetic satellite data...")
    dummy_image = torch.zeros(1, 6, 224, 224).to(device)
    
    # Fill bands to simulate a Vegetation signature (High NIR, Low Red)
    # Band Order in Config: Blue=0, Green=1, Red=2, NIR=3, SWIR1=4, SWIR2=5
    dummy_image[:, 2, :, :] = 0.1  # RED (Low)
    dummy_image[:, 3, :, :] = 0.8  # NIR (High) -> This should trigger High NDVI

    # 4. Extract Physics Context
    print("🔬 Running Physics Engine...")
    with torch.no_grad():
        z = get_spectral_fingerprint(dummy_image, SpectraConfig.BAND_MAP)
        SpectraContext.current_z = z

    # 5. Check the Physics Vector
    # We expect Index 0 (NDVI) to be high.
    z_val = z[0].numpy()
    print("\n" + "="*30)
    print("🔍 PHYSICS DIAGNOSTIC")
    print("="*30)
    print(f"NDVI (Vegetation): {z_val[0]:.2f} (Expected > 0.5)")
    print(f"NDWI (Water):      {z_val[2]:.2f}")
    print(f"NDBI (Urban):      {z_val[3]:.2f}")
    print("-" * 30)

    # 6. Run Inference
    print("🚀 Running Forward Pass...")
    with torch.no_grad():
        # Prithvi output shape is typically [Batch, Tokens, Dim]
        # We simulate the segmentation head output for this demo
        output_features = model(dummy_image)
        
        # In a real app, you would pass 'output_features' to a Decoder.
        # Here we just check if it ran without crashing.
    
    print("✅ Success! The model accepted the physics context and produced output.")
    print(f"Output Shape: {output_features.shape}")

if __name__ == "__main__":
    run_demo()