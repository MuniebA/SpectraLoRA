import torch
import numpy as np
import rasterio
import os
import sys
import matplotlib.pyplot as plt

# Fix path to import spectra_lora
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spectra_lora import (
    load_prithvi_model, 
    inject_spectra_lora, 
    SpectraConfig, 
    get_spectral_fingerprint,
    SpectraLoRALayer
)

# Context Patch (Same as demo)
class SpectraContext:
    current_z = None

def patch_model_for_context(model):
    for module in model.modules():
        if isinstance(module, SpectraLoRALayer):
            original_forward = module.forward
            def patched_forward(x):
                z = SpectraContext.current_z
                return original_forward(x, z)
            module.forward = patched_forward
    return model

def load_real_tiff(path):
    print(f"📂 Loading Tiff: {path}")
    with rasterio.open(path) as src:
        # Prithvi expects 6 bands: Blue, Green, Red, NIR, SWIR1, SWIR2
        # We read the first 6 bands.
        # Ensure your Tiff has at least 6 bands!
        img = src.read([1, 2, 3, 4, 5, 6]) 
        
        # Normalize: Sentinel-2 is uint16 (0-10000). AI needs 0-1 float.
        img = img.astype('float32') / 10000.0
        
        # Resize/Crop to 224x224 for the model
        # For this test, we just take the center crop 224x224
        c, h, w = img.shape
        start_h = (h - 224) // 2
        start_w = (w - 224) // 2
        
        if h < 224 or w < 224:
            raise ValueError(f"Image too small ({h}x{w}). Needs to be at least 224x224.")
            
        crop = img[:, start_h:start_h+224, start_w:start_w+224]
        
        # Convert to Tensor (1, 6, 224, 224)
        return torch.from_numpy(crop).unsqueeze(0)

def main():
    # 1. Settings
    tiff_path = r"C:\Users\dev\Desktop\SpectraLoRA\timeseries_001_2022-01.tif"
    device = torch.device("cpu")
    
    if not os.path.exists(tiff_path):
        print(f"❌ File not found: {tiff_path}")
        return

    # 2. Load Real Data
    try:
        image_tensor = load_real_tiff(tiff_path).to(device)
        print(f"✅ Data Loaded. Shape: {image_tensor.shape}")
    except Exception as e:
        print(f"❌ Data Load Error: {e}")
        return

    # 3. Load Model
    print("⏳ Loading Model...")
    model = load_prithvi_model()
    model = inject_spectra_lora(model)
    model = patch_model_for_context(model)
    model.eval()

    # 4. Run Physics Engine
    print("🔬 Analyzing Spectral Signature...")
    with torch.no_grad():
        z = get_spectral_fingerprint(image_tensor, SpectraConfig.BAND_MAP)
        SpectraContext.current_z = z
        
        # Print Real Physics Data
        z_val = z[0].numpy()
        print("\n" + "="*30)
        print(f"🌍 REAL PHYSICAL ANALYSIS")
        print("="*30)
        print(f"NDVI (Vegetation): {z_val[0]:.2f}")
        print(f"SAVI (Desert Veg): {z_val[1]:.2f}")
        print(f"NDWI (Water):      {z_val[2]:.2f}")
        print(f"NDBI (Urban):      {z_val[3]:.2f}")
        print(f"BSI  (Bare Soil):  {z_val[4]:.2f}")
        print("-" * 30)

        # Interpretation
        if z_val[0] > 0.4: print("👉 Diagnosis: High Vegetation (Forest/Crop)")
        elif z_val[2] > 0.3: print("👉 Diagnosis: Water Body")
        elif z_val[3] > 0.1: print("👉 Diagnosis: Urban/Built-up Area")
        else: print("👉 Diagnosis: Barren/Mixed")
        print("="*30)

    # 5. Run Inference
    print("🚀 Running Model Inference...")
    with torch.no_grad():
        output = model(image_tensor)
        print(f"✅ Inference Successful! Output Shape: {output.shape}")

if __name__ == "__main__":
    main()