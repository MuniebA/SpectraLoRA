import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import rasterio
import os
import sys
import math

# Ensure library is found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spectra_lora import (
    load_prithvi_model, 
    inject_spectra_lora, 
    SpectraConfig, 
    get_spectral_fingerprint,
    SpectraLoRALayer
)
from train import SpectraContext, patch_model_for_context

# --- 4-CLASS DECODER ---
class SimpleDecoder(nn.Module):
    def __init__(self, embed_dim=768, num_classes=4):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = x[:, 1:, :] 
        B, N, C = x.shape
        H = W = int(math.sqrt(N)) 
        x = x.transpose(1, 2).reshape(B, C, H, W) 
        return self.decode(x)

def load_chip(filepath):
    with rasterio.open(filepath) as src:
        raw_img = src.read().astype(np.float32) / 10000.0
    tensor_img = torch.from_numpy(raw_img).unsqueeze(0)
    rgb = np.stack([raw_img[2], raw_img[1], raw_img[0]], axis=-1)
    rgb = np.clip(rgb * 3.0, 0, 1) 
    return tensor_img, rgb

def run_batch_predictions(input_folder, output_folder):
    device = torch.device("cpu")
    config = SpectraConfig()
    
    # 1. Build and Load Model ONCE
    print("🏗️ Constructing Model...")
    encoder = load_prithvi_model()
    encoder = inject_spectra_lora(encoder)
    encoder = patch_model_for_context(encoder)
    decoder = SimpleDecoder(num_classes=4) 
    model = nn.Sequential(encoder, decoder).to(device)
    
    weight_path = "spectra_lora_weights.pth"
    if os.path.exists(weight_path):
        print(f"⚖️ Loading trained weights from {weight_path}...")
        model.load_state_dict(torch.load(weight_path, map_location=device))
    else:
        print("⚠️ WARNING: No weights found. Running UNTRAINED model.")
        
    model.eval()

    # 2. Setup Directories and Files
    os.makedirs(output_folder, exist_ok=True)
    test_files = [f for f in os.listdir(input_folder) if f.endswith('.tif')]
    
    if len(test_files) == 0:
        print(f"❌ No .tif files found in {input_folder}.")
        return

    print(f"🚀 Starting batch prediction on {len(test_files)} images...")
    cmap = ListedColormap(['#d3d3d3', '#2ca02c', '#1f77b4', '#d62728'])

    # 3. Loop through all images
    for i, filename in enumerate(test_files):
        chip_path = os.path.join(input_folder, filename)
        print(f"   -> Processing [{i+1}/{len(test_files)}]: {filename}")
        
        tensor_img, rgb_img = load_chip(chip_path)

        with torch.no_grad():
            z = get_spectral_fingerprint(tensor_img, config.BAND_MAP)
            SpectraContext.current_z = z
            
            logits = model(tensor_img)
            prediction = torch.argmax(logits, dim=1).squeeze().numpy()

        # 4. Generate and Save Plot (No pop-ups!)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1.imshow(rgb_img)
        ax1.set_title(f"True Satellite Image ({filename})")
        ax1.axis('off')
        
        im = ax2.imshow(prediction, cmap=cmap, vmin=0, vmax=3)
        ax2.set_title(f"SpectraLoRA Prediction\nContext (NDVI: {z[0,0]:.2f}, NDWI: {z[0,2]:.2f}, NDBI: {z[0,3]:.2f})")
        ax2.axis('off')
        
        cbar = plt.colorbar(im, ax=ax2, ticks=[0.375, 1.125, 1.875, 2.625], fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels(['Barren', 'Vegetation', 'Water', 'Urban'])
        
        plt.tight_layout()
        
        # Save to the output folder instead of displaying
        save_path = os.path.join(output_folder, f"pred_{filename.replace('.tif', '.png')}")
        plt.savefig(save_path, dpi=150)
        plt.close(fig) # Closes the figure to free up memory
        
    print(f"\n🎉 All predictions saved to the '{output_folder}' folder!")

if __name__ == "__main__":
    # Pointing to the Training Dataset
    target_input_folder = "test_dataset_224x224"
    
    # The folder where all the images will be saved
    target_output_folder = "predictions_output"
    
    run_batch_predictions(target_input_folder, target_output_folder)