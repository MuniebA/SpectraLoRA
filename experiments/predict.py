import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import rasterio
import os
import sys
import random
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

# Import the context patcher and decoder from train
from train import SpectraContext, patch_model_for_context

# --- 4-CLASS DECODER ---
class SimpleDecoder(nn.Module):
    def __init__(self, embed_dim=768, num_classes=4): # UPGRADED TO 4
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
    """Loads a 6-band chip and prepares it for the model and visualization."""
    with rasterio.open(filepath) as src:
        # Load all 6 bands and normalize
        raw_img = src.read().astype(np.float32) / 10000.0
        
    tensor_img = torch.from_numpy(raw_img).unsqueeze(0)
    
    # Extract RGB for visualization (Bands are: Blue=0, Green=1, Red=2)
    # We stack them as R, G, B
    rgb = np.stack([raw_img[2], raw_img[1], raw_img[0]], axis=-1)
    
    # Brighten the image for display (clip outliers)
    rgb = np.clip(rgb * 3.0, 0, 1) 
    return tensor_img, rgb

def run_visual_prediction(chip_path):
    device = torch.device("cpu")
    config = SpectraConfig()
    
    print(f"📂 Loading {chip_path}...")
    tensor_img, rgb_img = load_chip(chip_path)
    
    # 1. Build Model Architecture
    print("🏗️ Constructing Model...")
    encoder = load_prithvi_model()
    encoder = inject_spectra_lora(encoder)
    encoder = patch_model_for_context(encoder)
    decoder = SimpleDecoder(num_classes=4)
    model = nn.Sequential(encoder, decoder).to(device)
    
    # 2. Load Trained Weights
    weight_path = "spectra_lora_weights.pth"
    if os.path.exists(weight_path):
        print(f"⚖️ Loading trained weights from {weight_path}...")
        model.load_state_dict(torch.load(weight_path, map_location=device))
    else:
        print("⚠️ WARNING: No weights found. Running UNTRAINED model.")
        
    model.eval()

    # 3. Physics Engine
    print("🔬 Running Physics Router...")
    with torch.no_grad():
        z = get_spectral_fingerprint(tensor_img, config.BAND_MAP)
        SpectraContext.current_z = z
        
        # 4. Predict
        print("🚀 Predicting...")
        logits = model(tensor_img)
        # Convert logits [1, 3, 224, 224] to class predictions [224, 224]
        prediction = torch.argmax(logits, dim=1).squeeze().numpy()

    # 5. Visualize
    print("🎨 Generating Plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot 1: RGB Image
    ax1.imshow(rgb_img)
    ax1.set_title("True Satellite Image (RGB)")
    ax1.axis('off')
    
    # Plot 2: AI Prediction
    # 0: Barren (Light Gray), 1: Vegetation (Green), 2: Water (Blue), 3: Urban (Red)
    cmap = ListedColormap(['#d3d3d3', '#2ca02c', '#1f77b4', '#d62728'])
    im = ax2.imshow(prediction, cmap=cmap, vmin=0, vmax=3)
    ax2.set_title(f"SpectraLoRA Prediction\nContext (NDVI: {z[0,0]:.2f}, NDWI: {z[0,2]:.2f}, NDBI: {z[0,3]:.2f})")
    ax2.axis('off')
    
    # Legend
    cbar = plt.colorbar(im, ax=ax2, ticks=[0.375, 1.125, 1.875, 2.625], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(['Barren', 'Vegetation', 'Water', 'Urban'])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Pointing to the NEW test dataset folder
    test_folder = "test_dataset_224x224"
    
    # You can manually pick a file like "nw_000.tif" or "east_005.tif"
    # Or let the script randomly pick one for you each time you run it!
    test_files = [f for f in os.listdir(test_folder) if f.endswith('.tif')]
    
    if len(test_files) > 0:
        random_chip = random.choice(test_files)
        target_chip = os.path.join(test_folder, random_chip)
        
        print(f"🎲 Randomly selected test chip: {random_chip}")
        run_visual_prediction(target_chip)
    else:
        print(f"❌ No .tif files found in {test_folder}. Did you run create_dataset.py?")