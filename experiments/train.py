import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import rasterio
import numpy as np
import math
import sys

# Ensure we can import the library from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ---------------------------

# Import our library
from spectra_lora import (
    load_prithvi_model, 
    inject_spectra_lora, 
    SpectraConfig, 
    get_spectral_fingerprint,
    count_parameters,
    SpectraLoRALayer
)

# -------------------------------------------------------------------------
# 1. Real Satellite Dataset & Pseudo-Labeler
# -------------------------------------------------------------------------
class RealSatelliteDataset(Dataset):
    """ Loads real 224x224 .tif chips and creates physics-based training masks """
    def __init__(self, folder_path="dataset_224x224"):
        self.folder_path = folder_path
        if not os.path.exists(folder_path):
            raise RuntimeError(f"Folder '{folder_path}' not found! Run create_dataset.py first.")
            
        self.files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
        if len(self.files) == 0:
            raise RuntimeError(f"No .tif files found in {folder_path}!")
            
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        path = os.path.join(self.folder_path, self.files[idx])
        with rasterio.open(path) as src:
            # Read all 6 bands and normalize [0, 1]
            img = src.read().astype(np.float32) / 10000.0 
            
        img_tensor = torch.from_numpy(img)
        
        # --- GENERATE PSEUDO-LABELS FOR TRAINING ---
        # 0: Barren/Mixed, 1: Vegetation, 2: Water
        red, nir, green = img_tensor[2], img_tensor[3], img_tensor[1]
        ndvi = (nir - red) / (nir + red + 1e-8)
        ndwi = (green - nir) / (green + nir + 1e-8)
        
        mask = torch.zeros((224, 224), dtype=torch.long)
        mask[ndvi > 0.15] = 1  # Moderate vegetation threshold
        mask[(ndwi > 0.1) & (ndvi <= 0.15)] = 2 # Water threshold
        
        return img_tensor, mask

# -------------------------------------------------------------------------
# 2. The Decoder Head (Converts Tokens -> Image)
# -------------------------------------------------------------------------
class SimpleDecoder(nn.Module):
    """ Projects Prithvi's 1D tokens back into a 2D Segmentation Map """
    def __init__(self, embed_dim=768, num_classes=3):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Upsample 14x14 back to 224x224 (Scale factor 16)
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # x shape from Prithvi: [Batch, 197, 768]
        # Drop the CLS token (index 0)
        x = x[:, 1:, :] # -> [Batch, 196, 768]
        
        B, N, C = x.shape
        H = W = int(math.sqrt(N)) # sqrt(196) = 14
        
        # Reshape to Image format: [Batch, 768, 14, 14]
        x = x.transpose(1, 2).reshape(B, C, H, W) 
        
        # Decode to [Batch, 3, 224, 224]
        return self.decode(x)

# -------------------------------------------------------------------------
# 3. Context Bridge
# -------------------------------------------------------------------------
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

# -------------------------------------------------------------------------
# 4. The Training Loop
# -------------------------------------------------------------------------
def train_spectra_lora():
    config = SpectraConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Real-Data Training on {device}...")

    # Load Full Model (Encoder + Decoder)
    encoder = load_prithvi_model()
    encoder = inject_spectra_lora(encoder)
    encoder = patch_model_for_context(encoder)
    
    decoder = SimpleDecoder(num_classes=3)
    
    # Combine them
    model = nn.Sequential(encoder, decoder).to(device)
    
    # Optimizer (Only train Adapters, Gate, and Decoder!)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    # Load Real Data
    dataset = RealSatelliteDataset("dataset_224x224")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True) # Small batch for laptops
    
    model.train()
    num_epochs = 5
    
    for epoch in range(num_epochs):
        print(f"\n🌍 Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)
            
            # 1. Calculate Physics
            with torch.no_grad():
                z = get_spectral_fingerprint(images, config.BAND_MAP)
            SpectraContext.current_z = z
            
            # 2. Forward Pass
            optimizer.zero_grad()
            logits = model(images) # Encoder -> Decoder -> [B, 3, 224, 224]
            
            # 3. Backprop
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            print(f"   Batch {batch_idx+1}/{len(dataloader)}: Loss {loss.item():.4f} | Physics Context: NDVI={z[0,0]:.2f}")

        print(f"✅ Epoch {epoch+1} Complete. Avg Loss: {epoch_loss/len(dataloader):.4f}")
        
    # Save the adapter weights
    torch.save(model.state_dict(), "spectra_lora_weights.pth")
    print("💾 Model weights saved to 'spectra_lora_weights.pth'")

if __name__ == "__main__":
    train_spectra_lora()