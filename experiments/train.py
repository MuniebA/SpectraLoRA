import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os

# Import our library
from spectra_lora import (
    load_prithvi_model, 
    inject_spectra_lora, 
    SpectraConfig, 
    get_spectral_fingerprint,
    count_parameters,
    SpectraLoRALayer
)

"""
SpectraLoRA Trainer (Phase 4, Step 7)
-------------------------------------
The execution engine.
It handles:
1. Data Loading (Mock or Real).
2. Model Initialization & Surgery.
3. The "Context Injection" Patch (Crucial).
4. The Training Loop.
"""

# -------------------------------------------------------------------------
# 1. The Dataset (Mock Implementation)
# -------------------------------------------------------------------------
class SentinelDataset(Dataset):
    """
    A dummy dataset to simulate Satellite Tiffs.
    In a real scenario, use utils.load_geotiff here.
    """
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Simulate a 6-band Sentinel-2 image (224x224)
        # Random values [0, 1]
        image = torch.rand(6, 224, 224)
        
        # Simulate a Segmentation Mask (3 classes: Background, Crop, Water)
        mask = torch.randint(0, 3, (224, 224)).long()
        
        return image, mask

# -------------------------------------------------------------------------
# 2. The Context Manager (The Bridge)
# -------------------------------------------------------------------------
class SpectraContext:
    """
    Singleton to hold the physics vector 'z' during the forward pass.
    This allows layers deep in the network to access 'z' without
    Prithvi passing it down explicitly.
    """
    current_z = None

def patch_model_for_context(model):
    """
    Runtime Monkey-Patching.
    We wrap the 'forward' method of every SpectraLoRALayer instance
    so it grabs 'z' from SpectraContext automatically.
    """
    print("🔧 Patching model layers to accept global physics context...")
    
    for module in model.modules():
        if isinstance(module, SpectraLoRALayer):
            # We define a closure that binds the module
            original_forward = module.forward
            
            # The new forward only takes 'x', satisfying Prithvi's requirement
            def patched_forward(x):
                # Grab z from the global context
                z = SpectraContext.current_z
                if z is None:
                    raise ValueError("Context 'z' was not set before forward pass!")
                return original_forward(x, z)
            
            # Replace the method on this instance
            module.forward = patched_forward
            
    print("✅ Model patched successfully.")
    return model

# -------------------------------------------------------------------------
# 3. The Main Execution
# -------------------------------------------------------------------------
def train_spectra_lora():
    # A. Setup
    config = SpectraConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Training on {device}...")

    # B. Load & Modify Model
    model = load_prithvi_model()       # Step 5
    model = inject_spectra_lora(model) # Step 6
    
    # C. Apply the Context Patch
    model = patch_model_for_context(model)
    
    # D. Move to GPU & Freeze
    model.to(device)
    
    # E. Optimizer
    # IMPORTANT: Only pass parameters that require_grad (The Adapters + Gate)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Stats
    count_parameters(model)

    # F. Data Loader
    dataset = SentinelDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # G. Training Loop
    model.train()
    num_epochs = 3
    
    for epoch in range(num_epochs):
        print(f"\nExample Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)
            
            # --- PHASE 1: Extract Physics ---
            # We calculate z *before* the model runs
            with torch.no_grad():
                z = get_spectral_fingerprint(images, config.BAND_MAP)
            
            # --- Set the Context ---
            SpectraContext.current_z = z
            
            # --- Standard Training Step ---
            optimizer.zero_grad()
            
            # Now when model(images) is called:
            # 1. Prithvi runs standard logic.
            # 2. It hits a SpectraLoRALayer.
            # 3. The 'patched_forward' grabs 'SpectraContext.current_z'.
            # 4. It computes the Gate & Adapters.
            outputs = model(images)
            
            # Prithvi output is often [Batch, Tokens, Dim]. 
            # We need to reshape for Segmentation [Batch, Classes, H, W].
            # For this demo, we assume a simple projection head or reshape exists.
            # We'll simulate the output shape matching the mask for the loss.
            # (In a real run, you'd add a Decoder head here).
            
            # Mocking output for demonstration (since Prithvi is an encoder)
            # Just ensuring dimensions match for the loss function
            logits = torch.randn(4, 3, 224, 224).to(device) 
            
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"   Batch {batch_idx}: Loss {loss.item():.4f} | Physics Context: NDVI={z[0,0]:.2f}")

        print(f"Epoch {epoch+1} Complete. Avg Loss: {epoch_loss/len(dataloader):.4f}")

if __name__ == "__main__":
    train_spectra_lora()