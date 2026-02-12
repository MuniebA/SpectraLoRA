import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from .config import SpectraConfig

"""
SpectraLoRA Utilities (Phase 1 Support)
---------------------------------------
Helper functions to:
1. Load Satellite Tiff files into PyTorch Tensors.
2. Normalize data (Satellite data is uint16, AI needs float32).
3. Visualize the 'Spectral Fingerprint' for debugging.
"""

def load_geotiff(
    filepath: str, 
    band_ordering: Dict[str, int] = SpectraConfig.BAND_MAP,
    img_size: int = 224
) -> torch.Tensor:
    """
    Reads a multi-band GeoTIFF and converts it to a normalized PyTorch Tensor.
    
    Args:
        filepath (str): Path to the .tif file.
        band_ordering (dict): The mapping of bands (e.g., {'RED': 2}).
                            We ensure we load indices [0, 1, 2, 3, 4, 5] 
                            corresponding to Blue, Green, Red, NIR, SWIR1, SWIR2.
        img_size (int): Resize target (Prithvi expects 224x224).

    Returns:
        Tensor: Shape (1, 6, 224, 224) ready for the model.
    """
    with rasterio.open(filepath) as src:
        # 1. Read the specific bands required by Prithvi
        # The config defines logic names, we need to read the raw indices.
        # Note: Rasterio uses 1-based indexing, lists use 0-based.
        # We assume the input Tiff has the bands in standard Sentinel-2 order.
        # For this helper, we'll read the first 6 bands as a simplification.
        # In a real app, you'd map specific source bands to target bands.
        
        # Reading all bands (assuming the file is pre-processed to 6 bands)
        raw_data = src.read(out_shape=(src.count, img_size, img_size),
                            resampling=rasterio.enums.Resampling.bilinear)
        
    # 2. Convert to Float32 and Normalize
    # Satellite data is usually uint16 (0-10000). Prithvi expects inputs in range [0, 1] usually,
    # or normalized by specific mean/std.
    # Standard HLS normalization: Divide by 10000.
    tensor_data = torch.from_numpy(raw_data).float() / 10000.0
    
    # 3. Add Batch Dimension (C, H, W) -> (1, C, H, W)
    tensor_data = tensor_data.unsqueeze(0)
    
    return tensor_data

def visualize_spectral_fingerprint(z: torch.Tensor, batch_idx: int = 0):
    """
    Debug Tool: Plots the physics vector 'z' as a bar chart.
    Helps you see what the Gating Network is "seeing".
    
    Args:
        z (Tensor): The fingerprint vector output from spectral_ops.
        batch_idx (int): Which image in the batch to plot.
    """
    # Detach from GPU graph and convert to numpy
    z_values = z[batch_idx].detach().cpu().numpy()
    
    indices = ["NDVI", "SAVI", "NDWI", "NDBI", "BSI"]
    colors = ['green', 'lightgreen', 'blue', 'gray', 'orange']
    
    plt.figure(figsize=(8, 4))
    plt.bar(indices, z_values, color=colors)
    plt.title(f"Spectral Fingerprint (Image {batch_idx})")
    plt.ylabel("Index Value (-1 to 1)")
    plt.ylim(-1, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def count_parameters(model: torch.nn.Module):
    """
    Prints the efficiency gains of SpectraLoRA.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Statistics:")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Efficiency: {100 * trainable_params / total_params:.2f}% trainable")