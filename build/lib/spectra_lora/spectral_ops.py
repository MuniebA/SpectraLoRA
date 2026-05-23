import torch
import torch.nn.functional as F

"""
SpectraLoRA Physics Engine (Phase 1) - UPDATED
----------------------------------------------
Includes arid-region optimizations (SAVI, BSI) to separate 
sand from concrete and detect sparse vegetation.
"""

EPSILON = 1e-8

def calculate_normalized_difference(band_a: torch.Tensor, band_b: torch.Tensor) -> torch.Tensor:
    numerator = band_a - band_b
    denominator = band_a + band_b + EPSILON
    return numerator / denominator

def calculate_ndvi(red: torch.Tensor, nir: torch.Tensor) -> torch.Tensor:
    """Standard Vegetation Index (Best for dense forests)."""
    return calculate_normalized_difference(nir, red)

def calculate_savi(red: torch.Tensor, nir: torch.Tensor, L: float = 0.5) -> torch.Tensor:
    """
    Soil-Adjusted Vegetation Index (SAVI).
    Physics: In deserts (Qatar/Sudan), bright sand reflects light that drowns out 
    the chlorophyll signal. The 'L' factor corrects for this soil brightness.
    """
    numerator = (nir - red) * (1 + L)
    denominator = (nir + red + L) + EPSILON
    return numerator / denominator

def calculate_ndwi(green: torch.Tensor, nir: torch.Tensor) -> torch.Tensor:
    """Standard Water Index."""
    return calculate_normalized_difference(green, nir)

def calculate_ndbi(nir: torch.Tensor, swir: torch.Tensor) -> torch.Tensor:
    """Built-Up Index (Concrete/Asphalt)."""
    return calculate_normalized_difference(swir, nir)

def calculate_bsi(red: torch.Tensor, blue: torch.Tensor, nir: torch.Tensor, swir: torch.Tensor) -> torch.Tensor:
    """
    Bare Soil Index (BSI).
    Physics: Distinguishes 'Bare Earth' (Sand/Dirt) from 'Built Structures' (Concrete).
    Essential for accurate mapping in arid cities like Doha or Khartoum.
    Formula: ((SWIR + Red) - (NIR + Blue)) / ((SWIR + Red) + (NIR + Blue))
    """
    numerator = (swir + red) - (nir + blue)
    denominator = (swir + red) + (nir + blue) + EPSILON
    return numerator / denominator

def get_spectral_fingerprint(
    x: torch.Tensor, 
    band_indices: dict
) -> torch.Tensor:
    """
    Generates a 5-dimensional physics context vector 'z'.
    """
    # 1. Extract bands
    blue  = x[:, band_indices['BLUE'], :, :]
    red   = x[:, band_indices['RED'], :, :]
    green = x[:, band_indices['GREEN'], :, :]
    nir   = x[:, band_indices['NIR'], :, :]
    swir  = x[:, band_indices['SWIR'], :, :]

    # 2. Calculate Physics Maps
    ndvi = calculate_ndvi(red, nir)
    savi = calculate_savi(red, nir)  # NEW: Better for sparse shrubs
    ndwi = calculate_ndwi(green, nir)
    ndbi = calculate_ndbi(nir, swir) # City
    bsi  = calculate_bsi(red, blue, nir, swir) # NEW: Sand/Desert

    # 3. Global Average Pooling (Summarize the patch)
    # We take the mean of each map to get a single number per image
    z_ndvi = torch.mean(ndvi, dim=(1, 2))
    z_savi = torch.mean(savi, dim=(1, 2))
    z_ndwi = torch.mean(ndwi, dim=(1, 2))
    z_ndbi = torch.mean(ndbi, dim=(1, 2))
    z_bsi  = torch.mean(bsi, dim=(1, 2))

    # 4. Stack into Vector 'z' (Batch x 5)
    z = torch.stack([z_ndvi, z_savi, z_ndwi, z_ndbi, z_bsi], dim=1)

    return z