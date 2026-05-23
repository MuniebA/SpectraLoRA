"""
SpectraLoRA Configuration (Phase 1, Step 2)
-------------------------------------------
This file acts as the central 'Control Room' for the library.
It defines:
1. The Band Map (matching Prithvi's expected input).
2. The Physics Engine dimensions (connecting to spectral_ops).
3. The LoRA Hyperparameters (Rank, Alpha).
4. The Gating Network Architecture (Hidden layers).

By keeping these here, we avoid hardcoding 'magic numbers' in the complex layers.
"""

from dataclasses import dataclass

@dataclass
class SpectraConfig:
    # -------------------------------------------------------------------------
    # 1. The Data Interface (The Band Map)
    # -------------------------------------------------------------------------
    # The Prithvi-100M model is pre-trained on HLS (Harmonized Landsat Sentinel) 
    # data. It expects a 6-channel input in this specific order:
    # [Blue, Green, Red, NIR, SWIR1, SWIR2]
    #
    # This dictionary maps our "Physics Names" to the "Tensor Index".
    # NOTE: If using standard Sentinel-2 (L2A) with 12 bands, change indices here.
    BAND_MAP = {
        'BLUE': 0,
        'GREEN': 1,
        'RED': 2,
        'NIR': 3,
        'SWIR': 4,  # Short-Wave Infrared 1 (Essential for NDBI)
        'SWIR2': 5  # Short-Wave Infrared 2 (Essential for BSI)
    }

    # -------------------------------------------------------------------------
    # 2. Physics Engine Settings
    # -------------------------------------------------------------------------
    # This must match the output of 'get_spectral_fingerprint' in spectral_ops.py.
    # We calculate 5 indices: [NDVI, SAVI, NDWI, NDBI, BSI]
    PHYSICS_CONTEXT_DIM = 5

    # -------------------------------------------------------------------------
    # 3. Adapter Bank Settings (The "Experts")
    # -------------------------------------------------------------------------
    # Rank (r): Controls the capacity of each adapter.
    # - r=4 is standard for "Efficient" tuning.
    # - r=8 or 16 gives more power but uses more VRAM.
    LORA_R = 4
    
    # Alpha: Scaling factor. Standard practice is alpha = 2 * r.
    LORA_ALPHA = 8
    
    # Dropout: Prevents overfitting within the adapters.
    LORA_DROPOUT = 0.05
    
    # Number of Experts:
    # We use 3 to broadly cover: [1. Vegetation, 2. Urban/Rock, 3. Water/Soil]
    NUM_ADAPTERS = 3

    # -------------------------------------------------------------------------
    # 4. Gating Network Architecture (The "Router")
    # -------------------------------------------------------------------------
    # The Router is a tiny MLP (Multi-Layer Perceptron).
    # Input: 5 (Physics Context) -> Hidden -> Output: 3 (Num Adapters)
    GATE_HIDDEN_DIM = 32  # Small enough to be fast, big enough to be smart.
    
    # Temperature: Controls how "sharp" the decisions are.
    # - Low temp (0.1): "Hard" switching (Only pick 1 adapter).
    # - High temp (1.0): "Soft" mixing (Blend all adapters).
    GATE_TEMPERATURE = 1.0