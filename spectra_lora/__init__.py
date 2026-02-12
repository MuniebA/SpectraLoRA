"""
SpectraLoRA Library
-------------------
Physics-Aware Parameter-Efficient Fine-Tuning for Geospatial Foundation Models.

Usage:
    >>> from spectra_lora import load_prithvi_model, inject_spectra_lora
    >>> model = load_prithvi_model()
    >>> model = inject_spectra_lora(model)
"""

# 1. Configuration
from .config import SpectraConfig

# 2. The Core Physics Engine
from .spectral_ops import get_spectral_fingerprint

# 3. The Model wrappers (Loader + Injector)
from .model_wrapper import load_prithvi_model, inject_spectra_lora

# 4. The Custom Layer (exposed for advanced users who want to inspect it)
from .layers import SpectraLoRALayer

# 5. Utilities
from .utils import load_geotiff, count_parameters

# Define what gets imported when someone does 'from spectra_lora import *'
__all__ = [
    "SpectraConfig",
    "get_spectral_fingerprint",
    "load_prithvi_model",
    "inject_spectra_lora",
    "SpectraLoRALayer",
    "load_geotiff",
    "count_parameters"
]