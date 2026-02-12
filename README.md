# SpectraLoRA: Physics-Aware Fine-Tuning for Geospatial Foundation Models

![Status](https://img.shields.io/badge/Status-Research_Prototype-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)

**SpectraLoRA** is a specialized Parameter-Efficient Fine-Tuning (PEFT) library designed for multispectral satellite imagery. Unlike standard LoRA, which treats all input channels equally, SpectraLoRA introduces a **Physics-Aware Gating Mechanism** that dynamically routes information to specialized adapters based on the spectral signature of the terrain (e.g., Vegetation, Water, Urban).

This architecture is designed to fine-tune massive Geospatial Foundation Models (like **IBM/NASA Prithvi-100M**) on consumer hardware while enforcing physical consistency in predictions.

---

## 🏗️ System Architecture

SpectraLoRA operates as a "Sidecar" to the frozen foundation model. It intercepts data flow to inject physics context without altering the pre-trained weights.



### Key Innovations
1.  **Spectral Fingerprinting**: Calculates real-time physics indices (NDVI, NDWI, BSI) before the model runs.
2.  **Mixture-of-Experts (MoE)**: A bank of specialized Low-Rank Adapters (e.g., one for forests, one for cities).
3.  **Dynamic Gating**: A lightweight router that blends adapters based on the material properties of the image patch.

---

## 📂 Project Structure

```text
SpectraLoRA/
├── spectra_lora/               # The Core Library
│   ├── __init__.py             # API Exporter
│   ├── config.py               # Phase 1: Global Configuration & Band Maps
│   ├── spectral_ops.py         # Phase 1: The Physics Engine (NDVI/BSI Math)
│   ├── gating_network.py       # Phase 2: The Neural Router (MLP)
│   ├── layers.py               # Phase 2: The Custom SpectraLoRALayer
│   ├── model_wrapper.py        # Phase 3: The Surgeon (Model Loader & Injector)
│   └── utils.py                # Helpers: GeoTIFF Loading & Visualization
│
├── experiments/                # Execution Scripts
│   ├── train.py                # Phase 4: Training Loop & Context Patching
│   └── evaluate.py             # Phase 4: Physics-Aware Evaluation Metrics
│
├── requirements.txt            # Dependencies
└── README.md                   # This file

```

---

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt

```

### Usage (Python API)

Use SpectraLoRA to upgrade a standard Prithvi model with physics-aware adapters:

```python
from spectra_lora import load_prithvi_model, inject_spectra_lora

# 1. Load the frozen Foundation Model (Prithvi-100M)
model = load_prithvi_model()

# 2. Perform Surgery: Inject SpectraLoRA layers
model = inject_spectra_lora(model)

# 3. The model is now ready for training!
# Only the adapter parameters (approx 1-5%) are trainable.

```

---

## 📖 Detailed File Guide

### 1. The Physics Core (`spectra_lora/`)

* **`spectral_ops.py`**: The mathematical heart of the library. It takes raw satellite tensors and calculates 5 physics indices: **NDVI** (Vegetation), **SAVI** (Soil-Adjusted Vegetation), **NDWI** (Water), **NDBI** (Built-up), and **BSI** (Bare Soil). This ensures the model "sees" physics, not just pixels.
* **`config.py`**: The blueprint. Defines the **Band Map** (Blue=0, NIR=3, etc.) to ensure calculations are accurate for the Prithvi model. It also sets LoRA hyperparameters (Rank=4, Alpha=8).
* **`utils.py`**: Handles the messy parts of Geospatial AI. Contains `load_geotiff` to normalize 16-bit satellite data into float32 tensors, and visualization tools to debug the spectral fingerprint.

### 2. The Neural Architecture (`spectra_lora/`)

* **`gating_network.py`**: A tiny Multi-Layer Perceptron (MLP). It takes the 5-dimensional physics vector from `spectral_ops` and outputs soft mixing weights (e.g., `[0.8, 0.1, 0.1]`) to control the adapters.
* **`layers.py`**: The custom PyTorch module. It replaces standard Linear layers. It contains:
* The **Frozen** original weights.
* The **Adapter Bank** (3 parallel LoRA experts).
* The **Gating Network**.
* **Logic**: .



### 3. The Integration (`spectra_lora/`)

* **`model_wrapper.py`**: The "Surgeon." It downloads the Prithvi architecture code dynamically from Hugging Face (since it's not in standard `transformers`), loads the weights, and recursively swaps Attention layers with `SpectraLoRALayer`.
* **`__init__.py`**: Exposes the high-level API so you can import functions cleanly.

### 4. Experiments (`experiments/`)

* **`train.py`**: The training engine. It includes a critical **Runtime Patch** (`patch_model_for_context`) that allows the model to access the global "Physics Context" without rewriting the original Prithvi code.
* **`evaluate.py`**: A custom validator. Beyond standard **mIoU**, it calculates a **"Physics Violation Score"**—counting how many times the model predicted "Vegetation" in a pixel where the NDVI was clearly "Desert."

---

## 📊 Citation

If you use this code for your research, please cite:

```bibtex
@software{spectralora2026,
  author = {Your Name},
  title = {SpectraLoRA: Band-Specific Low-Rank Adaptation for Geospatial Foundation Models},
  year = {2026},
  url = {[https://github.com/MuniebA/spectra-lora](https://github.com/yourusername/spectra-lora)}
}
```
