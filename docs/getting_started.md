# Getting Started with SpectraLoRA

Welcome to the official documentation for **SpectraLoRA**, a Physics-Aware Parameter-Efficient Fine-Tuning (PEFT) library built for Geospatial Foundation Models. 
## The Problem with Standard LoRA in Geospatial AI
Standard Low-Rank Adaptation (LoRA) is an incredible tool for text models (LLMs). However, when applied to multi-spectral satellite imagery, standard LoRA treats all input channels blindly. It applies the same learned weight updates regardless of whether it is looking at the Sahara Desert, the Amazon Rainforest, or the concrete skyline of Doha.

## The SpectraLoRA Solution
SpectraLoRA introduces a **Physics-Aware Mixture-of-Experts (MoE)** architecture. Instead of one generic adapter, SpectraLoRA injects a bank of specialized LoRA adapters (e.g., an "Urban Expert," a "Water Expert," and a "Vegetation Expert"). 

Before the model processes an image patch, our **Physics Engine** calculates its Spectral Fingerprint (NDVI, NDWI, BSI, etc.). A tiny neural router then dynamically blends the adapters based on the physical properties of the terrain.

### Core Equation
The forward pass of a SpectraLoRALayer is defined as:

$$Output = Frozen(x) + \sum_{i=1}^{N} \left( Gate_i(z) \cdot Adapter_i(x) \right) \cdot \frac{\alpha}{r}$$

Where:
* $x$ is the input tensor.
* $z$ is the 5-dimensional physics context vector.
* $Gate_i(z)$ is the soft-routing probability for expert $i$.
* $Adapter_i(x)$ is the low-rank projection $B(A(x))$.

## Quick Installation

Install directly from PyPI:
```bash
pip install spectralora

```

## Basic Usage

```python
from spectra_lora import load_prithvi_model, inject_spectra_lora

# 1. Load the frozen foundation model
model = load_prithvi_model()

# 2. Inject the physics-aware adapters into the Attention layers
model = inject_spectra_lora(model)

# 3. Model is ready for your PyTorch training loop!

```
