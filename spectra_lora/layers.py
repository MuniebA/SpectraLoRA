import torch
import torch.nn as nn
import math
from .config import SpectraConfig
from .gating_network import SpectralGate

"""
SpectraLoRA Layers (Phase 2, Step 4)
------------------------------------
This module defines the 'Replacement Layer'.
It wraps the original frozen layer and adds the 'Mixture of Experts' LoRA sidecar.
"""

class LoRAAdapter(nn.Module):
    """
    A single LoRA Expert.
    Math: Output = Up(Down(x)) * Scaling
    """
    def __init__(self, in_features: int, out_features: int, config: SpectraConfig):
        super().__init__()
        
        self.r = config.LORA_R
        self.scaling = config.LORA_ALPHA / self.r
        
        # 1. The Low-Rank Matrices
        # Down Projection: Compresses data (Dim -> r)
        self.lora_A = nn.Linear(in_features, self.r, bias=False)
        # Up Projection: Expands data back (r -> Dim)
        self.lora_B = nn.Linear(self.r, out_features, bias=False)
        
        # 2. Dropout (Regularization)
        self.lora_dropout = nn.Dropout(p=config.LORA_DROPOUT)
        
        # 3. Initialization (CRITICAL)
        # We initialize 'A' with random Gaussian noise and 'B' with Zeros.
        # Why? So that at step 0, the adapter output is exactly 0.
        # This ensures the model behaves exactly like the pre-trained Prithvi 
        # at the start of training, preventing a "shock" to the gradients.
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard LoRA path: Dropout -> Down -> Up -> Scale
        x = self.lora_dropout(x)
        x = self.lora_A(x)
        x = self.lora_B(x)
        return x * self.scaling


class SpectraLoRALayer(nn.Module):
    """
    The Main Replacement Layer.
    Wraps a frozen Linear layer and adds the physics-aware sidecar.
    """
    def __init__(self, original_layer: nn.Linear, config: SpectraConfig = SpectraConfig):
        super().__init__()
        
        # 1. Inspect the original layer
        # We need to know its shape to build matching adapters.
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        # 2. Freeze and Store the Original Layer
        # We keep it as an attribute so we can use it in the forward pass.
        self.original_layer = original_layer
        for param in self.original_layer.parameters():
            param.requires_grad = False  # FREEZE IT!
            
        # 3. Create the "Router" (Gating Network)
        # This will decide which adapter to use based on physics.
        self.gate = SpectralGate(config)
        
        # 4. Create the "Adapter Bank" (Mixture of Experts)
        # We create a list of adapters (e.g., 3 independent experts).
        self.adapters = nn.ModuleList([
            LoRAAdapter(self.in_features, self.out_features, config)
            for _ in range(config.NUM_ADAPTERS)
        ])

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        The Hybrid Forward Pass.
        
        Args:
            x (Tensor): The image features (from the previous layer).
                        Shape: (Batch, Tokens, Dim)
            z (Tensor): The Physics Context Vector (from spectral_ops).
                        Shape: (Batch, 5)
        """
        # Path A: The Frozen Giant (Original Prithvi)
        # This is the "Base Prediction"
        frozen_out = self.original_layer(x)
        
        # Path B: The Physics Router
        # The gate looks at 'z' (NDVI/NDWI) and outputs weights (e.g., [0.8, 0.1, 0.1])
        # Shape: (Batch, Num_Adapters)
        gate_weights = self.gate(z)
        
        # Path C: The Adapter Bank
        # We run 'x' through all experts.
        # Note: In a highly optimized version, we might use "Top-K" to only run 1 or 2.
        # For simplicity/research, we run all 3 and mix them.
        adapter_outputs = []
        for adapter in self.adapters:
            adapter_outputs.append(adapter(x)) 
            # Each output is (Batch, Tokens, Dim)
        
        # Path D: Dynamic Mixing (The Summation)
        # We need to multiply each adapter's output by its gate weight.
        # Since 'x' has 3 dimensions (Batch, Tokens, Dim) and weights has 2 (Batch, Adapters),
        # we need to reshape weights to broadcast correctly.
        
        weighted_adapter_sum = torch.zeros_like(frozen_out)
        
        for i, output in enumerate(adapter_outputs):
            # Extract weight for this adapter: Shape (Batch, 1, 1)
            w = gate_weights[:, i].view(-1, 1, 1)
            
            # Weighted Sum: Accumulate (Weight * Adapter_Output)
            weighted_adapter_sum += w * output
            
        # Final Merge
        return frozen_out + weighted_adapter_sum