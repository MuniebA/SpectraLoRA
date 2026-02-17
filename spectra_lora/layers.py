import torch
import torch.nn as nn
import math
from .config import SpectraConfig
from .gating_network import SpectralGate

"""
SpectraLoRA Layers (Phase 2 - DEBUG & SAFEGUARD VERSION)
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
        self.lora_A = nn.Linear(in_features, self.r, bias=False)
        self.lora_B = nn.Linear(self.r, out_features, bias=False)
        
        # 2. Dropout
        self.lora_dropout = nn.Dropout(p=config.LORA_DROPOUT)
        
        # 3. Initialization
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        # DEBUG: Print creation details to catch the "Imposter" layer
        # print(f"🔨 DEBUG: Creating SpectraLoRALayer. Wrapped: {self.in_features}->{self.out_features}")
        
        # 2. Freeze and Store
        self.original_layer = original_layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
            
        # 3. Create Router & Adapters
        self.gate = SpectralGate(config)
        self.adapters = nn.ModuleList([
            LoRAAdapter(self.in_features, self.out_features, config)
            for _ in range(config.NUM_ADAPTERS)
        ])

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        The Hybrid Forward Pass.
        """
        # 1. Base Prediction (Frozen)
        frozen_out = self.original_layer(x)
        
        # --- DEBUG SAFETY CHECK ---
        # If this layer is supposed to be QKV, it MUST output 2304.
        # If it outputs 768, we have caught the bug.
        if self.out_features == 2304 and frozen_out.shape[-1] != 2304:
             print(f"🚨 FATAL: SpectraLoRALayer metadata says 2304, but original_layer returned {frozen_out.shape[-1]}!")

        # Path B: The Physics Router
        gate_weights = self.gate(z)
        
        # Path C: The Adapter Bank
        weighted_adapter_sum = torch.zeros_like(frozen_out)
        
        for i, adapter in enumerate(self.adapters):
            adapter_out = adapter(x)
            
            # --- SHAPE MISMATCH GUARD ---
            if adapter_out.shape != frozen_out.shape:
                # If this happens, it means the Adapter was built with wrong dimensions
                print(f"🚨 SHAPE MISMATCH: Frozen={frozen_out.shape}, Adapter={adapter_out.shape}")
                # Emergency reshape to prevent crash (debugging only)
                if adapter_out.shape[-1] == 768 and frozen_out.shape[-1] == 2304:
                     # This effectively disables the adapter but keeps the model running
                     adapter_out = adapter_out.repeat(1, 1, 3) 
            
            w = gate_weights[:, i].view(-1, 1, 1)
            weighted_adapter_sum += w * adapter_out
            
        return frozen_out + weighted_adapter_sum