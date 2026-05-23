import torch
import torch.nn as nn
from .config import SpectraConfig

"""
SpectraLoRA Gating Network (Phase 2, Step 3)
--------------------------------------------
The 'Router' Module.
This tiny neural network decides which Adapter Expert to use based on the
physics context vector 'z'.

Architecture:
    Input (z) -> Linear -> ReLU -> Linear -> Softmax -> Output (Weights)
"""

class SpectralGate(nn.Module):
    def __init__(self, config: SpectraConfig = SpectraConfig):
        super().__init__()
        
        # 1. Store configuration settings
        self.temperature = config.GATE_TEMPERATURE
        
        # 2. Define the MLP (Multi-Layer Perceptron)
        # Layer 1: Expansion. Takes the 5 physics indices and maps them to a hidden dimension (32).
        # Layer 2: Decision. Maps the hidden dimension to the number of experts (3).
        self.net = nn.Sequential(
            nn.Linear(config.PHYSICS_CONTEXT_DIM, config.GATE_HIDDEN_DIM),
            nn.ReLU(),  # Non-linearity allows it to learn complex boundaries (e.g., "High NDVI AND Low SAVI")
            nn.Linear(config.GATE_HIDDEN_DIM, config.NUM_ADAPTERS)
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Router.

        Args:
            z (Tensor): The Spectral Fingerprint Vector.
                        Shape: (Batch_Size, PHYSICS_CONTEXT_DIM)
                        Example: A batch of 4 images -> (4, 5)

        Returns:
            weights (Tensor): The mixing weights for the adapters.
                            Shape: (Batch_Size, NUM_ADAPTERS)
                            Example: (4, 3) where each row sums to 1.0
        """
        # 1. Pass through the MLP to get raw "logits" (scores)
        # Shape: (Batch, NUM_ADAPTERS)
        logits = self.net(z)
        
        # 2. Apply Temperature Scaling
        # Dividing by temperature allows us to control the "sharpness" of the decision.
        # - High Temp (>1): Weights are flatter (e.g., [0.33, 0.33, 0.33]) -> "Use everyone a little bit"
        # - Low Temp (<1): Weights are sharper (e.g., [0.01, 0.98, 0.01]) -> "Use only the specialist"
        scaled_logits = logits / self.temperature
        
        # 3. Softmax
        # Converts scores into probabilities (0.0 to 1.0) that sum to 1.
        weights = torch.softmax(scaled_logits, dim=1)
        
        return weights