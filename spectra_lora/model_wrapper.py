import torch
import os
import sys
import importlib.util
from huggingface_hub import hf_hub_download
from .layers import SpectraLoRALayer
from .config import SpectraConfig
import torch.nn as nn
from functools import partial

"""
SpectraLoRA Model Wrapper (Phase 3 - FIXED & INSTRUMENTED)
"""

# -------------------------------------------------------------------------
# A. Minimal ViT Architecture (Prithvi Backbone)
# -------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """ 3D Patch Embedding for temporal data """
    def __init__(self, img_size=224, patch_size=16, in_chans=6, embed_dim=768, num_frames=1):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(1, patch_size, patch_size), stride=(1, patch_size, patch_size))

    def forward(self, x):
        if x.ndim == 4:
            x = x.unsqueeze(2) 
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Standard QKV: Input 768 -> Output 2304 (768*3)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        
        # Calculate QKV
        qkv_out = self.qkv(x)
        
        # Diagnostic Check during Inference (can be removed in prod)
        if qkv_out.shape[-1] != C * 3:
             print(f"❌ CRITICAL ERROR IN FORWARD: qkv output shape is {qkv_out.shape}, expected last dim {C*3}")
        
        qkv = qkv_out.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity() 
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PrithviViT(nn.Module):
    def __init__(self, 
                 img_size=224, patch_size=16, in_chans=6, 
                 embed_dim=768, depth=12, num_heads=12, 
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=0.0)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = x + self.pos_embed[:, 1:, :]
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

# -------------------------------------------------------------------------
# B. The Loader Function
# -------------------------------------------------------------------------

def load_prithvi_model(repo_id: str = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M", filename: str = "Prithvi_100M.pt"):
    print(f"🌍 SpectraLoRA: Downloading Prithvi weights from {repo_id}...")
    try:
        weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as e:
        print(f"❌ Failed to download weights: {e}")
        raise e

    print("🏗️ Constructing Model Architecture (Local PrithviViT)...")
    model = PrithviViT(embed_dim=768, depth=12, num_heads=12)

    print("⚖️ Loading Weights...")
    checkpoint = torch.load(weights_path, map_location='cpu')
    if 'model' in checkpoint: state_dict = checkpoint['model']
    else: state_dict = checkpoint

    encoder_dict = {k: v for k, v in state_dict.items() if 'decoder' not in k and 'mask_token' not in k}
    msg = model.load_state_dict(encoder_dict, strict=False)
    print(f"✅ Model Loaded. (Ignored decoder weights: {len(msg.missing_keys) > 0})")
    return model

# -------------------------------------------------------------------------
# C. The Surgeon Function (Injector) - UPDATED & FIXED
# -------------------------------------------------------------------------

def inject_spectra_lora(model: torch.nn.Module, config: SpectraConfig = SpectraConfig):
    print("✨ SpectraLoRA: Beginning surgery on the model...")
    layers_modified = 0
    
    # We recursively traverse the model to find Linear layers to replace
    for name, module in model.named_children():
        _recursive_injection(model, name, module, config, counter_ref=[layers_modified])
        
    print(f"✅ Surgery Complete. SpectraLoRA layers injected: {layers_modified}")
    return model

def _recursive_injection(parent_module, child_name, child_module, config, counter_ref):
    """
    Recursive function to find and replace QKV and Proj layers.
    Includes strict dimension checks to avoid wrapping malformed layers.
    """
    # 1. If the module has children (like a Block or Attention layer), recurse down
    if len(list(child_module.children())) > 0:
        for name, grandchild in child_module.named_children():
            _recursive_injection(child_module, name, grandchild, config, counter_ref)
            
    # 2. If it is a Linear layer, check if it's a target for LoRA
    elif isinstance(child_module, torch.nn.Linear):
        
        # --- CASE A: QKV Layer (Attention) ---
        if "qkv" in child_name:
            # SAFETY CHECK: QKV must output 3x the input dimension (Query, Key, Value)
            # Standard Prithvi: 768 -> 2304
            expected_out = child_module.in_features * 3
            
            if child_module.out_features != expected_out:
                print(f"⚠️ SKIPPING QKV WRAP: Layer '{child_name}' has dimensions {child_module.in_features}->{child_module.out_features}. "
                      f"Expected x3 expansion (->{expected_out}). Skipping to prevent crash.")
                return

            print(f"   -> 💉 Injecting LoRA into QKV:  {child_module.in_features} -> {child_module.out_features}")
            new_layer = SpectraLoRALayer(child_module, config)
            setattr(parent_module, child_name, new_layer)
            counter_ref[0] += 1
            
        # --- CASE B: Projection Layer (Output of Attention) ---
        # elif "proj" in child_name:
        #     print(f"   -> 💉 Injecting LoRA into PROJ: {child_module.in_features} -> {child_module.out_features}")
        #     new_layer = SpectraLoRALayer(child_module, config)
        #     setattr(parent_module, child_name, new_layer)
        #     counter_ref[0] += 1