import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our library
from spectra_lora import (
    load_prithvi_model, 
    inject_spectra_lora, 
    SpectraConfig, 
    get_spectral_fingerprint,
    SpectraLoRALayer
)
import torch.nn as nn

# Import the real dataset and decoder from train.py
from train import SpectraContext, patch_model_for_context, RealSatelliteDataset, SimpleDecoder

class PhysicsEvaluator:
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))
        self.physics_violations = 0
        self.total_pixels = 0

    def update(self, preds, targets, ndvi_map):
        preds = preds.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()
        
        # 1. Standard IoU Matrix
        mask = (targets >= 0) & (targets < self.num_classes)
        self.confusion_matrix += np.bincount(
            self.num_classes * targets[mask].astype(int) + preds[mask].astype(int),
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        
        # 2. SpectraLoRA Physics Consistency Check
        pred_tensor = torch.from_numpy(preds).view(-1)
        ndvi_tensor = ndvi_map.cpu().view(-1)
        
        # Violation: Model guessed Vegetation (1) but real NDVI is less than 0.1 (Barren)
        veg_pixels = (pred_tensor == 1) 
        violations = veg_pixels & (ndvi_tensor < 0.1)
        
        self.physics_violations += torch.sum(violations).item()
        self.total_pixels += len(preds)

    def get_results(self):
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - intersection
        iou_per_class = intersection / (union + 1e-6)
        miou = np.nanmean(iou_per_class)
        
        return {
            "mIoU": miou,
            "Class_IoU": iou_per_class,
            "Physics_Violations_Count": self.physics_violations
        }

def evaluate_model():
    config = SpectraConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📉 Starting Real Evaluation on {device}...")

    # 1. Load the model architecture
    encoder = load_prithvi_model()
    encoder = inject_spectra_lora(encoder)
    encoder = patch_model_for_context(encoder)
    decoder = SimpleDecoder(num_classes=3)
    model = nn.Sequential(encoder, decoder).to(device)
    
    # 2. Load the trained weights
    try:
        model.load_state_dict(torch.load("spectra_lora_weights.pth"))
        print("✅ Loaded trained SpectraLoRA weights!")
    except:
        print("⚠️ Warning: Could not find 'spectra_lora_weights.pth'. Evaluating untrained model.")
        
    model.eval() 
    
    # 3. Load Real Dataset
    dataset = RealSatelliteDataset("dataset_224x224")
    dataloader = DataLoader(dataset, batch_size=2)
    evaluator = PhysicsEvaluator(num_classes=3)
    
    print("🚀 Running full-dataset inference...")
    with torch.no_grad():
        for images, masks in tqdm(dataloader):
            images = images.to(device)
            
            # Get real NDVI for the violation checker
            red = images[:, config.BAND_MAP['RED'], :, :]
            nir = images[:, config.BAND_MAP['NIR'], :, :]
            ndvi_map = (nir - red) / (nir + red + 1e-8)
            
            # Physics Router
            SpectraContext.current_z = get_spectral_fingerprint(images, config.BAND_MAP)
            
            # Predict
            logits = model(images)
            preds = torch.argmax(logits, dim=1) 
            
            evaluator.update(preds, masks, ndvi_map)
            
    # 4. Report
    results = evaluator.get_results()
    print("\n" + "="*45)
    print("🔬 SPECTRALORA FINAL EVALUATION REPORT")
    print("="*45)
    print(f"✅ Mean IoU (Accuracy):      {results['mIoU']:.4f}")
    print(f"📊 Class-wise IoU:          {results['Class_IoU']}")
    print(f"⚠️ Physics Violations:      {results['Physics_Violations_Count']} pixels")
    print("-" * 45)

if __name__ == "__main__":
    evaluate_model()