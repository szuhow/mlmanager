#!/usr/bin/env python3
"""
Test MONAI transforms fix - sprawdza czy usunięcie ScaleIntensityd dla labels działa poprawnie
"""

import torch
import numpy as np
from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd, 
    ScaleIntensityd, 
    ToTensord
)

def test_monai_transforms_fix():
    """Test czy MONAI transforms nie normalizują masek po usunięciu ScaleIntensityd"""
    
    print("============================================================")
    print("🧪 MONAI TRANSFORMS FIX TEST")
    print("============================================================")
    
    # Stwórz syntetyczne dane w formacie który oczekuje MONAI
    print("📊 Creating synthetic data...")
    
    # Symuluj dane wejściowe jak w prawdziwym przypadku
    sample_data = {
        "image": np.random.rand(512, 512, 3) * 255,  # RGB obraz
        "label": np.random.choice([0, 255], size=(512, 512))  # Binarna maska
    }
    
    print(f"Original label range: [{sample_data['label'].min()}, {sample_data['label'].max()}]")
    print(f"Original label dtype: {sample_data['label'].dtype}")
    
    # Test starych transforms (z ScaleIntensityd dla labels) - symulacja
    print("\n🔍 Testing OLD transforms (with ScaleIntensityd for labels)...")
    old_transforms = Compose([
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image", "label"]),  # To było problematyczne dla labels
        ToTensord(keys=["image", "label"])
    ])
    
    try:
        old_result = old_transforms(sample_data.copy())
        old_label_min = float(old_result["label"].min())
        old_label_max = float(old_result["label"].max())
        print(f"❌ OLD: Label range after transforms: [{old_label_min:.3f}, {old_label_max:.3f}]")
        print(f"❌ OLD: Label dtype: {old_result['label'].dtype}")
        
        if old_label_max <= 1.0:
            print("❌ OLD: Labels were normalized to [0,1] - THIS WAS THE PROBLEM!")
    except Exception as e:
        print(f"❌ OLD transforms failed: {e}")
    
    # Test nowych transforms (bez ScaleIntensityd dla labels) - nasze rozwiązanie
    print("\n🔍 Testing NEW transforms (without ScaleIntensityd for labels)...")
    new_transforms = Compose([
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),  # Tylko dla obrazów, nie dla labels
        ToTensord(keys=["image", "label"])
    ])
    
    try:
        new_result = new_transforms(sample_data.copy())
        new_label_min = float(new_result["label"].min())
        new_label_max = float(new_result["label"].max())
        print(f"✅ NEW: Label range after transforms: [{new_label_min:.3f}, {new_label_max:.3f}]")
        print(f"✅ NEW: Label dtype: {new_result['label'].dtype}")
        
        if new_label_max > 1.0:
            print("✅ NEW: Labels kept original values [0, 255] - PROBLEM FIXED!")
        else:
            print("⚠️ NEW: Labels still normalized - check implementation")
            
        # Test kompatybilności z funkcjami straty
        print("\n🧪 Testing loss function compatibility...")
        from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
        
        # Symuluj output z modelu (logits)
        batch_size = 2
        height, width = 256, 256
        num_classes = 2
        
        # Model outputs (przed sigmoid/softmax)
        model_output = torch.randn(batch_size, num_classes, height, width)
        
        # Ground truth labels z naszych transforms
        gt_labels = new_result["label"].unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        # Test BCE Loss (dla binary segmentation)
        if num_classes == 1 or new_label_max <= 1.0:
            # Jeśli labels są w [0,1], można użyć bezpośrednio
            bce_loss = BCEWithLogitsLoss()
            # Dla BCE, model output powinien mieć 1 kanał
            single_channel_output = model_output[:, 0:1, :, :]
            normalized_labels = gt_labels.float() / 255.0 if new_label_max > 1.0 else gt_labels.float()
            try:
                loss_value = bce_loss(single_channel_output, normalized_labels)
                print(f"✅ BCE Loss computed successfully: {loss_value:.4f}")
            except Exception as e:
                print(f"❌ BCE Loss failed: {e}")
        
        # Test CrossEntropy Loss
        try:
            ce_loss = CrossEntropyLoss()
            # Dla CE, labels powinny być long tensor z class indices
            ce_labels = (gt_labels.squeeze(1) / 255.0).long() if new_label_max > 1.0 else gt_labels.squeeze(1).long()
            loss_value = ce_loss(model_output, ce_labels)
            print(f"✅ CrossEntropy Loss computed successfully: {loss_value:.4f}")
        except Exception as e:
            print(f"❌ CrossEntropy Loss failed: {e}")
            
    except Exception as e:
        print(f"❌ NEW transforms failed: {e}")
    
    print("\n============================================================")
    print("✅ MONAI transforms fix test completed!")
    print("============================================================")

if __name__ == "__main__":
    # Sprawdź czy jesteśmy w odpowiednim środowisku
    try:
        import sys
        sys.path.append('/app')  # Dodaj ścieżkę aplikacji w kontenerze
        test_monai_transforms_fix()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
