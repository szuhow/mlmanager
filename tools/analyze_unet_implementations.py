#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '/home/rafal/Dokumenty/ivessystem/coronary/ives/coronary-experiments')

print("=== UNet Implementation Analysis ===")
print()

# Check what UNet implementations are available
print("1. MONAI UNet (from monai.networks.nets):")
try:
    from monai.networks.nets import UNet as MonaiUNet
    print("   ✓ Available - This is the CURRENT DEFAULT for 'unet' model type")
    print("   ✓ Professional medical imaging framework")
    print("   ✓ Used when model_type='unet' or 'monai_unet'")
except ImportError as e:
    print(f"   ✗ Not available: {e}")

print()

print("2. Local UNet (from shared/unet/):")
try:
    from shared.unet.unet_model import UNet as LocalUNet
    print("   ✓ Available - Custom PyTorch implementation")
    print("   ✓ Would be used via architecture registry system")
    print("   ✓ Traditional U-Net implementation")
except ImportError as e:
    print(f"   ✗ Not available: {e}")

print()

print("3. New Residual UNet (from shared/resunet/):")
try:
    from shared.resunet.resunet_model import ResUNet
    print("   ✓ Available - NEW implementation with residual connections")
    print("   ✓ Improved gradient flow and feature learning")
    print("   ✓ Would be used with model_type='resunet'")
except ImportError as e:
    print(f"   ✗ Not available: {e}")

print()

print("=== Current Model Selection Logic ===")
print("Based on shared/train.py analysis:")
print()
print("• If model_type in ['unet', 'monai_unet'] or 'unet' in model_type.lower():")
print("  → Uses MONAI UNet (monai.networks.nets.UNet)")
print()
print("• Otherwise:")
print("  → Uses architecture registry to load model from shared/ directories")
print()
print("=== Available Model Types ===")
print("Current system supports:")
print("• 'unet' or 'monai_unet' → MONAI UNet (DEFAULT)")
print("• 'resunet' → Residual UNet (NEW)")
print("• 'deep_resunet' → Deep Residual UNet (NEW)")
print("• 'attention_resunet' → Attention Residual UNet (NEW)")
print("• Other architectures discovered by registry system")
