#!/usr/bin/env python3
"""
Test ARCADE Dataset Integration
Test the integration of torch-arcade with MLManager
"""

import os
import sys
import django
import logging
from pathlib import Path

# Setup Django
sys.path.append('/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')
django.setup()

from ml.datasets.arcade_loader import (
    ARCADEDatasetAdapter,
    ARCADEDataLoader,
    is_arcade_dataset,
    ARCADE_AVAILABLE
)

from ml.datasets.torch_arcade_loader import (
    create_arcade_dataloader,
    get_arcade_dataset_info
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_arcade_availability():
    """Test if ARCADE components are available"""
    print("🧪 Testing ARCADE Availability")
    print("=" * 50)
    
    print(f"ARCADE_AVAILABLE: {ARCADE_AVAILABLE}")
    
    if not ARCADE_AVAILABLE:
        print("❌ torch-arcade not available. Install it manually:")
        print("   pip install git+https://github.com/szuhow/torch-arcade")
        return False
    
    print("✅ ARCADE components available")
    return True

def test_dataset_detection():
    """Test dataset type detection"""
    print("\n🧪 Testing Dataset Detection")
    print("=" * 50)
    
    test_paths = [
        "/app/data/datasets",
        "/app/data/datasets/coronary",
        "/app/shared/datasets/data"
    ]
    
    for path in test_paths:
        if os.path.exists(path):
            print(f"📁 Testing path: {path}")
            is_arcade = is_arcade_dataset(path)
            print(f"   Is ARCADE dataset: {is_arcade}")
            
            try:
                info = get_arcade_dataset_info(path)
                print(f"   Dataset info: {info}")
            except Exception as e:
                print(f"   Error getting info: {e}")
        else:
            print(f"📁 Path does not exist: {path}")

def test_adapter_creation():
    """Test creating ARCADE dataset adapters"""
    print("\n🧪 Testing Dataset Adapter Creation")
    print("=" * 50)
    
    data_path = "/app/data/datasets"
    
    if not os.path.exists(data_path):
        print(f"❌ Test data path does not exist: {data_path}")
        return False
    
    tasks = ['binary_segmentation', 'semantic_segmentation']
    
    for task in tasks:
        try:
            print(f"📊 Testing task: {task}")
            
            adapter = ARCADEDatasetAdapter(
                root=data_path,
                task=task,
                image_set='train',
                resolution=256,
                use_augmentation=False
            )
            
            print(f"   ✅ Created adapter with {len(adapter)} samples")
            
            # Test getting a sample
            if len(adapter) > 0:
                img, mask = adapter[0]
                print(f"   📷 Sample shapes - Image: {img.shape}, Mask: {mask.shape}")
            
        except Exception as e:
            print(f"   ❌ Failed to create adapter for {task}: {e}")

def test_dataloader_creation():
    """Test creating ARCADE data loaders"""
    print("\n🧪 Testing DataLoader Creation")
    print("=" * 50)
    
    data_path = "/app/data/datasets"
    
    if not os.path.exists(data_path):
        print(f"❌ Test data path does not exist: {data_path}")
        return False
    
    try:
        print("🔄 Creating data loaders...")
        
        train_loader, val_loader = ARCADEDataLoader.create_loaders(
            data_path=data_path,
            task='binary_segmentation',
            batch_size=2,
            validation_split=0.2,
            resolution=256,
            num_workers=1,
            use_augmentation=True
        )
        
        print(f"✅ Created data loaders")
        print(f"   Train: {len(train_loader)} batches")
        print(f"   Val: {len(val_loader)} batches")
        
        # Test getting a batch
        try:
            train_batch = next(iter(train_loader))
            img_batch, mask_batch = train_batch
            print(f"   📊 Batch shapes - Images: {img_batch.shape}, Masks: {mask_batch.shape}")
        except Exception as e:
            print(f"   ⚠️  Could not get batch: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create data loaders: {e}")
        return False

def test_torch_arcade_integration():
    """Test torch-arcade integration functions"""
    print("\n🧪 Testing Torch-ARCADE Integration")
    print("=" * 50)
    
    data_path = "/app/data/datasets"
    
    if not os.path.exists(data_path):
        print(f"❌ Test data path does not exist: {data_path}")
        return False
    
    try:
        print("🔄 Testing create_arcade_dataloader function...")
        
        train_loader = create_arcade_dataloader(
            root=data_path,
            task='binary_segmentation',
            batch_size=2,
            num_workers=1,
            image_set='train'
        )
        
        val_loader = create_arcade_dataloader(
            root=data_path,
            task='binary_segmentation',
            batch_size=2,
            num_workers=1,
            image_set='val'
        )
        
        print(f"✅ Created loaders via torch_arcade_loader")
        print(f"   Train: {len(train_loader)} batches")
        print(f"   Val: {len(val_loader)} batches")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed torch-arcade integration test: {e}")
        return False

def test_available_tasks():
    """Test getting available tasks"""
    print("\n🧪 Testing Available Tasks")
    print("=" * 50)
    
    try:
        tasks = ARCADEDataLoader.get_available_tasks()
        print(f"✅ Available tasks: {tasks}")
        return True
    except Exception as e:
        print(f"❌ Failed to get available tasks: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 ARCADE DATASET INTEGRATION TEST")
    print("=" * 60)
    
    tests = [
        ("ARCADE Availability", test_arcade_availability),
        ("Dataset Detection", test_dataset_detection),
        ("Available Tasks", test_available_tasks),
        ("Adapter Creation", test_adapter_creation),
        ("DataLoader Creation", test_dataloader_creation),
        ("Torch-ARCADE Integration", test_torch_arcade_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result if result is not None else True))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("🏁 TEST RESULTS")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nSummary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! ARCADE integration is working.")
    else:
        print("🔧 Some tests failed. Check the output above for details.")
        print("\n💡 Common issues:")
        print("   - torch-arcade not installed: pip install git+https://github.com/szuhow/torch-arcade")
        print("   - pycocotools not installed: pip install pycocotools")
        print("   - Dataset path issues: ensure /app/data/datasets exists with data")

if __name__ == "__main__":
    main()
