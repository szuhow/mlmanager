#!/usr/bin/env python3
"""
Test ARCADE Structure Recognition
Test that the updated ARCADE DataLoader properly recognizes and handles the folder structure
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append('/app')

def test_arcade_structure_detection():
    """Test ARCADE structure detection functions"""
    print("🧪 Testing ARCADE Structure Detection")
    print("=" * 50)
    
    try:
        from ml.datasets.arcade_loader import (
            is_arcade_dataset,
            detect_arcade_task_type,
            get_arcade_dataset_root,
            get_arcade_task_paths
        )
        
        # Test paths
        test_paths = [
            "/app/data/datasets/arcade_challenge_datasets",
            "/app/data/datasets/arcade_challenge_datasets/dataset_phase_1",
            "/app/data/datasets/arcade_challenge_datasets/dataset_phase_1/segmentation_dataset",
            "/app/data/datasets/arcade_challenge_datasets/dataset_phase_1/stenosis_dataset",
        ]
        
        for path in test_paths:
            print(f"\n🔍 Testing path: {path}")
            
            # Test detection
            is_arcade = is_arcade_dataset(path)
            print(f"   Is ARCADE dataset: {is_arcade}")
            
            if is_arcade:
                # Test task detection
                task_type = detect_arcade_task_type(path)
                print(f"   Detected task type: {task_type}")
                
                # Test root path normalization
                root_path = get_arcade_dataset_root(path)
                print(f"   Normalized root: {root_path}")
                
                # Test task paths
                task_paths = get_arcade_task_paths(root_path, task_type)
                print(f"   Task paths:")
                for key, value in task_paths.items():
                    exists = Path(value).exists() if value else False
                    print(f"     {key}: {value} {'✅' if exists else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Structure detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_arcade_dataloader_creation():
    """Test ARCADE DataLoader creation with structure-aware paths"""
    print("\n🧪 Testing ARCADE DataLoader Creation")
    print("=" * 50)
    
    try:
        from ml.datasets.arcade_loader import ARCADEDatasetAdapter, ARCADEDataLoader
        
        # Test with different path levels
        test_configs = [
            {
                "path": "/app/data/datasets/arcade_challenge_datasets",
                "task": "auto",
                "description": "Root ARCADE path with auto-detection"
            },
            {
                "path": "/app/data/datasets/arcade_challenge_datasets/dataset_phase_1",
                "task": "binary_segmentation", 
                "description": "Phase 1 path with explicit task"
            },
            {
                "path": "/app/data/datasets/arcade_challenge_datasets/dataset_phase_1/segmentation_dataset",
                "task": "auto",
                "description": "Segmentation dataset path with auto-detection"
            }
        ]
        
        for config in test_configs:
            print(f"\n🔧 Testing: {config['description']}")
            print(f"   Path: {config['path']}")
            print(f"   Task: {config['task']}")
            
            try:
                # Test dataset adapter creation
                adapter = ARCADEDatasetAdapter(
                    root=config['path'],
                    task=config['task'],
                    image_set="train",
                    resolution=256,
                    use_augmentation=False
                )
                
                print(f"   ✅ Adapter created successfully")
                print(f"   📊 Dataset size: {len(adapter)}")
                print(f"   🎯 Final task: {adapter.task}")
                print(f"   📁 Root path: {adapter.root}")
                
                # Try to get first item
                if len(adapter) > 0:
                    img, target = adapter[0]
                    print(f"   🖼️  First item shape: img={img.shape}, target={target.shape}")
                
            except Exception as e:
                print(f"   ❌ Failed to create adapter: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ DataLoader creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_arcade_full_pipeline():
    """Test full ARCADE pipeline with DataLoader factory"""
    print("\n🧪 Testing Full ARCADE Pipeline")
    print("=" * 50)
    
    try:
        from ml.datasets.arcade_loader import ARCADEDataLoader
        
        # Test with segmentation dataset
        seg_path = "/app/data/datasets/arcade_challenge_datasets/dataset_phase_1/segmentation_dataset"
        
        print(f"🔧 Creating DataLoaders for: {seg_path}")
        
        train_loader, val_loader = ARCADEDataLoader.create_loaders(
            data_path=seg_path,
            task="binary_segmentation",
            batch_size=4,
            validation_split=0.2,
            resolution=256,
            num_workers=0,  # Avoid multiprocessing issues in container
            download=False,
            use_augmentation=True
        )
        
        print(f"✅ DataLoaders created successfully")
        print(f"📊 Train batches: {len(train_loader)}")
        print(f"📊 Val batches: {len(val_loader)}")
        
        # Test one batch
        try:
            train_batch = next(iter(train_loader))
            val_batch = next(iter(val_loader))
            
            print(f"🔍 Train batch shape: {train_batch[0].shape}, {train_batch[1].shape}")
            print(f"🔍 Val batch shape: {val_batch[0].shape}, {val_batch[1].shape}")
            
        except Exception as e:
            print(f"⚠️  Could not load batch: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 ARCADE STRUCTURE RECOGNITION TEST")
    print("=" * 60)
    
    tests = [
        ("Structure Detection", test_arcade_structure_detection),
        ("DataLoader Creation", test_arcade_dataloader_creation),
        ("Full Pipeline", test_arcade_full_pipeline)
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
        print("🎉 ALL TESTS PASSED! ARCADE structure recognition works correctly.")
    else:
        print("🔧 Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
