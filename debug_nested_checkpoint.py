#!/usr/bin/env python3
"""
Debug nested checkpoint format to understand structure
"""
import torch
import os

checkpoint_path = "data/mlflow/307f67e76c6140c79d282e409c1f36c0/artifacts/final_model/weights/model.pth"

if os.path.exists(checkpoint_path):
    print(f"🔍 Analyzing nested checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Type: {type(checkpoint)}")
    print(f"Keys: {list(checkpoint.keys())}")
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"\nmodel_state_dict:")
        print(f"  Type: {type(state_dict)}")
        print(f"  Num keys: {len(state_dict)}")
        print(f"  First 10 keys: {list(state_dict.keys())[:10]}")
        
    if 'model_metadata' in checkpoint:
        metadata = checkpoint['model_metadata']
        print(f"\nmodel_metadata:")
        print(f"  Type: {type(metadata)}")
        if isinstance(metadata, dict):
            print(f"  Keys: {list(metadata.keys())}")
            for key, value in metadata.items():
                print(f"    {key}: {value}")
                
    if 'training_args' in checkpoint:
        training_args = checkpoint['training_args']
        print(f"\ntraining_args:")
        print(f"  Type: {type(training_args)}")
        if isinstance(training_args, dict):
            print(f"  Keys: {list(training_args.keys())}")
            for key, value in training_args.items():
                print(f"    {key}: {value}")
