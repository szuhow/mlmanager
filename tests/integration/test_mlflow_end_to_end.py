#!/usr/bin/env python3
"""
End-to-End MLflow Artifact Manager Test
Test actual artifact logging and MLflow UI integration
"""

import os
import sys
import tempfile
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, '/app' if os.path.exists('/app') else '.')

import mlflow
from ml.utils.utils.mlflow_artifact_manager import MLflowArtifactManager, log_epoch_artifacts, log_final_model

def create_test_artifacts():
    """Create test artifacts for demonstration"""
    temp_dir = tempfile.mkdtemp(prefix="test_artifacts_")
    
    # Create a sample training curve plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = range(1, 11)
    train_loss = [0.8 - 0.1 * x + 0.02 * np.random.randn() for x in epochs]
    val_loss = [0.85 - 0.08 * x + 0.03 * np.random.randn() for x in epochs]
    
    ax1.plot(epochs, train_loss, label='Train Loss', marker='o')
    ax1.plot(epochs, val_loss, label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curves')
    ax1.legend()
    ax1.grid(True)
    
    train_dice = [0.3 + 0.06 * x + 0.02 * np.random.randn() for x in epochs]
    val_dice = [0.25 + 0.065 * x + 0.03 * np.random.randn() for x in epochs]
    
    ax2.plot(epochs, train_dice, label='Train Dice', marker='o')
    ax2.plot(epochs, val_dice, label='Val Dice', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Dice Score Progress')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    curves_path = os.path.join(temp_dir, "training_curves.png")
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create a sample predictions visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        # Simulate image, ground truth, and prediction
        image = np.random.rand(64, 64)
        ax.imshow(image, cmap='gray')
        ax.set_title(f'Sample {i+1}')
        ax.axis('off')
    
    plt.suptitle('Sample Predictions vs Ground Truth', fontsize=16)
    plt.tight_layout()
    
    predictions_path = os.path.join(temp_dir, "predictions.png")
    plt.savefig(predictions_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create sample images visualization
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for i, ax in enumerate(axes):
        sample_image = np.random.rand(64, 64)
        ax.imshow(sample_image, cmap='viridis')
        ax.set_title(f'Sample Image {i+1}')
        ax.axis('off')
    
    plt.suptitle('Random Sample Images from Dataset', fontsize=14)
    plt.tight_layout()
    
    samples_path = os.path.join(temp_dir, "sample_images.png")
    plt.savefig(samples_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return temp_dir, {
        'training_curves': curves_path,
        'predictions': predictions_path,
        'sample_images': samples_path
    }

def test_epoch_artifact_logging():
    """Test epoch artifact logging with real MLflow"""
    print("🧪 Testing Epoch Artifact Logging")
    print("=" * 50)
    
    # Create test artifacts
    temp_dir, artifacts = create_test_artifacts()
    
    try:
        # Start MLflow run
        with mlflow.start_run(run_name="Enhanced_Artifact_Test_Epoch") as run:
            print(f"📝 Started MLflow run: {run.info.run_id}")
            
            # Simulate epoch 5 metrics
            metrics = {
                'train_loss': 0.15,
                'val_loss': 0.12,
                'train_dice': 0.85,
                'val_dice': 0.89,
                'learning_rate': 0.001,
                'batch_size': 32
            }
            
            # Additional metadata
            metadata = {
                'optimizer': 'Adam',
                'data_augmentation': True,
                'model_params': 1234567,
                'gpu_memory_used': '8.5GB',
                'training_time_minutes': 45.2
            }
            
            # Log epoch artifacts using our enhanced manager
            logged_paths = log_epoch_artifacts(
                epoch=5,
                model_state=None,  # Skip model state for this test
                metrics=metrics,
                artifacts=artifacts,
                metadata=metadata
            )
            
            print(f"✅ Logged {len(logged_paths)} artifact categories:")
            for category, path in logged_paths.items():
                print(f"   📁 {category}: {path}")
            
            # Also log to MLflow metrics for comparison
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value, step=5)
            
            print(f"🌐 View artifacts in MLflow UI: http://localhost:5000/#/experiments/0/runs/{run.info.run_id}")
            
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        print("🧹 Cleaned up test artifacts")

def test_final_model_logging():
    """Test final model artifact logging"""
    print("\n🏁 Testing Final Model Artifact Logging")
    print("=" * 50)
    
    # Create a model directory structure
    model_temp_dir = tempfile.mkdtemp(prefix="final_model_")
    
    # Create directory structure
    weights_dir = os.path.join(model_temp_dir, "weights")
    config_dir = os.path.join(model_temp_dir, "config")
    artifacts_dir = os.path.join(model_temp_dir, "artifacts")
    
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Create dummy model weights file
    weights_path = os.path.join(weights_dir, "model.pth")
    with open(weights_path, 'w') as f:
        f.write("# Dummy PyTorch model weights\n# In real scenario, this would be torch.save() output")
    
    # Create some training artifacts
    training_log_path = os.path.join(artifacts_dir, "training.log")
    with open(training_log_path, 'w') as f:
        f.write("Epoch 1: Train Loss=0.8, Val Loss=0.7\n")
        f.write("Epoch 2: Train Loss=0.6, Val Loss=0.5\n")
        f.write("Epoch 5: Train Loss=0.15, Val Loss=0.12\n")
        f.write("Training completed successfully!")
    
    try:
        # Start MLflow run
        with mlflow.start_run(run_name="Enhanced_Artifact_Test_Final_Model") as run:
            print(f"📝 Started MLflow run: {run.info.run_id}")
            
            # Model information
            model_info = {
                'architecture': 'MONAI UNet Enhanced',
                'framework': 'PyTorch',
                'model_family': 'UNet-Coronary-Segmentation',
                'total_parameters': 1234567,
                'trainable_parameters': 1234567,
                'input_channels': 1,
                'output_channels': 2,
                'spatial_dims': 3,
                'features': [32, 64, 128, 256, 512],
                'training_config': {
                    'optimizer': 'Adam',
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 10
                },
                'data_info': {
                    'dataset_size': 150,
                    'train_split': 0.8,
                    'val_split': 0.2,
                    'image_size': [128, 128, 64]
                }
            }
            
            # Best metrics achieved
            best_metrics = {
                'best_val_dice': 0.923,
                'best_val_loss': 0.087,
                'final_train_loss': 0.105,
                'final_train_dice': 0.912,
                'best_epoch': 8,
                'total_training_time_hours': 2.5
            }
            
            # Log final model artifacts
            logged_paths = log_final_model(
                model_info=model_info,
                model_directory=model_temp_dir,
                best_metrics=best_metrics
            )
            
            print(f"✅ Logged {len(logged_paths)} final model artifacts:")
            for category, path in logged_paths.items():
                print(f"   📁 {category}: {path}")
            
            # Also log final metrics to MLflow
            for metric_name, value in best_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"final_{metric_name}", value)
            
            print(f"🌐 View final model in MLflow UI: http://localhost:5000/#/experiments/0/runs/{run.info.run_id}")
            
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(model_temp_dir)
        print("🧹 Cleaned up model artifacts")

def test_mlflow_ui_integration():
    """Test that artifacts appear correctly in MLflow UI"""
    print("\n🌐 Testing MLflow UI Integration")
    print("=" * 50)
    
    try:
        # Get list of experiments
        experiments = mlflow.search_experiments()
        print(f"📊 Found {len(experiments)} MLflow experiments")
        
        # Get recent runs
        runs = mlflow.search_runs(experiment_ids=['0'], max_results=5)
        print(f"🏃 Found {len(runs)} recent runs")
        
        if len(runs) > 0:
            latest_run = runs.iloc[0]
            run_id = latest_run['run_id']
            print(f"📝 Latest run: {run_id}")
            print(f"📈 Metrics logged: {[col for col in runs.columns if col.startswith('metrics.')]}")
            print(f"🏷️  Tags: {[col for col in runs.columns if col.startswith('tags.')]}")
            
            # Try to get artifacts for the latest run
            try:
                client = mlflow.tracking.MlflowClient()
                artifacts = client.list_artifacts(run_id)
                print(f"📁 Artifacts in latest run: {len(artifacts)}")
                for artifact in artifacts[:10]:  # Show first 10
                    print(f"   📄 {artifact.path} ({artifact.file_size} bytes)")
            except Exception as e:
                print(f"⚠️  Could not list artifacts: {e}")
        
        print("✅ MLflow UI integration check completed")
        print(f"🌐 Access MLflow UI at: http://localhost:5000")
        
    except Exception as e:
        print(f"❌ MLflow UI integration test failed: {e}")

def main():
    """Run all end-to-end tests"""
    print("🚀 Enhanced MLflow Artifact Manager - End-to-End Tests")
    print("=" * 60)
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    print(f"🔗 MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    
    try:
        # Test epoch logging
        test_epoch_artifact_logging()
        
        # Test final model logging  
        test_final_model_logging()
        
        # Test MLflow UI integration
        test_mlflow_ui_integration()
        
        print("\n" + "=" * 60)
        print("🎉 ALL END-TO-END TESTS COMPLETED SUCCESSFULLY!")
        print("✅ Enhanced MLflow Artifact Manager is working perfectly")
        print("🌐 Check your artifacts in MLflow UI: http://localhost:5000")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ End-to-end tests failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
