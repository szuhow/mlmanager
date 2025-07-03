#!/usr/bin/env python3
"""
Test script to verify MLflow artifact logging and UI display
"""
import mlflow
import mlflow.pytorch
import tempfile
import os
import json
from datetime import datetime

def test_mlflow_artifacts():
    """Test MLflow artifact logging and verify they appear in UI"""
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Create experiment if it doesn't exist
    experiment_name = "test_artifacts_experiment"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
    except Exception as e:
        print(f"Error with experiment: {e}")
        experiment_id = "0"  # Default experiment
    
    print(f"Using experiment ID: {experiment_id}")
    
    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        run_id = run.info.run_id
        print(f"Started run: {run_id}")
        
        # Log some metrics
        mlflow.log_metric("test_metric", 0.95)
        mlflow.log_metric("accuracy", 0.87)
        
        # Log some parameters
        mlflow.log_param("test_param", "test_value")
        mlflow.log_param("model_type", "test_model")
        
        # Create and log artifacts
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test file
            test_file = os.path.join(temp_dir, "test_artifact.txt")
            with open(test_file, "w") as f:
                f.write("This is a test artifact file\n")
                f.write(f"Created at: {datetime.now()}\n")
            
            # Create a JSON config file
            config_file = os.path.join(temp_dir, "config.json")
            config_data = {
                "model_name": "test_model",
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001
            }
            with open(config_file, "w") as f:
                json.dump(config_data, f, indent=2)
            
            # Log individual files
            mlflow.log_artifact(test_file, "test_files")
            mlflow.log_artifact(config_file, "configs")
            
            # Log entire directory
            artifacts_dir = os.path.join(temp_dir, "model_artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)
            
            model_info_file = os.path.join(artifacts_dir, "model_info.txt")
            with open(model_info_file, "w") as f:
                f.write("Model Information\n")
                f.write("================\n")
                f.write("Architecture: ResNet50\n")
                f.write("Parameters: 25M\n")
            
            mlflow.log_artifacts(artifacts_dir, "model")
        
        print(f"Logged artifacts for run: {run_id}")
        
        # Verify artifacts using MLflow API
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        
        print("\nArtifacts logged (via MLflow API):")
        for artifact in artifacts:
            print(f"  - {artifact.path} (size: {artifact.file_size} bytes)")
            
            # List subdirectories if any
            if artifact.is_dir:
                sub_artifacts = client.list_artifacts(run_id, artifact.path)
                for sub_artifact in sub_artifacts:
                    print(f"    - {sub_artifact.path} (size: {sub_artifact.file_size} bytes)")
        
        # Check artifact store path
        artifact_uri = mlflow.get_artifact_uri()
        print(f"\nArtifact URI: {artifact_uri}")
        
        # Check if artifacts exist on disk
        if artifact_uri.startswith("file://"):
            local_path = artifact_uri[7:]  # Remove file:// prefix
            if os.path.exists(local_path):
                print(f"Artifacts exist on disk at: {local_path}")
                print("Files on disk:")
                for root, dirs, files in os.walk(local_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, local_path)
                        print(f"  - {rel_path}")
            else:
                print(f"Artifact path does not exist: {local_path}")
        
        print(f"\nMLflow UI should show artifacts at: http://localhost:5000/#/experiments/{experiment_id}/runs/{run_id}")
        
        return run_id, experiment_id

if __name__ == "__main__":
    try:
        run_id, experiment_id = test_mlflow_artifacts()
        print(f"\n✅ Test completed successfully!")
        print(f"Check MLflow UI at: http://localhost:5000/#/experiments/{experiment_id}/runs/{run_id}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
