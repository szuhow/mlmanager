#!/usr/bin/env python3
"""
Force refresh MLflow artifacts by creating a test run
"""

import mlflow
import tempfile
import os
import json

def create_test_artifacts():
    """Create a test run with artifacts to verify MLflow UI"""
    
    print("🧪 Creating test MLflow run with artifacts...")
    
    try:
        # Connect to MLflow
        mlflow.set_tracking_uri("http://localhost:5000")
        
        # Set experiment
        experiment_name = "coronary-experiments-dev"
        mlflow.set_experiment(experiment_name)
        
        # Start a new run
        with mlflow.start_run(run_name="artifact_test_run"):
            print(f"📝 Started test run: {mlflow.active_run().info.run_id}")
            
            # Create temporary artifacts
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create a simple text file
                test_file = os.path.join(temp_dir, "test_artifact.txt")
                with open(test_file, 'w') as f:
                    f.write("This is a test artifact to verify MLflow UI display\n")
                    f.write(f"Created at: {os.popen('date').read().strip()}\n")
                
                # Create a JSON config file
                config_file = os.path.join(temp_dir, "test_config.json")
                with open(config_file, 'w') as f:
                    json.dump({
                        "test": True,
                        "purpose": "MLflow UI artifact verification",
                        "artifacts_should_be_visible": True
                    }, f, indent=2)
                
                # Log artifacts
                mlflow.log_artifact(test_file, "test_artifacts")
                mlflow.log_artifact(config_file, "config")
                
                print("✅ Logged test artifacts:")
                print("   - test_artifacts/test_artifact.txt")
                print("   - config/test_config.json")
                
                # Log some metrics
                mlflow.log_metric("test_metric", 0.95)
                mlflow.log_param("test_param", "artifact_verification")
                
                print("✅ Logged test metrics and parameters")
                
                # Create a simple image artifact
                try:
                    import matplotlib.pyplot as plt
                    import numpy as np
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
                    x = np.linspace(0, 10, 100)
                    y = np.sin(x)
                    ax.plot(x, y)
                    ax.set_title("Test Plot - MLflow Artifact Verification")
                    ax.set_xlabel("X")
                    ax.set_ylabel("sin(X)")
                    
                    plot_file = os.path.join(temp_dir, "test_plot.png")
                    fig.savefig(plot_file, dpi=100, bbox_inches='tight')
                    plt.close()
                    
                    mlflow.log_artifact(plot_file, "visualizations")
                    print("✅ Logged test plot: visualizations/test_plot.png")
                    
                except ImportError:
                    print("ℹ️  Matplotlib not available, skipping plot creation")
                
                print(f"\n🎯 Test run completed successfully!")
                print(f"🌐 Check MLflow UI at: http://localhost:5000")
                print(f"📁 Look for experiment: {experiment_name}")
                print(f"🏃 Run ID: {mlflow.active_run().info.run_id}")
                
                return True
                
    except Exception as e:
        print(f"❌ Error creating test artifacts: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    create_test_artifacts()
