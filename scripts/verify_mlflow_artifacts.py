#!/usr/bin/env python3
"""
MLflow Artifact Verification Script
Verifies that artifacts are properly logged and accessible
"""

import mlflow
import os
import sys

def verify_mlflow_artifacts():
    """Verify MLflow artifacts are properly logged"""
    
    print("🔍 MLflow Artifact Verification")
    print("=" * 50)
    
    try:
        # Set tracking URI
        mlflow.set_tracking_uri("http://localhost:5000")
        print(f"✅ Connected to MLflow: {mlflow.get_tracking_uri()}")
        
        # Get experiments
        experiments = mlflow.search_experiments()
        print(f"📁 Found {len(experiments)} experiments")
        
        # Find the coronary experiments
        coronary_exp = None
        for exp in experiments:
            print(f"   - {exp.name} (ID: {exp.experiment_id})")
            if 'coronary' in exp.name.lower():
                coronary_exp = exp
        
        if not coronary_exp:
            print("❌ No coronary experiment found")
            return False
        
        print(f"\n🎯 Using experiment: {coronary_exp.name}")
        print(f"📂 Artifact location: {coronary_exp.artifact_location}")
        
        # Get recent runs
        runs = mlflow.search_runs(
            experiment_ids=[coronary_exp.experiment_id], 
            max_results=5,
            order_by=["start_time DESC"]
        )
        
        print(f"\n📊 Found {len(runs)} runs:")
        
        client = mlflow.tracking.MlflowClient()
        
        for i, (_, run) in enumerate(runs.iterrows()):
            print(f"\n🏃 Run {i+1}: {run.run_id[:8]}...")
            print(f"   Status: {run.status}")
            print(f"   Start: {run.start_time}")
            
            # List artifacts
            try:
                artifacts = client.list_artifacts(run.run_id)
                print(f"   📦 Artifacts: {len(artifacts)} items")
                
                # Count total files recursively
                def count_artifacts(path=""):
                    count = 0
                    items = client.list_artifacts(run.run_id, path)
                    for item in items:
                        if item.is_dir:
                            count += count_artifacts(item.path)
                        else:
                            count += 1
                    return count
                
                total_files = count_artifacts()
                print(f"   📄 Total files: {total_files}")
                
                # Show top-level artifacts
                for artifact in artifacts[:5]:
                    if artifact.is_dir:
                        print(f"      📁 {artifact.path}/")
                    else:
                        size_mb = artifact.file_size / (1024*1024) if artifact.file_size else 0
                        print(f"      📄 {artifact.path} ({size_mb:.1f} MB)")
                
                if len(artifacts) > 5:
                    print(f"      ... and {len(artifacts) - 5} more items")
                    
            except Exception as e:
                print(f"   ❌ Error listing artifacts: {e}")
        
        # Test artifact download
        if len(runs) > 0:
            latest_run = runs.iloc[0]
            run_id = latest_run.run_id
            
            print(f"\n🔍 Testing artifact access for run {run_id[:8]}...")
            
            try:
                # Try to download a small artifact
                artifacts = client.list_artifacts(run_id)
                small_artifacts = []
                
                def find_small_artifacts(path=""):
                    items = client.list_artifacts(run_id, path)
                    for item in items:
                        if item.is_dir:
                            find_small_artifacts(item.path)
                        elif item.file_size and item.file_size < 10000:  # < 10KB
                            small_artifacts.append(item.path)
                
                find_small_artifacts()
                
                if small_artifacts:
                    test_artifact = small_artifacts[0]
                    print(f"   📥 Testing download of: {test_artifact}")
                    
                    # Download to temp location
                    import tempfile
                    with tempfile.TemporaryDirectory() as temp_dir:
                        downloaded_path = client.download_artifacts(run_id, test_artifact, temp_dir)
                        if os.path.exists(downloaded_path):
                            print(f"   ✅ Successfully downloaded artifact")
                            print(f"   📁 Downloaded to: {downloaded_path}")
                        else:
                            print(f"   ❌ Download failed - file not found")
                else:
                    print(f"   ℹ️  No small artifacts found for testing")
                    
            except Exception as e:
                print(f"   ❌ Error testing artifact download: {e}")
        
        print(f"\n✅ Verification complete!")
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    verify_mlflow_artifacts()
