#!/usr/bin/env python3
"""
MLflow Status Debugging Script
Diagnoses MLflow run status issues and checks for active training processes
"""

import os
import sys
import subprocess
import json
import glob
from datetime import datetime, timedelta
from pathlib import Path

def check_running_processes():
    """Check for any running training processes"""
    print("=== CHECKING RUNNING PROCESSES ===")
    
    try:
        # Check for Python processes related to training
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        processes = result.stdout.split('\n')
        
        training_processes = []
        for line in processes:
            if any(keyword in line.lower() for keyword in ['train.py', 'python', 'mlflow', 'torch']):
                if 'train' in line or 'mlflow' in line:
                    training_processes.append(line.strip())
        
        if training_processes:
            print("Found potential training processes:")
            for proc in training_processes:
                print(f"  {proc}")
        else:
            print("No training processes found")
            
    except Exception as e:
        print(f"Error checking processes: {e}")
    
    print()

def check_mlflow_experiments():
    """Check MLflow experiments and run status"""
    print("=== CHECKING MLFLOW EXPERIMENTS ===")
    
    try:
        import mlflow
        mlflow.set_tracking_uri('file:./mlruns')
        
        experiments = mlflow.search_experiments()
        for exp in experiments:
            print(f"Experiment: {exp.name} (ID: {exp.experiment_id})")
            
            # Get all runs for this experiment
            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id], 
                order_by=['start_time DESC'],
                max_results=20
            )
            
            if not runs.empty:
                print(f"Found {len(runs)} runs:")
                for idx, run in runs.iterrows():
                    run_id = run['run_id']
                    status = run['status']
                    start_time = run['start_time']
                    end_time = run['end_time']
                    
                    print(f"  Run: {run_id[:8]}...")
                    print(f"    Status: {status}")
                    print(f"    Started: {start_time}")
                    print(f"    Ended: {end_time}")
                    
                    # Check if this might be a stuck run
                    if status == 'RUNNING' and end_time is None:
                        if start_time:
                            age = datetime.now() - start_time.replace(tzinfo=None)
                            print(f"    ⚠️  Run has been 'RUNNING' for {age}")
                    
                    print()
            else:
                print("  No runs found")
                
    except Exception as e:
        print(f"Error checking MLflow: {e}")
    
    print()

def check_mlflow_directory_structure():
    """Check MLflow directory structure for clues"""
    print("=== CHECKING MLFLOW DIRECTORY STRUCTURE ===")
    
    mlruns_path = Path('./mlruns')
    if not mlruns_path.exists():
        print("No mlruns directory found")
        return
    
    # Get all experiment directories
    exp_dirs = [d for d in mlruns_path.iterdir() if d.is_dir() and d.name != '.trash']
    exp_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print(f"Found {len(exp_dirs)} experiment directories:")
    
    for exp_dir in exp_dirs[:5]:  # Check last 5
        print(f"\nExperiment: {exp_dir.name}")
        
        # Check for meta.yaml
        meta_file = exp_dir / 'meta.yaml'
        if meta_file.exists():
            try:
                with open(meta_file, 'r') as f:
                    content = f.read()
                    print(f"  Meta: {content[:100]}...")
            except:
                pass
        
        # Check artifacts
        artifacts_dir = exp_dir / 'artifacts'
        if artifacts_dir.exists():
            artifact_count = len(list(artifacts_dir.rglob('*')))
            print(f"  Artifacts: {artifact_count} files/dirs")
            
            # Check for recent files
            recent_files = []
            for file_path in artifacts_dir.rglob('*'):
                if file_path.is_file():
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if datetime.now() - mtime < timedelta(hours=1):
                        recent_files.append((file_path, mtime))
            
            if recent_files:
                print(f"  Recent files (last hour): {len(recent_files)}")
                for file_path, mtime in recent_files[:3]:
                    rel_path = file_path.relative_to(artifacts_dir)
                    print(f"    {rel_path} - {mtime}")
        
        # Check modification time
        mtime = datetime.fromtimestamp(exp_dir.stat().st_mtime)
        print(f"  Last modified: {mtime}")
    
    print()

def check_lock_files():
    """Check for any lock files that might indicate running processes"""
    print("=== CHECKING FOR LOCK FILES ===")
    
    lock_patterns = [
        '**/*.lock',
        '**/.lock',
        '**/mlflow.lock',
        '**/training.lock',
        '**/*.pid'
    ]
    
    found_locks = []
    for pattern in lock_patterns:
        locks = list(Path('.').glob(pattern))
        found_locks.extend(locks)
    
    if found_locks:
        print("Found lock files:")
        for lock in found_locks:
            mtime = datetime.fromtimestamp(lock.stat().st_mtime)
            print(f"  {lock} - {mtime}")
    else:
        print("No lock files found")
    
    print()

def check_log_files():
    """Check recent log files for training activity"""
    print("=== CHECKING LOG FILES ===")
    
    log_patterns = [
        '**/*.log',
        '**/logs/**/*',
        '**/output*.txt'
    ]
    
    recent_logs = []
    for pattern in log_patterns:
        logs = list(Path('.').glob(pattern))
        for log in logs:
            if log.is_file():
                mtime = datetime.fromtimestamp(log.stat().st_mtime)
                if datetime.now() - mtime < timedelta(hours=2):
                    recent_logs.append((log, mtime))
    
    recent_logs.sort(key=lambda x: x[1], reverse=True)
    
    if recent_logs:
        print("Recent log files (last 2 hours):")
        for log, mtime in recent_logs[:5]:
            print(f"  {log} - {mtime}")
            
            # Check last few lines
            try:
                with open(log, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_lines = lines[-3:]
                        print("    Last lines:")
                        for line in last_lines:
                            print(f"      {line.strip()}")
            except:
                pass
            print()
    else:
        print("No recent log files found")
    
    print()

def suggest_solutions():
    """Suggest solutions based on findings"""
    print("=== SUGGESTED SOLUTIONS ===")
    
    solutions = [
        "1. Check if training is actually running:",
        "   - Use 'ps aux | grep train' to check for training processes",
        "   - Look for recent artifact files in mlruns/",
        "",
        "2. If training appears stuck:",
        "   - Kill any zombie training processes",
        "   - Clean up any lock files",
        "   - Start a fresh training run",
        "",
        "3. For MLflow UI issues:",
        "   - Restart MLflow server: mlflow ui --host 0.0.0.0 --port 5000",
        "   - Check MLflow tracking URI is correct",
        "   - Verify run status in database",
        "",
        "4. For missing images in UI:",
        "   - Check artifact paths are correct",
        "   - Verify image files exist in mlruns/",
        "   - Ensure proper MIME types for images",
        "",
        "5. To fix run status:",
        "   - Use mlflow.end_run() to properly close runs",
        "   - Check for unclosed context managers",
        "   - Verify exception handling in training code"
    ]
    
    for solution in solutions:
        print(solution)

def main():
    """Main diagnostic function"""
    print("MLflow Status Diagnostic Tool")
    print("=" * 40)
    print()
    
    # Use current directory (works both locally and in container)
    if not os.path.exists('./mlruns'):
        print("Warning: No mlruns directory found in current directory")
        print(f"Current directory: {os.getcwd()}")
        print("Contents:", os.listdir('.'))
        return
    
    check_running_processes()
    check_mlflow_experiments()
    check_mlflow_directory_structure()
    check_lock_files()
    check_log_files()
    suggest_solutions()
    
    print("\nDiagnostic complete!")

if __name__ == "__main__":
    main()
