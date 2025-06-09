#!/usr/bin/env python3
"""
Script to organize loose artifacts and ensure proper directory structure
"""

import os
import shutil
import glob
import argparse
from pathlib import Path
from datetime import datetime

def organize_artifacts(project_dir=None, use_model_id=None):
    """
    Organize loose artifact files into proper directory structure
    
    Args:
        project_dir: Project directory path (defaults to current directory)
        use_model_id: Optional model ID to organize artifacts for specific model
    """
    if project_dir is None:
        project_dir = Path.cwd()
    else:
        project_dir = Path(project_dir)
    
    print(f"Working in project directory: {project_dir}")
    
    # Use existing /artifacts directory (not models/artifacts)
    artifacts_dir = project_dir / 'artifacts'
    
    # If model_id is provided, create model-specific subdirectory
    if use_model_id:
        artifacts_dir = artifacts_dir / f'model_{use_model_id}'
    
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for different types of artifacts
    subdirs = [
        'training_logs',
        'model_summaries', 
        'predictions',
        'training_curves',
        'configs'
    ]
    
    for subdir in subdirs:
        (artifacts_dir / subdir).mkdir(exist_ok=True)
    
    print(f"Ensured artifacts directory structure in {artifacts_dir}")
    
    # Define file patterns for organization
    file_patterns = {
        'training_logs': ['*.log', 'training*.log'],
        'model_summaries': ['model_summary*.txt', '*summary*.txt'],
        'predictions': ['predictions_*.png', 'pred_*.png', '*predictions*.png'],
        'training_curves': ['training_curves_*.png', '*_curves_*.png', '*loss*.png', '*accuracy*.png'],
        'configs': ['training_config*.json', 'config*.json', '*config*.json']
    }
    
    moved_files = []
    
    # Look for loose files in project root and artifacts root
    search_dirs = [project_dir, project_dir / 'artifacts']
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
            
        print(f"Searching for loose files in: {search_dir}")
        
        for category, patterns in file_patterns.items():
            target_dir = artifacts_dir / category
            
            for pattern in patterns:
                for file_path in search_dir.glob(pattern):
                    # Skip files already in subdirectories
                    if file_path.parent.name in subdirs:
                        continue
                        
                    # Skip files already in the target directory
                    if file_path.parent == target_dir:
                        continue
                        
                    if file_path.is_file():
                        target_path = target_dir / file_path.name
                        
                        # If file already exists, add timestamp
                        if target_path.exists():
                            stem = target_path.stem
                            suffix = target_path.suffix
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            target_path = target_dir / f"{stem}_{timestamp}{suffix}"
                        
                        shutil.move(str(file_path), str(target_path))
                        moved_files.append(f"{file_path.name} -> {category}/")
                        print(f"Moved {file_path.name} to {category}/")
    
    if not moved_files:
        print("No loose artifact files found to move")
    else:
        print(f"\nMoved {len(moved_files)} files:")
        for move in moved_files:
            print(f"  {move}")
    
    return artifacts_dir

def create_gitignore_for_artifacts(project_dir=None):
    """Create .gitignore for artifacts directory"""
    if project_dir is None:
        project_dir = Path.cwd()
    else:
        project_dir = Path(project_dir)
        
    artifacts_dir = project_dir / 'artifacts'
    artifacts_dir.mkdir(exist_ok=True)
    
    gitignore_path = artifacts_dir / '.gitignore'
    
    gitignore_content = """# Ignore all files in artifacts directory except structure
*
!.gitignore
!README.md

# But keep directory structure
!*/
!*/.gitkeep
"""
    
    with open(gitignore_path, 'w') as f:
        f.write(gitignore_content)
    
    # Create .gitkeep files to preserve directory structure
    subdirs = ['training_logs', 'model_summaries', 'predictions', 'training_curves', 'configs']
    for subdir in subdirs:
        subdir_path = artifacts_dir / subdir
        subdir_path.mkdir(exist_ok=True)
        gitkeep_path = subdir_path / '.gitkeep'
        gitkeep_path.touch()
    
    print(f"Created .gitignore and .gitkeep files for artifacts")

def create_artifacts_readme(project_dir=None):
    """Create README for artifacts directory"""
    if project_dir is None:
        project_dir = Path.cwd()
    else:
        project_dir = Path(project_dir)
        
    artifacts_dir = project_dir / 'artifacts'
    readme_path = artifacts_dir / 'README.md'
    
    readme_content = """# Artifacts Directory

This directory contains training artifacts organized by type:

## Directory Structure

- **training_logs/**: Training log files and console outputs
- **model_summaries/**: Model architecture summaries and parameter counts
- **predictions/**: Sample prediction images and comparisons
- **training_curves/**: Training progress plots and metrics visualizations
- **configs/**: Training configuration files and hyperparameters

## File Naming Conventions

- Training logs: `training_*.log`, `model_*.log`
- Model summaries: `model_summary_*.txt`, `*summary*.txt`
- Predictions: `predictions_epoch_*.png`, `pred_*.png`
- Training curves: `training_curves_*.png`, `*_curves_*.png`
- Configs: `training_config_*.json`, `config_*.json`

## Usage

This directory is automatically populated during training runs. Files are organized by the `organize_artifacts.py` script to maintain a clean project structure.

All files in this directory are ignored by git except for this README and the directory structure.

## Model-Specific Organization

You can organize artifacts for a specific model using:
```bash
python organize_artifacts.py --model-id 123
```

This will create `artifacts/model_123/` with the organized structure.
"""
    
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"Created README.md for artifacts directory")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize project artifacts")
    parser.add_argument('--project-dir', type=str, help='Project directory path')
    parser.add_argument('--model-id', type=str, help='Model ID for organizing artifacts')
    parser.add_argument('--clean-only', action='store_true', help='Only create directory structure without moving files')
    
    args = parser.parse_args()
    
    print("Organizing project artifacts...")
    
    if args.clean_only:
        print("Creating clean directory structure only...")
        create_gitignore_for_artifacts(args.project_dir)
        create_artifacts_readme(args.project_dir)
    else:
        artifacts_dir = organize_artifacts(args.project_dir, args.model_id)
        create_gitignore_for_artifacts(args.project_dir)
        create_artifacts_readme(args.project_dir)
        print(f"\nArtifacts organization complete! All files organized in {artifacts_dir}")
