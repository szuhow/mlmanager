"""
MLflow Artifact Manager - Enhanced artifact saving with organized structure
Based on best practices for ML experiment tracking and artifact organization
"""

import os
import mlflow
import tempfile
import shutil
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class MLflowArtifactManager:
    """
    Enhanced MLflow artifact manager for organized experiment tracking
    
    Features:
    - Hierarchical artifact organization
    - Versioned artifact tracking
    - Automatic artifact categorization
    - Metadata preservation
    - Cleanup and rollback capabilities
    """
    
    def __init__(self, run_id: Optional[str] = None):
        """
        Initialize the artifact manager
        
        Args:
            run_id: MLflow run ID (uses active run if None)
        """
        self.run_id = run_id
        self.temp_dirs = []  # Track temporary directories for cleanup
        
    def __enter__(self):
        """Context manager entry"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup temporary directories"""
        self.cleanup_temp_dirs()
    
    def cleanup_temp_dirs(self):
        """Clean up temporary directories"""
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir {temp_dir}: {e}")
        self.temp_dirs.clear()
    
    def create_temp_dir(self) -> str:
        """Create and track a temporary directory"""
        temp_dir = tempfile.mkdtemp(prefix="mlflow_artifacts_")
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def log_training_artifacts(self, 
                              epoch: int,
                              model_state: Any,
                              metrics: Dict[str, float],
                              artifacts: Dict[str, str],
                              metadata: Optional[Dict] = None) -> Dict[str, str]:
        """
        Log comprehensive training artifacts for an epoch
        
        Args:
            epoch: Current epoch number
            model_state: Model state dict or model object
            metrics: Training metrics for this epoch
            artifacts: Dictionary of artifact_type -> file_path
            metadata: Additional metadata to log
        
        Returns:
            Dictionary of logged artifact paths in MLflow
        """
        logged_paths = {}
        
        try:
            # Create organized temporary structure
            temp_dir = self.create_temp_dir()
            epoch_dir = os.path.join(temp_dir, f"epoch_{epoch:03d}")
            os.makedirs(epoch_dir, exist_ok=True)
            
            # 1. Log model checkpoint
            if model_state is not None:
                model_path = self._save_model_checkpoint(model_state, epoch_dir, epoch)
                if model_path:
                    mlflow.log_artifact(model_path, artifact_path=f"checkpoints/epoch_{epoch:03d}")
                    logged_paths['model_checkpoint'] = f"checkpoints/epoch_{epoch:03d}/model_epoch_{epoch:03d}.pth"
            
            # 2. Log metrics as JSON
            metrics_path = self._save_metrics_json(metrics, epoch_dir, epoch)
            mlflow.log_artifact(metrics_path, artifact_path=f"metrics/epoch_{epoch:03d}")
            logged_paths['metrics'] = f"metrics/epoch_{epoch:03d}/metrics_epoch_{epoch:03d}.json"
            
            # 3. Log categorized artifacts
            for artifact_type, file_path in artifacts.items():
                if file_path and os.path.exists(file_path):
                    target_path = self._get_artifact_path(artifact_type, epoch)
                    mlflow.log_artifact(file_path, artifact_path=target_path)
                    logged_paths[artifact_type] = f"{target_path}/{os.path.basename(file_path)}"
            
            # 4. Log metadata
            if metadata:
                metadata_path = self._save_metadata_json(metadata, epoch_dir, epoch)
                mlflow.log_artifact(metadata_path, artifact_path=f"metadata/epoch_{epoch:03d}")
                logged_paths['metadata'] = f"metadata/epoch_{epoch:03d}/metadata_epoch_{epoch:03d}.json"
            
            # 5. Log epoch summary
            summary_path = self._create_epoch_summary(epoch, metrics, artifacts, epoch_dir)
            mlflow.log_artifact(summary_path, artifact_path=f"summaries/epoch_{epoch:03d}")
            logged_paths['summary'] = f"summaries/epoch_{epoch:03d}/summary_epoch_{epoch:03d}.md"
            
            logger.info(f"Successfully logged {len(logged_paths)} artifacts for epoch {epoch}")
            return logged_paths
            
        except Exception as e:
            logger.error(f"Failed to log training artifacts for epoch {epoch}: {e}")
            raise
    
    def log_model_artifacts(self,
                           model_info: Dict,
                           model_directory: str,
                           best_metrics: Dict[str, float]) -> Dict[str, str]:
        """
        Log final model artifacts with comprehensive metadata
        
        Args:
            model_info: Model architecture and configuration info
            model_directory: Path to organized model directory
            best_metrics: Best training metrics achieved
        
        Returns:
            Dictionary of logged artifact paths
        """
        logged_paths = {}
        
        try:
            # 1. Log model weights
            weights_path = os.path.join(model_directory, "weights", "model.pth")
            if os.path.exists(weights_path):
                mlflow.log_artifact(weights_path, artifact_path="model/weights")
                logged_paths['weights'] = "model/weights/model.pth"
            
            # 2. Log model configuration
            config_path = self._save_model_config(model_info, model_directory)
            if config_path:
                mlflow.log_artifact(config_path, artifact_path="model/config")
                logged_paths['config'] = "model/config/model_config.json"
            
            # 3. Log training artifacts directory
            artifacts_dir = os.path.join(model_directory, "artifacts")
            if os.path.exists(artifacts_dir):
                for root, dirs, files in os.walk(artifacts_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, artifacts_dir)
                        dir_path = os.path.dirname(rel_path)
                        if dir_path:
                            artifact_path = f"training_artifacts/{dir_path}"
                        else:
                            artifact_path = "training_artifacts"
                        mlflow.log_artifact(file_path, artifact_path=artifact_path)
            
            # 4. Log final model summary
            summary_path = self._create_final_model_summary(model_info, best_metrics, model_directory)
            mlflow.log_artifact(summary_path, artifact_path="model/summary")
            logged_paths['model_summary'] = "model/summary/final_model_summary.md"
            
            logger.info(f"Successfully logged final model artifacts: {len(logged_paths)} items")
            return logged_paths
            
        except Exception as e:
            logger.error(f"Failed to log model artifacts: {e}")
            raise
    
    def log_prediction_artifacts(self,
                                predictions_dir: str,
                                epoch: Optional[int] = None) -> List[str]:
        """
        Log prediction images and results
        
        Args:
            predictions_dir: Directory containing prediction files
            epoch: Optional epoch number for organization
        
        Returns:
            List of logged artifact paths
        """
        logged_paths = []
        
        try:
            if not os.path.exists(predictions_dir):
                logger.warning(f"Predictions directory not found: {predictions_dir}")
                return logged_paths
            
            # Determine artifact path
            if epoch is not None:
                artifact_path = f"predictions/epoch_{epoch:03d}"
            else:
                artifact_path = "predictions/final"
            
            # Log all prediction files
            for file in os.listdir(predictions_dir):
                file_path = os.path.join(predictions_dir, file)
                if os.path.isfile(file_path):
                    mlflow.log_artifact(file_path, artifact_path=artifact_path)
                    logged_paths.append(f"{artifact_path}/{file}")
            
            logger.info(f"Logged {len(logged_paths)} prediction artifacts")
            return logged_paths
            
        except Exception as e:
            logger.error(f"Failed to log prediction artifacts: {e}")
            raise
    
    def log_training_logs(self, 
                         log_files: List[str],
                         log_type: str = "training") -> List[str]:
        """
        Log training log files with organization
        
        Args:
            log_files: List of log file paths
            log_type: Type of logs (training, validation, etc.)
        
        Returns:
            List of logged artifact paths
        """
        logged_paths = []
        
        try:
            for log_file in log_files:
                if os.path.exists(log_file):
                    artifact_path = f"logs/{log_type}"
                    mlflow.log_artifact(log_file, artifact_path=artifact_path)
                    logged_paths.append(f"{artifact_path}/{os.path.basename(log_file)}")
            
            logger.info(f"Logged {len(logged_paths)} log files")
            return logged_paths
            
        except Exception as e:
            logger.error(f"Failed to log training logs: {e}")
            raise
    
    def _save_model_checkpoint(self, model_state: Any, epoch_dir: str, epoch: int) -> str:
        """Save model checkpoint"""
        import torch
        
        checkpoint_path = os.path.join(epoch_dir, f"model_epoch_{epoch:03d}.pth")
        
        if hasattr(model_state, 'state_dict'):
            # PyTorch model object
            torch.save(model_state.state_dict(), checkpoint_path)
        elif isinstance(model_state, dict):
            # State dict
            torch.save(model_state, checkpoint_path)
        else:
            # Assume it's already a state dict
            torch.save(model_state, checkpoint_path)
        
        return checkpoint_path
    
    def _save_metrics_json(self, metrics: Dict[str, float], epoch_dir: str, epoch: int) -> str:
        """Save metrics as JSON"""
        metrics_path = os.path.join(epoch_dir, f"metrics_epoch_{epoch:03d}.json")
        
        # Add timestamp and epoch info
        enhanced_metrics = {
            'epoch': epoch,
            'timestamp': mlflow.utils.time.get_current_time_millis(),
            'metrics': metrics
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(enhanced_metrics, f, indent=2, default=str)
        
        return metrics_path
    
    def _save_metadata_json(self, metadata: Dict, epoch_dir: str, epoch: int) -> str:
        """Save metadata as JSON"""
        metadata_path = os.path.join(epoch_dir, f"metadata_epoch_{epoch:03d}.json")
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return metadata_path
    
    def _save_model_config(self, model_info: Dict, model_directory: str) -> str:
        """Save model configuration"""
        config_dir = os.path.join(model_directory, "config")
        os.makedirs(config_dir, exist_ok=True)
        
        config_path = os.path.join(config_dir, "model_config.json")
        
        with open(config_path, 'w') as f:
            json.dump(model_info, f, indent=2, default=str)
        
        return config_path
    
    def _get_artifact_path(self, artifact_type: str, epoch: int) -> str:
        """Get organized artifact path based on type"""
        type_mapping = {
            'training_curves': f'visualizations/training_curves/epoch_{epoch:03d}',
            'predictions': f'predictions/epoch_{epoch:03d}',
            'sample_images': f'samples/epoch_{epoch:03d}',
            'model_summary': f'summaries/model/epoch_{epoch:03d}',
            'config': f'config/epoch_{epoch:03d}',
            'logs': f'logs/epoch_{epoch:03d}',
            'metrics_plot': f'visualizations/metrics/epoch_{epoch:03d}',
            'comparison': f'visualizations/comparisons/epoch_{epoch:03d}'
        }
        
        return type_mapping.get(artifact_type, f'misc/epoch_{epoch:03d}')
    
    def _create_epoch_summary(self, epoch: int, metrics: Dict, artifacts: Dict, epoch_dir: str) -> str:
        """Create epoch summary markdown file"""
        summary_path = os.path.join(epoch_dir, f"summary_epoch_{epoch:03d}.md")
        
        summary_content = f"""# Epoch {epoch} Summary

## Metrics
{self._format_metrics_table(metrics)}

## Artifacts
{self._format_artifacts_list(artifacts)}

## Timestamp
Generated at: {mlflow.utils.time.get_current_time_millis()}
"""
        
        with open(summary_path, 'w') as f:
            f.write(summary_content)
        
        return summary_path
    
    def _create_final_model_summary(self, model_info: Dict, best_metrics: Dict, model_directory: str) -> str:
        """Create final model summary"""
        summary_path = os.path.join(model_directory, "final_model_summary.md")
        
        summary_content = f"""# Final Model Summary

## Model Information
- **Architecture**: {model_info.get('architecture', 'Unknown')}
- **Framework**: {model_info.get('framework', 'PyTorch')}
- **Model Family**: {model_info.get('model_family', 'UNet-Coronary')}

## Best Performance
{self._format_metrics_table(best_metrics)}

## Model Directory Structure
```
{self._get_directory_tree(model_directory)}
```

## Timestamp
Generated at: {mlflow.utils.time.get_current_time_millis()}
"""
        
        with open(summary_path, 'w') as f:
            f.write(summary_content)
        
        return summary_path
    
    def _format_metrics_table(self, metrics: Dict) -> str:
        """Format metrics as markdown table"""
        if not metrics:
            return "No metrics available"
        
        table = "| Metric | Value |\n|--------|-------|\n"
        for key, value in metrics.items():
            if isinstance(value, float):
                table += f"| {key} | {value:.6f} |\n"
            else:
                table += f"| {key} | {value} |\n"
        
        return table
    
    def _format_artifacts_list(self, artifacts: Dict) -> str:
        """Format artifacts as markdown list"""
        if not artifacts:
            return "No artifacts available"
        
        artifact_list = ""
        for artifact_type, file_path in artifacts.items():
            artifact_list += f"- **{artifact_type}**: `{file_path}`\n"
        
        return artifact_list
    
    def _get_directory_tree(self, directory: str, max_depth: int = 3) -> str:
        """Get directory tree structure"""
        tree_lines = []
        
        def add_tree_line(path, prefix="", depth=0):
            if depth > max_depth:
                return
            
            if os.path.isdir(path):
                tree_lines.append(f"{prefix}{os.path.basename(path)}/")
                try:
                    items = sorted(os.listdir(path))
                    for i, item in enumerate(items):
                        item_path = os.path.join(path, item)
                        is_last = i == len(items) - 1
                        new_prefix = prefix + ("    " if is_last else "│   ")
                        tree_prefix = prefix + ("└── " if is_last else "├── ")
                        tree_lines.append(f"{tree_prefix}{item}")
                        if os.path.isdir(item_path) and depth < max_depth:
                            add_tree_line(item_path, new_prefix, depth + 1)
                except PermissionError:
                    tree_lines.append(f"{prefix}    [Permission Denied]")
        
        add_tree_line(directory)
        return "\n".join(tree_lines[:50])  # Limit output


# Convenience functions for easy integration
def log_epoch_artifacts(epoch: int, 
                       model_state: Any = None,
                       metrics: Dict[str, float] = None,
                       artifacts: Dict[str, str] = None,
                       metadata: Dict = None) -> Dict[str, str]:
    """
    Convenience function to log epoch artifacts
    
    Usage:
        artifacts_dict = {
            'training_curves': '/path/to/curves.png',
            'predictions': '/path/to/predictions.png',
            'sample_images': '/path/to/samples.png'
        }
        
        logged_paths = log_epoch_artifacts(
            epoch=5,
            model_state=model.state_dict(),
            metrics={'train_loss': 0.15, 'val_dice': 0.89},
            artifacts=artifacts_dict,
            metadata={'learning_rate': 0.001, 'batch_size': 32}
        )
    """
    with MLflowArtifactManager() as manager:
        return manager.log_training_artifacts(
            epoch=epoch,
            model_state=model_state,
            metrics=metrics or {},
            artifacts=artifacts or {},
            metadata=metadata
        )


def log_final_model(model_info: Dict,
                   model_directory: str,
                   best_metrics: Dict[str, float]) -> Dict[str, str]:
    """
    Convenience function to log final model artifacts
    
    Usage:
        model_info = {
            'architecture': 'MONAI UNet',
            'framework': 'PyTorch',
            'model_family': 'UNet-Coronary',
            'parameters': 1234567
        }
        
        logged_paths = log_final_model(
            model_info=model_info,
            model_directory='/path/to/model/dir',
            best_metrics={'best_val_dice': 0.92, 'best_val_loss': 0.08}
        )
    """
    with MLflowArtifactManager() as manager:
        return manager.log_model_artifacts(
            model_info=model_info,
            model_directory=model_directory,
            best_metrics=best_metrics
        )


# Example usage integration function
def enhanced_mlflow_logging_example():
    """
    Example of how to integrate enhanced MLflow logging into training loop
    """
    
    # During training loop for each epoch:
    epoch = 5
    metrics = {
        'train_loss': 0.15,
        'val_loss': 0.12,
        'train_dice': 0.85,
        'val_dice': 0.89,
        'learning_rate': 0.001
    }
    
    # Prepare artifacts dictionary
    artifacts = {
        'training_curves': '/tmp/training_curves_epoch_5.png',
        'predictions': '/tmp/predictions_epoch_5.png',
        'sample_images': '/tmp/samples_epoch_5.png'
    }
    
    # Additional metadata
    metadata = {
        'batch_size': 32,
        'optimizer': 'Adam',
        'data_augmentation': True,
        'model_params': 1234567
    }
    
    # Log everything
    logged_paths = log_epoch_artifacts(
        epoch=epoch,
        model_state=None,  # model.state_dict() if saving checkpoints
        metrics=metrics,
        artifacts=artifacts,
        metadata=metadata
    )
    
    print(f"Logged artifacts: {logged_paths}")
    
    # At the end of training:
    model_info = {
        'architecture': 'MONAI UNet',
        'framework': 'PyTorch',
        'model_family': 'UNet-Coronary',
        'total_parameters': 1234567,
        'trainable_parameters': 1234567
    }
    
    best_metrics = {
        'best_val_dice': 0.92,
        'best_val_loss': 0.08,
        'final_train_loss': 0.10
    }
    
    final_paths = log_final_model(
        model_info=model_info,
        model_directory='/path/to/organized/model/dir',
        best_metrics=best_metrics
    )
    
    print(f"Final model artifacts: {final_paths}")
