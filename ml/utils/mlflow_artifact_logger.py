"""
Enhanced MLflow Artifact Logger
Ensures all training artifacts are properly logged to MLflow
"""

import os
import sys
import logging
import mlflow
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class MLflowArtifactLogger:
    """Enhanced artifact logger for MLflow"""
    
    def __init__(self, run_id: str = None):
        self.run_id = run_id
        self.logged_artifacts = []
        
    def ensure_mlflow_connection(self):
        """Ensure MLflow is properly connected"""
        try:
            # Check if we have an active run
            current_run = mlflow.active_run()
            if not current_run:
                if self.run_id:
                    mlflow.start_run(run_id=self.run_id)
                    logger.info(f"[MLFLOW] Started run: {self.run_id}")
                else:
                    mlflow.start_run()
                    logger.info("[MLFLOW] Started new run")
            
            # Test connection by getting experiment info
            exp_id = mlflow.active_run().info.experiment_id
            experiment = mlflow.get_experiment(exp_id)
            logger.info(f"[MLFLOW] Connected to experiment: {experiment.name}")
            logger.info(f"[MLFLOW] Artifact location: {experiment.artifact_location}")
            
            return True
            
        except Exception as e:
            logger.error(f"[MLFLOW] Connection failed: {e}")
            return False
    
    def log_artifact_with_retry(self, file_path: str, artifact_path: str = None, max_retries: int = 3):
        """Log artifact with retry mechanism"""
        if not os.path.exists(file_path):
            logger.warning(f"[ARTIFACT] File not found: {file_path}")
            return False
            
        for attempt in range(max_retries):
            try:
                mlflow.log_artifact(file_path, artifact_path=artifact_path)
                rel_path = f"{artifact_path}/{os.path.basename(file_path)}" if artifact_path else os.path.basename(file_path)
                self.logged_artifacts.append(rel_path)
                logger.info(f"[ARTIFACT] Logged: {rel_path}")
                return True
                
            except Exception as e:
                logger.warning(f"[ARTIFACT] Attempt {attempt + 1} failed for {file_path}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"[ARTIFACT] Failed to log {file_path} after {max_retries} attempts")
        
        return False
    
    def log_directory_artifacts(self, directory: str, base_artifact_path: str = None):
        """Log all artifacts in a directory"""
        if not os.path.exists(directory):
            logger.warning(f"[ARTIFACT] Directory not found: {directory}")
            return []
        
        logged_files = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                # Only log relevant file types
                if file.endswith(('.png', '.jpg', '.jpeg', '.json', '.txt', '.html', '.pth', '.pkl', '.csv')):
                    file_path = os.path.join(root, file)
                    
                    # Calculate relative artifact path
                    rel_dir = os.path.relpath(root, directory)
                    if rel_dir == '.':
                        artifact_path = base_artifact_path
                    else:
                        artifact_path = f"{base_artifact_path}/{rel_dir}" if base_artifact_path else rel_dir
                    
                    if self.log_artifact_with_retry(file_path, artifact_path):
                        logged_files.append(file_path)
        
        logger.info(f"[ARTIFACT] Logged {len(logged_files)} files from {directory}")
        return logged_files
    
    def log_training_artifacts(self, model_dir: str):
        """Log all training artifacts from model directory"""
        if not self.ensure_mlflow_connection():
            logger.error("[ARTIFACT] Cannot log artifacts - MLflow connection failed")
            return []
        
        all_logged = []
        
        # Log artifacts directory
        artifacts_dir = os.path.join(model_dir, 'artifacts')
        if os.path.exists(artifacts_dir):
            logged = self.log_directory_artifacts(artifacts_dir, "training_artifacts")
            all_logged.extend(logged)
        
        # Log model weights
        weights_dir = os.path.join(model_dir, 'weights')
        if os.path.exists(weights_dir):
            logged = self.log_directory_artifacts(weights_dir, "model_weights")
            all_logged.extend(logged)
        
        # Log logs directory
        logs_dir = os.path.join(model_dir, 'logs')
        if os.path.exists(logs_dir):
            logged = self.log_directory_artifacts(logs_dir, "training_logs")
            all_logged.extend(logged)
        
        # Log individual important files
        important_files = [
            ('config.json', 'config'),
            ('model_summary.txt', 'model_info'),
            ('training_config.json', 'config'),
        ]
        
        for filename, artifact_path in important_files:
            file_path = os.path.join(model_dir, filename)
            if os.path.exists(file_path):
                self.log_artifact_with_retry(file_path, artifact_path)
                all_logged.append(file_path)
        
        logger.info(f"[ARTIFACT] Total artifacts logged: {len(all_logged)}")
        return all_logged
    
    def get_logged_artifacts(self):
        """Get list of logged artifacts"""
        return self.logged_artifacts.copy()

def force_log_all_artifacts(model_dir: str, run_id: str = None):
    """Force log all artifacts to MLflow - use this as fallback"""
    logger.info(f"[FORCE_LOG] Starting forced artifact logging for {model_dir}")
    
    artifact_logger = MLflowArtifactLogger(run_id)
    logged_artifacts = artifact_logger.log_training_artifacts(model_dir)
    
    if logged_artifacts:
        logger.info(f"[FORCE_LOG] Successfully logged {len(logged_artifacts)} artifacts")
        
        # Log a summary
        try:
            summary_content = f"""# Artifact Logging Summary

## Model Directory: {model_dir}
## Total Artifacts Logged: {len(logged_artifacts)}

### Logged Files:
{chr(10).join([f"- {os.path.basename(f)}" for f in logged_artifacts[:20]])}
{'...' if len(logged_artifacts) > 20 else ''}

Generated at: {str(os.path.getmtime(model_dir))}
"""
            
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                f.write(summary_content)
                summary_path = f.name
            
            mlflow.log_artifact(summary_path, "summaries")
            os.unlink(summary_path)
            
        except Exception as e:
            logger.warning(f"[FORCE_LOG] Failed to create summary: {e}")
    
    return logged_artifacts
