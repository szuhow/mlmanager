# Enhanced MLflow Artifact Management

This document describes the enhanced MLflow artifact management system implemented for the coronary segmentation project.

## Overview

The enhanced MLflow artifact management system provides:

1. **Hierarchical organization** of training artifacts
2. **Automatic categorization** based on artifact type
3. **Versioned tracking** with epoch-based organization
4. **Comprehensive metadata** preservation
5. **Fallback mechanisms** for robustness

## Key Features

### 1. Organized Artifact Structure

The new system organizes artifacts in a hierarchical structure:

```
mlflow_run/
├── checkpoints/
│   ├── epoch_001/
│   ├── epoch_005/
│   └── epoch_010/
├── metrics/
│   ├── epoch_001/
│   ├── epoch_005/
│   └── epoch_010/
├── visualizations/
│   ├── training_curves/
│   ├── predictions/
│   └── comparisons/
├── logs/
│   ├── training/
│   └── validation/
├── model/
│   ├── weights/
│   ├── config/
│   └── summary/
└── metadata/
    ├── epoch_001/
    ├── epoch_005/
    └── epoch_010/
```

### 2. Enhanced Artifact Types

The system categorizes artifacts into the following types:

- **Model Checkpoints**: Saved model state dictionaries
- **Training Curves**: Loss and metric visualization plots
- **Predictions**: Sample prediction images and comparisons
- **Model Summaries**: Architecture details and parameter counts
- **Configuration**: Training parameters and hyperparameters
- **Logs**: Training and validation logs
- **Metrics**: JSON files with detailed metrics
- **Metadata**: Additional context and experiment information

### 3. Automatic Metadata Generation

Each epoch automatically generates:

- **Metrics JSON**: Complete metrics with timestamps
- **Metadata JSON**: Training context and configuration
- **Summary Markdown**: Human-readable epoch summary

## Usage Examples

### Basic Usage in Training Loop

```python
from shared.utils.mlflow_artifact_manager import log_epoch_artifacts

# During training loop
epoch = 5
metrics = {
    'train_loss': 0.15,
    'val_loss': 0.12,
    'train_dice': 0.85,
    'val_dice': 0.89
}

artifacts = {
    'training_curves': '/path/to/curves.png',
    'predictions': '/path/to/predictions.png',
    'sample_images': '/path/to/samples.png'
}

metadata = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'optimizer': 'Adam'
}

logged_paths = log_epoch_artifacts(
    epoch=epoch,
    model_state=model.state_dict(),
    metrics=metrics,
    artifacts=artifacts,
    metadata=metadata
)
```

### Final Model Logging

```python
from shared.utils.mlflow_artifact_manager import log_final_model

model_info = {
    'architecture': 'MONAI UNet',
    'framework': 'PyTorch',
    'model_family': 'UNet-Coronary',
    'total_parameters': 1234567
}

best_metrics = {
    'best_val_dice': 0.92,
    'best_val_loss': 0.08,
    'convergence_epoch': 15
}

final_paths = log_final_model(
    model_info=model_info,
    model_directory='/path/to/model/dir',
    best_metrics=best_metrics
)
```

### Context Manager for Resource Management

```python
from shared.utils.mlflow_artifact_manager import MLflowArtifactManager

with MLflowArtifactManager() as manager:
    # Automatic cleanup of temporary directories
    logged_paths = manager.log_training_artifacts(...)
    prediction_paths = manager.log_prediction_artifacts(...)
```

## Integration with Existing System

The enhanced artifact manager is integrated into the existing training pipeline with:

1. **Fallback mechanisms**: If enhanced logging fails, the system falls back to basic MLflow logging
2. **Backward compatibility**: Existing artifact paths and names are preserved
3. **Gradual adoption**: Can be enabled/disabled per training run

## Configuration

The artifact manager uses these configuration options:

- **Hierarchical organization**: Automatically enabled
- **Checkpoint saving**: Only for best models by default
- **Temporary directory cleanup**: Automatic via context manager
- **Artifact categorization**: Based on file type and content

## Benefits

### For Data Scientists
- **Easy artifact discovery**: Hierarchical organization makes finding specific artifacts intuitive
- **Rich metadata**: Comprehensive context for each experiment
- **Visual summaries**: Automatic generation of markdown summaries

### For MLOps
- **Consistent structure**: Standardized artifact organization across experiments
- **Version tracking**: Clear epoch-based versioning
- **Resource management**: Automatic cleanup of temporary files

### for Reproducibility
- **Complete context**: All training context preserved as metadata
- **Artifact relationships**: Clear mapping between epochs and artifacts
- **Configuration tracking**: Full hyperparameter and environment tracking

## Implementation Details

### Artifact Path Mapping

The system uses intelligent path mapping:

```python
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
```

### Automatic Summarization

Each epoch generates comprehensive summaries:

```markdown
# Epoch 5 Summary

## Metrics
| Metric | Value |
|--------|-------|
| train_loss | 0.150000 |
| val_loss | 0.120000 |
| train_dice | 0.850000 |
| val_dice | 0.890000 |

## Artifacts
- **training_curves**: `/visualizations/training_curves/epoch_005/curves.png`
- **predictions**: `/predictions/epoch_005/predictions.png`

## Timestamp
Generated at: 1672531200000
```

### Error Handling

The system implements comprehensive error handling:

1. **Graceful degradation**: Falls back to basic logging on errors
2. **Resource cleanup**: Automatic cleanup of temporary directories
3. **Detailed logging**: Clear error messages and context
4. **Rollback capabilities**: Can recover from partial failures

## Migration Guide

To adopt the enhanced artifact manager:

1. **Update imports**:
   ```python
   from shared.utils.mlflow_artifact_manager import log_epoch_artifacts
   ```

2. **Replace basic logging**:
   ```python
   # Old way
   mlflow.log_artifact(file_path)
   
   # New way
   log_epoch_artifacts(epoch=1, artifacts={'type': file_path})
   ```

3. **Add metadata**:
   ```python
   metadata = {
       'learning_rate': lr,
       'batch_size': bs,
       'optimizer': 'Adam'
   }
   ```

4. **Use context managers**:
   ```python
   with MLflowArtifactManager() as manager:
       # All operations
   ```

## Future Enhancements

Planned improvements include:

1. **Automatic artifact compression**: For large files
2. **Artifact deduplication**: Avoid storing identical files
3. **Cross-run artifact comparison**: Compare artifacts across experiments
4. **Artifact lineage tracking**: Track artifact dependencies
5. **Performance metrics**: Track artifact logging performance
6. **Cloud storage optimization**: Optimized uploads for cloud backends

## Troubleshooting

### Common Issues

1. **Permission errors**: Ensure write access to temporary directories
2. **Disk space**: Monitor available disk space for large artifacts
3. **Network timeouts**: Configure appropriate timeouts for cloud backends
4. **Memory usage**: Large artifacts may require streaming uploads

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.getLogger('shared.utils.mlflow_artifact_manager').setLevel(logging.DEBUG)
```

## Performance Considerations

- **Temporary storage**: Uses system temp directory, ensure adequate space
- **Concurrent access**: Thread-safe for multiple training processes
- **Memory efficiency**: Streams large files to avoid memory issues
- **Network optimization**: Batches uploads when possible

## Compliance and Security

- **Data privacy**: No sensitive data in metadata by default
- **Access control**: Respects MLflow's existing access controls
- **Audit trail**: Complete logging of all artifact operations
- **Encryption**: Uses MLflow's encryption capabilities when configured
