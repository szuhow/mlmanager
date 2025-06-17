# Final Model Summary

## Model Information
- **Architecture**: U-Net Classifier
- **Framework**: PyTorch
- **Model Family**: UNet-Coronary

## Best Performance
| Metric | Value |
|--------|-------|
| best_val_dice | 0.741071 |
| final_train_loss | 0.680227 |
| final_val_loss | 0.614210 |
| final_train_dice | 0.625000 |
| final_val_dice | 0.732143 |
| total_epochs_trained | 4 |
| convergence_epoch | 3 |


## Model Directory Structure
```
unet-coronary_20250617_095203_2fc7ffe9_v1.0.0/
├── artifacts
│   artifacts/
│   ├── detailed_metrics_epoch_2.json
│   ├── detailed_metrics_epoch_3.json
│   ├── detailed_metrics_epoch_4.json
│   ├── enhanced_training_curves_epoch_2.png
│   ├── enhanced_training_curves_epoch_3.png
│   ├── enhanced_training_curves_epoch_4.png
│   ├── performance_radar_epoch_2.png
│   ├── performance_radar_epoch_3.png
│   ├── performance_radar_epoch_4.png
│   └── visualizations.json
├── config
│   config/
│   └── model_config.json
├── logs
│   logs/
│   ├── epoch_001_config.json
│   ├── epoch_002_config.json
│   ├── epoch_003_config.json
│   ├── epoch_004_config.json
│   └── training.log
├── metadata.json
├── metrics
│   metrics/
├── model_summary.txt
├── predictions
│   predictions/
│   ├── epoch_001
│   │   epoch_001/
│   │   └── predictions_epoch_001.png
│   ├── epoch_002
│   │   epoch_002/
│   │   └── predictions_epoch_002.png
│   ├── epoch_003
│   │   epoch_003/
│   │   └── predictions_epoch_003.png
│   └── epoch_004
│       epoch_004/
│       └── predictions_epoch_004.png
├── training_config.json
└── weights
    weights/
    ├── best_model_context.json
    └── model.pth
```

## Timestamp
Generated at: 1750147317688
