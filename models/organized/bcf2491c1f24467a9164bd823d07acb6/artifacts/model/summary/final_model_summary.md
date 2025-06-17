# Final Model Summary

## Model Information
- **Architecture**: MONAI UNet
- **Framework**: PyTorch
- **Model Family**: UNet-Coronary

## Best Performance
| Metric | Value |
|--------|-------|
| best_val_dice | 0.108394 |
| final_train_loss | 0.915944 |
| final_val_loss | 0.913541 |
| final_train_dice | 0.101730 |
| final_val_dice | 0.108394 |
| total_epochs_trained | 1 |
| convergence_epoch | 1 |


## Model Directory Structure
```
unet-coronary_20250617_121124_53e97df7_v1.0.0/
├── artifacts
│   artifacts/
│   ├── sample_inputs_and_masks.png
│   ├── val_sample_inputs_and_masks.png
│   └── visualizations.json
├── config
│   config/
│   └── model_config.json
├── logs
│   logs/
│   ├── epoch_001_config.json
│   └── training.log
├── metadata.json
├── metrics
│   metrics/
├── model_summary.txt
├── predictions
│   predictions/
│   └── epoch_001
│       epoch_001/
│       └── predictions_epoch_001.png
├── training_config.json
└── weights
    weights/
    ├── best_model_context.json
    └── model.pth
```

## Timestamp
Generated at: 1750155134557
