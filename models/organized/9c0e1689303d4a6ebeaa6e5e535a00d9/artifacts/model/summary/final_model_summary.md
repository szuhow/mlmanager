# Final Model Summary

## Model Information
- **Architecture**: U-Net Classifier
- **Framework**: PyTorch
- **Model Family**: UNet-Coronary

## Best Performance
| Metric | Value |
|--------|-------|
| best_val_dice | 0.705357 |
| final_train_loss | 0.692342 |
| final_val_loss | 0.583242 |
| final_train_dice | 0.587891 |
| final_val_dice | 0.705357 |
| total_epochs_trained | 1 |
| convergence_epoch | 1 |


## Model Directory Structure
```
unet-coronary_20250617_094410_a66a872f_v1.0.0/
├── artifacts
│   artifacts/
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
Generated at: 1750146380008
