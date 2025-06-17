# Final Model Summary

## Model Information
- **Architecture**: MONAI UNet
- **Framework**: PyTorch
- **Model Family**: UNet-Coronary

## Best Performance
| Metric | Value |
|--------|-------|
| best_val_dice | 0.099107 |
| final_train_loss | 0.922618 |
| final_val_loss | 0.916948 |
| final_train_dice | 0.085846 |
| final_val_dice | 0.099107 |
| total_epochs_trained | 1 |
| convergence_epoch | 1 |


## Model Directory Structure
```
unet-coronary_20250617_121922_287737c8_v1.0.0/
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
Generated at: 1750155613618
