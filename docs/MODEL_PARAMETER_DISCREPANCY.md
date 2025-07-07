# Model Parameter Discrepancy Analysis

## Issue Description
We observed a discrepancy between the number of parameters reported during model inference/training and the number shown in the model architecture preview:

- **Actual model parameters (logs):** 1,625,420
- **Model architecture preview:** ~31 million parameters

## Root Cause Analysis

The discrepancy occurs because:

1. During actual inference/training, the system correctly uses the MONAI UNet implementation with ~1.6M parameters
2. In the model architecture preview, the system was using a fallback implementation that doesn't match the actual MONAI UNet structure

The preview system attempts to create a model for visualization purposes only, and was not using the same model implementation as the training/inference code.

## Solution

We've updated the model summary generation code to:

1. Detect MONAI UNet models specifically
2. Try to import and use the actual `get_monai_unet` function from `ml.utils.monai_utils` for accurate parameter counts
3. Fall back to the standard preview model only if the actual MONAI implementation is not available

This ensures that the parameter count shown in the architecture preview will match the actual model used during training and inference.

## Expected Results

After this fix, the architecture preview should show ~1.6M parameters for MONAI UNet models, matching the count reported in the logs during actual model usage.

## Notes

- The model visualization is purely for UI/display purposes and does not affect actual model training or inference
- The fallback mechanism still exists for other model types, ensuring the preview system works even when specific model implementations aren't directly accessible
