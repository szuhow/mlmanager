import os
import argparse
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms

# Handle both absolute and relative imports
try:
    from ..training.models.unet import UNet
    from ..training.models.classification_models import UNetClassifier, ResUNetClassifier
except ImportError:
    # Fallback for absolute import when script is run directly
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from training.models.unet import UNet
    from training.models.classification_models import UNetClassifier, ResUNetClassifier

def find_model_file(model_dir, epoch):
    model_files = list(Path(model_dir).glob(f'best_model_epoch_{epoch}_*.pth'))
    if not model_files:
        raise FileNotFoundError(f"No model file found for epoch {epoch} in directory {model_dir}")
    return model_files[0]

def load_model(model_path, model_type='segmentation'):
    """Load model based on type (segmentation or classification)"""
    
    # Try to load checkpoint first to inspect metadata
    try:
        checkpoint = torch.load(model_path, weights_only=True, map_location='cpu')
        print(f"✅ Checkpoint loaded from: {model_path}")
        
        # Debug checkpoint structure
        if isinstance(checkpoint, dict):
            print(f"📊 Checkpoint is dict with keys: {list(checkpoint.keys())}")
            if 'model_state_dict' in checkpoint:
                actual_state_dict = checkpoint['model_state_dict']
                print(f"📊 model_state_dict has {len(actual_state_dict)} parameters")
            else:
                print("⚠️ No 'model_state_dict' key found, treating entire checkpoint as state_dict")
        else:
            print(f"📊 Checkpoint is {type(checkpoint)}, not a dict")
            
    except Exception as e:
        raise Exception(f"Failed to load checkpoint from {model_path}: {e}")
    
    # Check if we have model metadata that indicates the model type
    model_metadata = checkpoint.get('model_metadata', {})
    detected_type = model_metadata.get('model_type', model_type)
    
    # Get model configuration
    input_channels = model_metadata.get('input_channels', 3)
    num_classes = model_metadata.get('num_classes', 1)
    
    print(f"Loading model: type={detected_type}, channels={input_channels}, classes={num_classes}")
    
    if detected_type == 'classification' or 'classifier' in model_path.lower():
        # Classification model
        if num_classes <= 2:
            print("Loading UNetClassifier for binary classification")
            model = UNetClassifier(input_channels=input_channels, num_classes=num_classes)
        else:
            print(f"Loading UNetClassifier for {num_classes}-class classification")
            model = UNetClassifier(input_channels=input_channels, num_classes=num_classes)
    else:
        # Segmentation model (default)
        print("Loading UNet for segmentation")
        model = UNet(n_channels=input_channels, n_classes=num_classes, bilinear=False)
    
    # Load state dict with enhanced error handling for parameter mismatches
    try:
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            # Fallback: assume the checkpoint is the state dict directly
            state_dict = checkpoint
        
        # Debug: Print some information about the checkpoint and model
        print(f"📊 Checkpoint keys: {len(state_dict.keys())} parameters")
        print(f"📊 Model keys: {len(model.state_dict().keys())} parameters")
        
        # Show first few keys from both
        checkpoint_keys = list(state_dict.keys())[:5]
        model_keys = list(model.state_dict().keys())[:5]
        print(f"🔍 First checkpoint keys: {checkpoint_keys}")
        print(f"🔍 First model keys: {model_keys}")
        
        # Try direct loading first
        model.load_state_dict(state_dict, strict=True)
        print("✅ Model loaded successfully with strict=True")
        
    except RuntimeError as e:
        if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
            print(f"⚠️ State dict mismatch detected: {e}")
            print("🔧 Attempting to fix parameter name mismatches...")
            
            # Common fix: Remove prefixes like 'module.' or 'model.' from keys
            fixed_state_dict = {}
            for key, value in state_dict.items():
                # Remove common prefixes
                new_key = key
                for prefix in ['module.', 'model.', '_orig_mod.']:
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                        break
                fixed_state_dict[new_key] = value
            
            # Check if prefix removal helped
            if fixed_state_dict != state_dict:
                print("🔄 Trying with prefix-cleaned state dict...")
                try:
                    model.load_state_dict(fixed_state_dict, strict=True)
                    print("✅ Model loaded successfully after prefix removal!")
                    return model
                except RuntimeError as prefix_error:
                    print(f"⚠️ Prefix removal didn't help: {prefix_error}")
                    state_dict = fixed_state_dict  # Use cleaned version for further attempts
            
            # Try loading with strict=False to allow missing keys
            try:
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    print(f"⚠️ Missing keys in model: {missing_keys[:10]}")  # Show first 10
                if unexpected_keys:
                    print(f"⚠️ Unexpected keys in state_dict: {unexpected_keys[:10]}")  # Show first 10
                
                # Check if the model architecture is fundamentally different
                model_param_names = set(model.state_dict().keys())
                checkpoint_param_names = set(state_dict.keys())
                
                # If there's a significant mismatch, try creating model with different parameters
                if len(model_param_names.intersection(checkpoint_param_names)) < len(model_param_names) * 0.5:
                    print("🔄 Significant parameter mismatch detected, trying alternative model creation...")
                    
                    # For classification models, try both naming conventions
                    if detected_type == 'classification':
                        try:
                            # Try with n_channels/n_classes naming convention
                            model_alt = UNetClassifier(n_channels=input_channels, n_classes=num_classes)
                            model_alt.load_state_dict(state_dict, strict=False)
                            model = model_alt
                            print("✅ Model loaded with alternative parameter naming (n_channels/n_classes)")
                        except Exception as alt_error:
                            print(f"❌ Alternative naming also failed: {alt_error}")
                            # Continue with the original model with partial loading
                            pass
                    else:
                        # For segmentation models, try different parameter combinations
                        try:
                            # Try different channel configurations based on what's in the checkpoint
                            first_layer_key = next((k for k in state_dict.keys() if 'conv' in k.lower() and 'weight' in k), None)
                            if first_layer_key and len(state_dict[first_layer_key].shape) >= 2:
                                actual_in_channels = state_dict[first_layer_key].shape[1]
                                if actual_in_channels != input_channels:
                                    print(f"🔄 Adjusting input channels from {input_channels} to {actual_in_channels}")
                                    model_alt = UNet(n_channels=actual_in_channels, n_classes=num_classes, bilinear=False)
                                    model_alt.load_state_dict(state_dict, strict=False)
                                    model = model_alt
                                    print("✅ Model loaded with corrected input channels")
                        except Exception as alt_error:
                            print(f"❌ Channel correction failed: {alt_error}")
                            # Continue with the original model with partial loading
                            pass
                
                print("✅ Model loaded with partial state dict (some parameters may be missing)")
                
            except Exception as partial_error:
                print(f"❌ Even partial loading failed: {partial_error}")
                raise RuntimeError(f"Failed to load model state dict: {e}") from e
        else:
            # Re-raise other types of RuntimeError
            raise
    
    model.eval()
    return model, detected_type

def inference(model, image, model_type='segmentation', post_processing_config=None):
    """
    Enhanced inference with configurable post-processing.
    
    Args:
        model: Trained model
        image: Input image
        model_type: 'segmentation' or 'classification'
        post_processing_config: Configuration for post-processing
    """
    # Default post-processing configuration - restored older, simpler settings
    if post_processing_config is None:
        post_processing_config = {
            'threshold': 0.5,  # Back to standard 0.5 threshold
            'min_component_size': 50,
            'morphology_kernel_size': 3,
            'apply_opening': True,
            'apply_closing': False,  # Simplified - less aggressive
            'confidence_threshold': 0.5  # Lowered from 0.6
        }
    
    # Step 2: Preprocess the image to match training transforms
    transform = transforms.Compose([
        # Keep RGB format - do NOT convert to grayscale since models expect 3 channels
        transforms.ToTensor(),         # Convert to a PyTorch tensor (RGB)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Same normalization as training
    ])
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Wykonaj inferencję
    with torch.no_grad():
        output = model(input_tensor)

    if model_type == 'classification':
        # Classification output processing
        print(f"Classification output shape: {output.shape}")
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities.max().item()
        
        print(f"Predicted class: {predicted_class}, Confidence: {confidence:.3f}")
        print(f"All probabilities: {probabilities.flatten()}")
        
        # Return classification results
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy(),
            'type': 'classification'
        }
    else:
        # Segmentation output processing - restored to older version
        num_output_channels = output.shape[1]
        if num_output_channels == 1:
            # Binary segmentation - apply sigmoid and threshold
            output = torch.sigmoid(output)  # Apply sigmoid for binary segmentation
            # Apply threshold from config (default 0.5 instead of 0.6)
            threshold = post_processing_config.get('threshold', 0.5)
            output = (output > threshold).float()
            print(f"Applied binary segmentation (sigmoid + threshold {threshold})")
        else:
            # Multi-class semantic segmentation - use softmax but keep probabilities
            probabilities = torch.softmax(output, dim=1)  # Apply softmax for multi-class
            # For multi-class, we might want to keep probabilities or apply threshold per class
            output = probabilities
            print(f"Applied multi-class segmentation (softmax) for {num_output_channels} classes")
        
        # Convert to numpy for post-processing
        if num_output_channels == 1:
            # Binary segmentation - output is already thresholded
            output_array = output.squeeze().cpu().numpy()
        else:
            # Multi-class - use argmax to get final predictions
            output_array = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        # Apply basic post-processing (simplified from enhanced version)
        try:
            from ml.utils.post_processing import enhanced_post_processing
            
            # Use simpler config for post-processing
            simple_config = {
                'threshold': post_processing_config.get('threshold', 0.5),
                'min_component_size': post_processing_config.get('min_component_size', 50),
                'apply_morphology': post_processing_config.get('apply_opening', True)
            }
            
            post_processed_result = enhanced_post_processing(
                output_array,
                model_type='segmentation',
                **simple_config
            )
            
            processed_mask = post_processed_result['processed_output']
            processing_stats = post_processed_result['stats']
            
            print(f"Post-processing applied: {processing_stats.get('components_after_filtering', 0)} components")
            
            # Convert to 8-bit image
            output_image = (processed_mask * 255).astype('uint8')
            
        except ImportError:
            # Fallback to basic processing
            print("Enhanced post-processing not available, using basic processing")
            output_image = output_array
            
            # Normalize to 0-255 range
            if output_image.max() <= 1.0:
                output_image = (output_image * 255).astype('uint8')
            else:
                output_image = output_image.astype('uint8')
            processing_stats = {}
        
        print("Output min:", output_image.min())
        print("Output max:", output_image.max())
        print("Output mean:", output_image.mean())

        return {
            'prediction_image': output_image,
            'processing_stats': processing_stats,
            'type': 'segmentation'
        }

def run_prediction(model_path, image_path, model_type='segmentation', post_processing_config=None):
    """
    Enhanced API function for running predictions from Django views.
    
    Args:
        model_path: Path to the trained model
        image_path: Path to the input image
        model_type: 'segmentation' or 'classification'
        post_processing_config: Configuration for post-processing
    """
    try:
        print(f"Running enhanced prediction: model_path={model_path}, image_path={image_path}, type={model_type}")
        
        # Load model with type detection
        model, detected_type = load_model(model_path, model_type)
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Run inference with enhanced post-processing
        result = inference(model, image, detected_type, post_processing_config)
        
        if detected_type == 'classification':
            # For classification, return structured results
            return {
                'success': True,
                'type': 'classification',
                'predicted_class': result['predicted_class'],
                'confidence_score': result['confidence'],
                'probabilities': result['probabilities'].tolist(),
                'prediction_image': None  # No image output for classification
            }
        else:
            # For segmentation, return image with processing stats
            return {
                'success': True,
                'type': 'segmentation',
                'prediction_image': result['prediction_image'],
                'processing_stats': result.get('processing_stats', {}),
                'confidence_score': 1.0  # Placeholder for segmentation
            }
            
    except Exception as e:
        print(f"Error in run_prediction: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description='Enhanced inference script with post-processing')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model_dir', type=str, default='data/models', help='Directory containing the trained models')
    parser.add_argument('--epoch', type=int, required=True, help='Epoch number of the model to load')
    parser.add_argument('--model_type', type=str, default='segmentation', choices=['segmentation', 'classification'], help='Type of model')
    
    # Post-processing arguments
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for segmentation')
    parser.add_argument('--min_component_size', type=int, default=50, help='Minimum component size to keep')
    parser.add_argument('--morphology_kernel_size', type=int, default=3, help='Morphological operations kernel size')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='Confidence threshold for post-processing')
    parser.add_argument('--no_opening', action='store_true', help='Skip morphological opening')
    parser.add_argument('--no_closing', action='store_true', help='Skip morphological closing')
    
    args = parser.parse_args()

    # Znajdź odpowiedni plik modelu na podstawie numeru epoki
    model_path = find_model_file(args.model_dir, args.epoch)
    print(f"Loading model from {model_path}")

    # Prepare post-processing configuration
    post_processing_config = {
        'threshold': args.threshold,
        'min_component_size': args.min_component_size,
        'morphology_kernel_size': args.morphology_kernel_size,
        'confidence_threshold': args.confidence_threshold,
        'apply_opening': not args.no_opening,
        'apply_closing': not args.no_closing
    }

    # Load model
    model, detected_type = load_model(model_path, args.model_type)

    # Load image
    image = Image.open(args.image_path).convert('RGB')

    # Run enhanced inference
    result = inference(model, image, detected_type, post_processing_config)
    
    if detected_type == 'classification':
        print(f"Classification result: {result}")
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.3f}")
    else:
        # Visualization for segmentation
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Enhanced Model Output")
        plt.imshow(result['prediction_image'], cmap='gray')
        plt.axis('off')
        
        # Show processing statistics
        stats = result.get('processing_stats', {})
        if stats:
            plt.subplot(1, 3, 3)
            plt.axis('off')
            plt.title("Processing Stats")
            stats_text = []
            for key, value in stats.items():
                if isinstance(value, float):
                    stats_text.append(f"{key}: {value:.3f}")
                else:
                    stats_text.append(f"{key}: {value}")
            plt.text(0.1, 0.5, '\n'.join(stats_text), transform=plt.gca().transAxes, 
                    verticalalignment='center', fontsize=10)

        plt.tight_layout()
        plt.show()
        
        print(f"Processing stats: {stats}")

if __name__ == '__main__':
    main()