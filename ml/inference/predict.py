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
    checkpoint = torch.load(model_path, weights_only=True, map_location='cpu')
    
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
        
        # Try direct loading first
        model.load_state_dict(state_dict, strict=True)
        print("✅ Model loaded successfully with strict=True")
        
    except RuntimeError as e:
        if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
            print(f"⚠️ State dict mismatch detected: {e}")
            print("🔧 Attempting to fix parameter name mismatches...")
            
            # Try loading with strict=False to allow missing keys
            try:
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    print(f"⚠️ Missing keys in model: {missing_keys[:5]}...")  # Show first 5
                if unexpected_keys:
                    print(f"⚠️ Unexpected keys in state_dict: {unexpected_keys[:5]}...")  # Show first 5
                
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

def inference(model, image, model_type='segmentation'):
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
        # Segmentation output processing
        num_output_channels = output.shape[1]
        if num_output_channels == 1:
            # Binary segmentation
            output = torch.sigmoid(output)  # Apply sigmoid for binary segmentation
            print(f"Applied binary segmentation post-processing (sigmoid)")
        else:
            # Multi-class semantic segmentation
            output = torch.softmax(output, dim=1)  # Apply softmax for multi-class
            output = torch.argmax(output, dim=1, keepdim=True).float()  # Get class predictions
            print(f"Applied multi-class segmentation post-processing (softmax + argmax) for {num_output_channels} classes")
        
        output_image = output.squeeze().cpu().numpy()  # Convert back to numpy for visualization
        output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min()) * 255
        output_image = output_image.astype('uint8')
        print("Output min:", output_image.min())
        print("Output max:", output_image.max())
        print("Output mean:", output_image.mean())

        return output_image

def run_prediction(model_path, image_path, model_type='segmentation'):
    """
    API function for running predictions from Django views
    """
    try:
        print(f"Running prediction: model_path={model_path}, image_path={image_path}, type={model_type}")
        
        # Load model with type detection
        model, detected_type = load_model(model_path, model_type)
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Run inference
        result = inference(model, image, detected_type)
        
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
            # For segmentation, return image
            return {
                'success': True,
                'type': 'segmentation',
                'prediction_image': result,
                'confidence_score': 1.0  # Placeholder for segmentation
            }
            
    except Exception as e:
        print(f"Error in run_prediction: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model_dir', type=str, default='data/models', help='Directory containing the trained models')
    parser.add_argument('--epoch', type=int, required=True, help='Epoch number of the model to load')
    parser.add_argument('--model_type', type=str, default='segmentation', choices=['segmentation', 'classification'], help='Type of model')
    args = parser.parse_args()

    # Znajdź odpowiedni plik modelu na podstawie numeru epoki
    model_path = find_model_file(args.model_dir, args.epoch)
    print(f"Loading model from {model_path}")

    # Załaduj model
    model, detected_type = load_model(model_path, args.model_type)

    # Wczytaj obraz
    image = Image.open(args.image_path).convert('RGB')

    # Wykonaj inferencję
    result = inference(model, image, detected_type)
    
    if detected_type == 'classification':
        print(f"Classification result: {result}")
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.3f}")
    else:
        # Wizualizacja wyników dla segmentacji
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title("Input Image")
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Model Output")
        plt.imshow(result, cmap='gray')
        plt.axis('off')

        plt.show()

if __name__ == '__main__':
    main()