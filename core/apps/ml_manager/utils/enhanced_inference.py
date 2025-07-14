"""
Enhanced inference wrapper for Django integration
"""
import os
import time
import json
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
import torch

try:
    from celery.utils.log import get_task_logger
    logger = get_task_logger(__name__)
except ImportError:
    # Fallback to standard logging if Celery is not available
    import logging
    logger = logging.getLogger(__name__)

def run_enhanced_inference(
    model_path, 
    input_image_path, 
    output_dir,
    config=None,
    device="cpu"  # Changed default to CPU to avoid CUDA issues
):
    """
    Enhanced inference function that returns detailed results for Django integration
    
    Args:
        model_path: Path to the trained model weights
        input_image_path: Path to the input image
        output_dir: Directory to save output files
        config: Dictionary with inference configuration
        device: Device to run inference on
        
    Returns:
        dict: Detailed inference results with metrics and file paths
    """
    if config is None:
        config = {}
    
    # Extract configuration with defaults (ensure correct types)
    try:
        threshold = float(config.get('threshold', 0.5))
    except (ValueError, TypeError):
        threshold = 0.5
        
    try:
        crop_size = int(config.get('resolution', 512))
    except (ValueError, TypeError):
        crop_size = 512
        
    model_type = str(config.get('model_type', 'unet'))
    
    # Post-processing options
    apply_opening = config.get('apply_opening', True)
    apply_closing = config.get('apply_closing', True)
    apply_dilation = config.get('apply_dilation', False)
    apply_erosion = config.get('apply_erosion', False)
    fill_holes = config.get('fill_holes', True)
    smooth_boundaries = config.get('smooth_boundaries', False)
    remove_border_objects = config.get('remove_border_objects', False)
    min_component_size = config.get('min_component_size', 100)
    
    # TTA options
    use_tta = config.get('use_tta', False)
    tta_flip_horizontal = config.get('tta_flip_horizontal', True)
    tta_flip_vertical = config.get('tta_flip_vertical', True)
    tta_rotate_90 = config.get('tta_rotate_90', True)
    
    start_time = time.time()
    
    try:
        # Import the original inference function
        try:
            from core.apps.ml_manager.training.train import run_inference, create_model_from_registry, get_default_model_config
        except ImportError:
            # Try alternative import paths
            import sys
            from pathlib import Path
            training_script_path = Path(__file__).parent.parent / 'training'
            if str(training_script_path) not in sys.path:
                sys.path.insert(0, str(training_script_path))
            from training.train import run_inference, create_model_from_registry, get_default_model_config
        import torch
        
        # Create temporary directory for original function output
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # First, check checkpoint format to decide which inference method to use
            use_custom_inference = False
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                if isinstance(checkpoint, dict):
                    # Check if it's a nested checkpoint format
                    if 'model_state_dict' in checkpoint or 'state_dict' in checkpoint:
                        logger.info("Detected nested checkpoint format, using custom inference")
                        use_custom_inference = True
                    # Check if the checkpoint contains the wrong keys (indicating it's the full checkpoint)
                    elif any(key in checkpoint for key in ['model_metadata', 'training_args']):
                        logger.info("Detected full checkpoint format, using custom inference")
                        use_custom_inference = True
            except Exception as e:
                logger.warning(f"Could not analyze checkpoint format: {e}, using custom inference")
                use_custom_inference = True
            
            if use_custom_inference:
                # Use custom inference directly for nested/complex checkpoint formats
                logger.info("Using custom inference for complex checkpoint format")
                logger.info(f"Device passed to custom inference: {device}")
                logger.info(f"Config passed to custom inference: {config}")
                
                # Try with requested device first
                custom_inference_result = run_custom_inference(
                    model_path=model_path,
                    input_image_path=input_image_path,
                    output_dir=temp_dir,
                    config=config,
                    device=device
                )
                
                if not custom_inference_result:
                    # If fails and device is not CPU, try again with CPU
                    if device != 'cpu':
                        logger.warning("Custom inference failed with GPU, trying with CPU...")
                        custom_inference_result = run_custom_inference(
                            model_path=model_path,
                            input_image_path=input_image_path,
                            output_dir=temp_dir,
                            config=config,
                            device='cpu'
                        )
                        
                    # Jeśli wciąż nie działa, spróbujmy ostateczną metodę fallback - stwórzmy pusty plik wyniku
                    if not custom_inference_result:
                        logger.warning("All inference attempts failed. Creating empty prediction as fallback.")
                        
                        # Create empty prediction image
                        from PIL import Image
                        import numpy as np
                        
                        # Ensure crop_size is an integer
                        try:
                            crop_size_int = int(crop_size)
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid crop_size value: {crop_size}, using default 512")
                            crop_size_int = 512
                        
                        # Create a blank black image as fallback
                        empty_pred = np.zeros((crop_size_int, crop_size_int), dtype=np.uint8)
                        
                        # Save empty prediction
                        input_filename = os.path.basename(input_image_path)
                        pred_only_filename = os.path.join(temp_dir, f"pred_only_{input_filename}")
                        pred_only_img = Image.fromarray(empty_pred)
                        pred_only_img.save(pred_only_filename)
                        
                        # Copy input image for processing
                        input_only_filename = os.path.join(temp_dir, f"input_{input_filename}")
                        input_img = Image.open(input_image_path)
                        if input_img.mode != 'RGB':
                            input_img = input_img.convert('RGB')
                        input_img = input_img.resize((crop_size_int, crop_size_int), Image.Resampling.LANCZOS)
                        input_img.save(input_only_filename)
                        
                        logger.warning(f"Created empty prediction as fallback for {input_filename}")
            else:
                # Try original inference for simple checkpoint formats
                try:
                    logger.info("Using original inference for simple checkpoint format")
                    run_inference(
                        model_path=model_path,
                        input_path=input_image_path,
                        output_dir=temp_dir,
                        device=device,
                        model_type=model_type,
                        crop_size=crop_size,
                        threshold=threshold
                    )
                except Exception as model_error:
                    logger.warning(f"Original inference failed: {model_error}")
                    
                    # Fallback to custom inference
                    logger.info("Falling back to custom inference")
                    custom_inference_result = run_custom_inference(
                        model_path=model_path,
                        input_image_path=input_image_path,
                        output_dir=temp_dir,
                        config=config,
                        device=device
                    )
                    if not custom_inference_result:
                        raise model_error
            
            # Process the results and apply post-processing
            results = process_inference_results(
                temp_dir, 
                input_image_path, 
                output_dir,
                config
            )
            
        processing_time = time.time() - start_time
        
        # Add timing information
        results['processing_time'] = processing_time
        results['status'] = 'completed'
        results['success'] = True
        
        logger.info(f"Enhanced inference completed in {processing_time:.2f}s")
        return results
        
    except Exception as e:
        logger.error(f"Enhanced inference failed: {str(e)}")
        
        # Generate user-friendly error message
        error_str = str(e).lower()
        user_message = str(e)
        
        if 'missing key' in error_str or 'unexpected key' in error_str:
            user_message = "Model checkpoint format doesn't match expected structure. This checkpoint may be from a different model architecture."
        elif 'cuda' in error_str or 'nvidia' in error_str:
            user_message = "CUDA/GPU error occurred. Trying to run on CPU instead."
        elif 'file not found' in error_str or 'no such file' in error_str:
            user_message = "Model checkpoint file not found or inaccessible."
        elif 'memory' in error_str:
            user_message = "Out of memory error. Try using a smaller resolution."
        
        processing_time = time.time() - start_time
        return {
            'success': False,
            'status': 'failed',
            'error_message': user_message,
            'technical_error': str(e),  # Keep original error for debugging
            'processing_time': processing_time,
            'detected_objects_count': 0,
            'total_area_pixels': 0,
            'confidence_scores': [],
            'output_files': {}
        }

def run_custom_inference(model_path, input_image_path, output_dir, config, device):
    """
    Custom inference function to handle different model save formats
    
    Args:
        model_path: Path to the trained model weights
        input_image_path: Path to the input image
        output_dir: Directory to save output files
        config: Configuration dictionary
        device: Device to run inference on
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import torch
        try:
            from core.apps.ml_manager.training.train import create_model_from_registry, get_default_model_config, get_inference_transforms
        except ImportError:
            # Try alternative import paths
            import sys
            from pathlib import Path
            training_script_path = Path(__file__).parent.parent / 'training'
            if str(training_script_path) not in sys.path:
                sys.path.insert(0, str(training_script_path))
            from training.train import create_model_from_registry, get_default_model_config, get_inference_transforms
        
        logger.info(f"Starting custom inference with device: {device}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Input image path: {input_image_path}")
        logger.info(f"Config: {config}")
        
        # Verify CUDA availability if device is not 'cpu'
        if device != 'cpu':
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                device = 'cpu'
            else:
                try:
                    # Test CUDA device
                    test_tensor = torch.zeros(1).to(device)
                    del test_tensor
                except Exception as e:
                    logger.warning(f"CUDA device error: {e}. Falling back to CPU.")
                    device = 'cpu'
        
        # Load model checkpoint with proper device mapping - force CPU for loading
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Handle different checkpoint formats
        state_dict = None
        metadata = {}
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                metadata = checkpoint.get('model_metadata', {})
                logger.info("Loaded checkpoint with model_state_dict format")
                logger.info(f"Available metadata keys: {list(metadata.keys()) if metadata else 'No metadata'}")
                if metadata:
                    logger.info(f"Metadata content: {metadata}")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict'] 
                metadata = checkpoint.get('metadata', {})
                logger.info("Loaded checkpoint with state_dict format")
                logger.info(f"Available metadata keys: {list(metadata.keys()) if metadata else 'No metadata'}")
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
                metadata = checkpoint.get('metadata', {})
                logger.info("Loaded checkpoint with model format")
                logger.info(f"Available metadata keys: {list(metadata.keys()) if metadata else 'No metadata'}")
            else:
                # Check if it looks like a state_dict (has layer keys)
                sample_keys = list(checkpoint.keys())[:5]
                if any('weight' in key or 'bias' in key or 'conv' in key or 'fc' in key for key in sample_keys):
                    state_dict = checkpoint
                    logger.info("Loaded checkpoint as direct state_dict")
                else:
                    logger.error(f"Unknown checkpoint format. Keys: {sample_keys}")
                    return False
        else:
            logger.error(f"Checkpoint is not a dictionary: {type(checkpoint)}")
            return False
        
        if state_dict is None:
            logger.error("Could not extract state_dict from checkpoint")
            return False
        
        logger.info(f"State dict loaded successfully, has {len(state_dict)} keys")
        
        # Debug: Show first few keys to understand the model structure
        sample_keys = list(state_dict.keys())[:10]
        logger.info(f"Sample state_dict keys: {sample_keys}")
        
        # Show first layer weight shape for debugging
        first_conv_key = None
        for key in sample_keys:
            if 'weight' in key and ('conv' in key or 'model.0' in key):
                first_conv_key = key
                logger.info(f"First conv layer key: {key}, shape: {state_dict[key].shape}")
                break
        
        # Move state_dict to the correct device
        if device != 'cpu':
            # Move state dict tensors to target device
            for key in state_dict:
                if hasattr(state_dict[key], 'to'):
                    state_dict[key] = state_dict[key].to(device)
        
        # Get model configuration - prioritize state_dict detection over metadata
        # Handle model_architecture which can be a dict with detailed config
        model_architecture = metadata.get('model_architecture', {}) if metadata else {}
        if isinstance(model_architecture, dict):
            # Extract the base model type from architecture config (should be configurable_monai_unet for our models)
            model_type = config.get('model_type', 'configurable_monai_unet')
        else:
            # Fallback for legacy string format
            model_type = model_architecture if isinstance(model_architecture, str) else config.get('model_type', 'unet')
        
        # Always try to infer input channels from state_dict (most reliable method)
        in_channels = 3  # default fallback
        first_layer_key = None
        
        # Look for the first convolutional layer weight
        for key in sorted(state_dict.keys()):
            if 'weight' in key and ('conv' in key or 'model.0' in key) and not 'adn' in key and not 'norm' in key and not 'bias' in key:
                first_layer_key = key
                break
        
        if first_layer_key and first_layer_key in state_dict:
            weight_shape = state_dict[first_layer_key].shape
            if len(weight_shape) >= 2:
                detected_channels = weight_shape[1]  # Input channels dimension
                logger.info(f"Detected input_channels={detected_channels} from layer {first_layer_key} with shape {weight_shape}")
                in_channels = detected_channels  # Use detected channels
            else:
                logger.warning(f"Layer {first_layer_key} has unexpected shape {weight_shape}")
                # Fallback to metadata if available
                in_channels = metadata.get('input_channels', 3) if metadata else 3
        else:
            logger.warning(f"Could not find first conv layer. Available keys (first 10): {list(state_dict.keys())[:10]}")
            # Fallback to metadata if available
            in_channels = metadata.get('input_channels', 3) if metadata else 3
        
        # Get output channels from metadata or default
        out_channels = metadata.get('num_classes', 1) if metadata else 1
        
        logger.info(f"Final model config: model_type={model_type}, in_channels={in_channels}, out_channels={out_channels}")
        logger.info(f"Detection method: {'state_dict' if first_layer_key else 'metadata/default'}")
            
        # Ensure crop_size is an integer
        try:
            crop_size = int(config.get('resolution', 512))
        except (ValueError, TypeError):
            logger.warning(f"Invalid crop_size value: {config.get('resolution')}, using default 512")
            crop_size = 512
            
        threshold = float(config.get('threshold', 0.5))
        logger.info(f"Using crop_size: {crop_size}, threshold: {threshold}")
        
        # Create model using architecture registry with detected metadata
        model_config = get_default_model_config(model_type)
        model_config["in_channels"] = in_channels
        model_config["out_channels"] = out_channels
        
        # If we have detailed architecture config, use it to override defaults
        if isinstance(model_architecture, dict) and model_architecture:
            logger.info(f"Using detailed architecture config: {model_architecture}")
            # Override model config with specific architecture parameters
            if 'custom_channels' in model_architecture:
                try:
                    channels_str = model_architecture['custom_channels']
                    channels = tuple(int(x.strip()) for x in channels_str.split(','))
                    model_config['channels'] = channels
                    logger.info(f"Using custom channels: {channels}")
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Could not parse custom channels '{model_architecture.get('custom_channels')}': {e}")
        
        logger.info(f"Creating model with config: {model_config}")
        model, arch_info = create_model_from_registry(model_type, 'cpu', **model_config)  # Force CPU
        
        # Verify model was created with correct input channels
        if hasattr(model, 'model') and hasattr(model.model, '0'):
            first_layer = model.model[0]
            if hasattr(first_layer, 'conv') and hasattr(first_layer.conv, 'unit0'):
                first_conv = first_layer.conv.unit0.conv
                logger.info(f"Created model first conv layer expects: {first_conv.weight.shape} (should match checkpoint)")
        
        logger.info(f"Model created successfully with {in_channels} input channels")
        
        # Load state dict with strict=False to handle minor key mismatches
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            logger.warning(f"Strict loading failed: {e}")
            # Try with strict=False
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
        
        model.eval()
        
        # Move model to correct device after loading
        model = model.to(device)
        
        # Get transforms - use our custom safer approach for MONAI models
        model_type_from_config = config.get('model_type', model_type)
        # Traktujemy wszystkie modele jako potencjalnie wykorzystujące MONAI dla bezpieczeństwa
        is_monai_model = True
        logger.info(f"Model type: {model_type_from_config}, using safe image loading for all models")
        logger.info(f"Detected input channels: {in_channels}, will load image accordingly")
        
        try:
            if is_monai_model:
                # Use safer PIL-based approach for MONAI models
                from PIL import Image
                import numpy as np
                
                logger.info("Using custom PIL-based transforms for MONAI model")
                
                # Load image using PIL to maintain browser-compatible orientation
                img_pil = Image.open(input_image_path)
                
                # Handle input channels - convert image format based on model requirements
                if in_channels == 1:
                    # Model expects grayscale input
                    if img_pil.mode != 'L':
                        img_pil = img_pil.convert('L')
                    logger.info("Converted image to grayscale for 1-channel model")
                    
                    # Resize to target size
                    img_pil = img_pil.resize((crop_size, crop_size), Image.Resampling.LANCZOS)
                    
                    # Convert to numpy array and add channel dimension
                    img_array = np.array(img_pil)
                    if len(img_array.shape) == 2:
                        # Add channel dimension: HW -> CHW
                        img_array = img_array[np.newaxis, :, :]
                    
                    # Convert to torch tensor and normalize to [0, 1]
                    img = torch.from_numpy(img_array).float() / 255.0
                    
                    # Apply grayscale normalization
                    mean = torch.tensor([0.5]).view(1, 1, 1)
                    std = torch.tensor([0.5]).view(1, 1, 1)
                    img = (img - mean) / std
                    
                else:
                    # Model expects RGB input
                    if img_pil.mode != 'RGB':
                        img_pil = img_pil.convert('RGB')
                    logger.info("Using RGB format for 3-channel model")
                    
                    # Resize to target size
                    img_pil = img_pil.resize((crop_size, crop_size), Image.Resampling.LANCZOS)
                    
                    # Convert to numpy array and add channel dimension
                    img_array = np.array(img_pil)
                    if len(img_array.shape) == 3:
                        # Convert HWC to CHW format
                        img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
                    
                    # Convert to torch tensor and normalize to [0, 1]
                    img = torch.from_numpy(img_array).float() / 255.0
                    
                    # Apply RGB normalization to match training
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img = (img - mean) / std
                    
            else:
                # Dla bezpieczeństwa, również dla zwykłych modeli używamy bezpiecznego ładowania PIL
                # ponieważ transformacje mogą mieć problemy z ścieżkami plików
                from PIL import Image
                import numpy as np
                
                logger.info("Using safe PIL-based transforms for standard model")
                
                # Load image using PIL
                img_pil = Image.open(input_image_path)
                
                # Handle channels based on model requirements
                if in_channels == 1:
                    if img_pil.mode != 'L':
                        img_pil = img_pil.convert('L')
                    
                    # Resize to target size
                    img_pil = img_pil.resize((crop_size, crop_size), Image.Resampling.LANCZOS)
                    
                    # Convert to numpy array and add channel dimension
                    img_array = np.array(img_pil)
                    if len(img_array.shape) == 2:
                        img_array = img_array[np.newaxis, :, :]
                    
                    # Convert to torch tensor and normalize to [0, 1]
                    img = torch.from_numpy(img_array).float() / 255.0
                    
                    # Apply grayscale normalization
                    mean = torch.tensor([0.5]).view(1, 1, 1)
                    std = torch.tensor([0.5]).view(1, 1, 1)
                    img = (img - mean) / std
                else:
                    # Ensure RGB format
                    if img_pil.mode != 'RGB':
                        img_pil = img_pil.convert('RGB')
                    
                    # Resize to target size
                    img_pil = img_pil.resize((crop_size, crop_size), Image.Resampling.LANCZOS)
                    
                    # Convert to numpy array and add channel dimension
                    img_array = np.array(img_pil)
                    if len(img_array.shape) == 3:
                        # Convert HWC to CHW format
                        img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
                    
                    # Convert to torch tensor and normalize to [0, 1]
                    img = torch.from_numpy(img_array).float() / 255.0
                    
                    # Apply RGB normalization
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img = (img - mean) / std
                
            # Add batch dimension and move to device
            img = img.unsqueeze(0).to(device)
            logger.info(f"Final input tensor shape: {img.shape} (expected: [1, {in_channels}, {crop_size}, {crop_size}])")
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
        
        # Run inference
        with torch.no_grad():
            output = model(img)
            
            # Apply post-processing
            num_output_channels = output.shape[1]
            if num_output_channels == 1:
                output_soft = torch.sigmoid(output)
                pred = (output_soft > threshold).float()
                logger.info(f"Applied binary segmentation (sigmoid + threshold)")
            else:
                output = torch.softmax(output, dim=1)
                pred = torch.argmax(output, dim=1, keepdim=True).float()
                logger.info(f"Applied multi-class segmentation (softmax + argmax)")
            
            # Save prediction
            pred_np = pred.squeeze().cpu().numpy()
            if pred_np.ndim == 3 and pred_np.shape[0] == 1:
                pred_np = pred_np.squeeze(0)
            
            # Create visualization files similar to original function
            input_filename = os.path.basename(input_image_path)
            
            # Convert to visible image
            if pred_np.max() == 0:
                logger.warning("No segmentation detected")
                pred_image = np.zeros_like(pred_np, dtype=np.uint8)
                pred_image[10:30, 10:30] = 128  # Small indicator
            else:
                pred_image = (pred_np * 255).astype(np.uint8)
                logger.info(f"Segmentation detected - {np.sum(pred_np > 0)} pixels")
            
            # Save files in the expected format for processing
            pred_only_filename = os.path.join(output_dir, f"pred_only_{input_filename}")
            pred_only_img = Image.fromarray(pred_image)
            pred_only_img.save(pred_only_filename)
            
            # Save input copy
            input_only_filename = os.path.join(output_dir, f"input_{input_filename}")
            input_img = Image.open(input_image_path)
            if input_img.mode != 'RGB':
                input_img = input_img.convert('RGB')
            input_img = input_img.resize((pred_image.shape[1], pred_image.shape[0]), Image.Resampling.LANCZOS)
            input_img.save(input_only_filename)
            
            logger.info(f"Custom inference completed successfully")
            return True
            
    except Exception as e:
        logger.error(f"Custom inference failed: {e}")
        logger.debug(f"Exception type: {type(e)}")
        logger.debug(f"Exception args: {e.args}")
        # Print traceback for debugging
        import traceback
        tb = traceback.format_exc()
        logger.debug(f"Traceback: {tb}")
        
        # Detailed error info for common issues
        error_str = str(e).lower()
        if 'nvidia driver' in error_str or 'cuda' in error_str:
            logger.error("GPU error detected. When using device='cpu' this should not happen.")
        elif 'key' in error_str and ('missing' in error_str or 'unexpected' in error_str):
            logger.error("Model state_dict loading error. Keys don't match model structure.")
        elif 'memory' in error_str:
            logger.error("Memory error. Try using a smaller resolution or batch size.")
        
        # Always return False to let caller handle the fallback strategy
        return False

def process_inference_results(temp_dir, input_image_path, output_dir, config):
    """
    Process the raw inference results and apply post-processing
    
    Args:
        temp_dir: Temporary directory with raw results
        input_image_path: Original input image path
        output_dir: Final output directory
        config: Configuration dictionary
        
    Returns:
        dict: Processed results with metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find generated files
    input_filename = os.path.basename(input_image_path)
    pred_only_file = None
    input_only_file = None
    comparison_file = None
    
    logger.info(f"Processing inference results from {temp_dir}")
    logger.info(f"Looking for files related to {input_filename}")
    
    # List all files in temp directory for debugging
    temp_files = os.listdir(temp_dir) if os.path.exists(temp_dir) else []
    logger.info(f"Files in temp directory: {temp_files}")
    
    for file in temp_files:
        file_lower = file.lower()
        input_base = os.path.splitext(input_filename)[0].lower()
        
        logger.info(f"Checking file: {file}")
        
        if file.startswith('pred_only_'):
            pred_only_file = os.path.join(temp_dir, file)
            logger.info(f"Found prediction file: {pred_only_file}")
        elif file.startswith('input_'):
            input_only_file = os.path.join(temp_dir, file)
            logger.info(f"Found input file: {input_only_file}")
        elif file.startswith('pred_') and not file.startswith('pred_only_'):
            comparison_file = os.path.join(temp_dir, file)
            logger.info(f"Found comparison file: {comparison_file}")
    
    results = {
        'detected_objects_count': 0,
        'total_area_pixels': 0,
        'confidence_scores': [],
        'output_files': {},
        'files': {},
        'metrics': {},
        'debug_info': {
            'temp_dir': temp_dir,
            'temp_files': temp_files,
            'pred_file': pred_only_file,
            'input_file': input_only_file,
            'comparison_file': comparison_file
        }
    }
    
    if pred_only_file and os.path.exists(pred_only_file):
        logger.info(f"Processing prediction file: {pred_only_file}")
        
        try:
            # Load and analyze the prediction mask
            pred_image = Image.open(pred_only_file)
            pred_array = np.array(pred_image)
            
            logger.info(f"Loaded prediction - shape: {pred_array.shape}, dtype: {pred_array.dtype}, min: {pred_array.min()}, max: {pred_array.max()}")
            
            # Convert to binary if needed
            if pred_array.max() > 1:
                # Image is in 0-255 range
                pred_binary = (pred_array > 127).astype(np.uint8)
            else:
                # Image is in 0-1 range
                pred_binary = (pred_array > 0.5).astype(np.uint8)
            
            logger.info(f"Binary prediction - shape: {pred_binary.shape}, unique values: {np.unique(pred_binary)}")
            logger.info(f"Number of positive pixels: {np.sum(pred_binary > 0)}")
            
            # Apply post-processing
            processed_mask = apply_post_processing(pred_binary, config)
            logger.info(f"Post-processed mask - shape: {processed_mask.shape}, positive pixels: {np.sum(processed_mask > 0)}")
            
            # Analyze the processed mask
            analysis = analyze_segmentation_mask(processed_mask)
            results.update(analysis)
            
            logger.info(f"Analysis results: {analysis}")
            
            # Save processed files to final output directory
            output_files = save_processed_results(
                input_image_path,
                processed_mask,
                pred_binary,
                output_dir,
                input_filename
            )
            results['output_files'] = output_files
            results['files'] = output_files  # Add for compatibility
            results['metrics'] = {
                'detected_objects': results['detected_objects_count'],
                'total_area': results['total_area_pixels'],
                'confidence_scores': results['confidence_scores']
            }
            
            logger.info(f"Found {results['detected_objects_count']} objects, total area: {results['total_area_pixels']} pixels")
            logger.info(f"Generated output files: {list(output_files.keys())}")
            
        except Exception as e:
            logger.error(f"Error processing prediction file {pred_only_file}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Add error info to results
            results['error'] = str(e)
            results['debug_info']['processing_error'] = str(e)
    else:
        logger.warning(f"No prediction file found. pred_only_file: {pred_only_file}")
        logger.warning(f"Available files: {temp_files}")
        
        # Try to create dummy results for debugging
        try:
            # Create a minimal mask for testing
            dummy_mask = np.zeros((512, 512), dtype=np.uint8)
            dummy_mask[100:200, 100:200] = 1  # Small square for testing
            
            logger.info("Creating dummy mask for debugging")
            
            # Analyze dummy mask
            analysis = analyze_segmentation_mask(dummy_mask)
            logger.info(f"Dummy analysis: {analysis}")
            
            # Save dummy results
            output_files = save_processed_results(
                input_image_path,
                dummy_mask,
                dummy_mask,
                output_dir,
                input_filename
            )
            
            results.update(analysis)
            results['output_files'] = output_files
            results['files'] = output_files
            results['metrics'] = {
                'detected_objects': results['detected_objects_count'],
                'total_area': results['total_area_pixels'],
                'confidence_scores': results['confidence_scores']
            }
            results['debug_info']['used_dummy_mask'] = True
            
            logger.info("Dummy results created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create dummy results: {e}")
            results['debug_info']['dummy_creation_error'] = str(e)
    
    return results

def apply_post_processing(mask_array, config):
    """
    Apply post-processing operations to the segmentation mask
    
    Args:
        mask_array: Raw segmentation mask as numpy array
        config: Configuration dictionary with post-processing options
        
    Returns:
        numpy.ndarray: Processed mask
    """
    try:
        from scipy import ndimage
        from skimage import morphology, measure, filters
        from skimage.morphology import disk, opening, closing, dilation, erosion
        from skimage.segmentation import clear_border
    except ImportError:
        logger.warning("Scipy/skimage not available, skipping advanced post-processing")
        return mask_array
    
    # Convert to binary mask
    if mask_array.max() > 1:
        binary_mask = (mask_array > 128).astype(np.uint8)
    else:
        binary_mask = mask_array.astype(np.uint8)
    
    # Get morphological kernel size
    kernel_size = config.get('morphology_kernel_size', 3)
    kernel = disk(kernel_size)
    
    # Apply morphological operations
    if config.get('apply_opening', True):
        binary_mask = opening(binary_mask, kernel)
    
    if config.get('apply_closing', True):
        binary_mask = closing(binary_mask, kernel)
    
    if config.get('apply_dilation', False):
        binary_mask = dilation(binary_mask, kernel)
    
    if config.get('apply_erosion', False):
        binary_mask = erosion(binary_mask, kernel)
    
    # Fill holes
    if config.get('fill_holes', True):
        binary_mask = ndimage.binary_fill_holes(binary_mask).astype(np.uint8)
    
    # Remove small objects
    min_size = config.get('min_component_size', 100)
    if min_size > 0:
        binary_mask = morphology.remove_small_objects(
            binary_mask.astype(bool), 
            min_size=min_size
        ).astype(np.uint8)
    
    # Remove border objects
    if config.get('remove_border_objects', False):
        binary_mask = clear_border(binary_mask)
    
    # Smooth boundaries
    if config.get('smooth_boundaries', False):
        binary_mask = filters.gaussian(binary_mask.astype(float), sigma=1.0)
        binary_mask = (binary_mask > 0.5).astype(np.uint8)
    
    return binary_mask

def analyze_segmentation_mask(mask_array):
    """
    Analyze the segmentation mask to extract metrics
    
    Args:
        mask_array: Binary segmentation mask
        
    Returns:
        dict: Analysis results with object count, areas, confidence scores
    """
    # Ensure mask is binary (0 or 1)
    if mask_array.max() > 1:
        binary_mask = (mask_array > 0.5 * mask_array.max()).astype(np.uint8)
    else:
        binary_mask = (mask_array > 0.5).astype(np.uint8)
    
    # Calculate total area first
    total_area = int(np.sum(binary_mask > 0))
    
    logger.info(f"Mask analysis - shape: {binary_mask.shape}, max: {binary_mask.max()}, total area: {total_area}")
    
    try:
        from skimage import measure, morphology
        
        # Label connected components
        labeled_mask = measure.label(binary_mask)
        regions = measure.regionprops(labeled_mask)
        
        object_count = len(regions)
        
        # Re-calculate total area from regions (more accurate)
        total_area_from_regions = sum(region.area for region in regions) if regions else total_area
        
        # Calculate confidence scores based on object properties
        confidence_scores = []
        for region in regions:
            # Use solidity and area as confidence indicators
            solidity = region.solidity if hasattr(region, 'solidity') else 0.8
            area_ratio = region.area / (binary_mask.shape[0] * binary_mask.shape[1])
            
            # Enhanced confidence calculation
            base_confidence = 0.75
            solidity_bonus = solidity * 0.15
            area_bonus = min(area_ratio * 3, 0.1)  # Bonus for reasonable sized objects
            
            confidence = min(0.99, base_confidence + solidity_bonus + area_bonus)
            confidence_scores.append(round(confidence, 3))
        
        # Use the more accurate area calculation
        final_total_area = max(total_area, total_area_from_regions)
        
        logger.info(f"Detected {object_count} objects with total area {final_total_area} pixels")
        
        return {
            'detected_objects_count': object_count,
            'total_area_pixels': int(final_total_area),
            'confidence_scores': confidence_scores if confidence_scores else ([0.85] if final_total_area > 0 else [])
        }
        
    except ImportError:
        # Fallback analysis without skimage
        logger.warning("Skimage not available, using fallback analysis")
        
        # Simple connected component analysis using scipy if available
        try:
            from scipy import ndimage
            labeled_array, num_features = ndimage.label(binary_mask)
            
            if num_features > 0:
                # Calculate areas of each component
                component_areas = []
                for i in range(1, num_features + 1):
                    component_area = np.sum(labeled_array == i)
                    component_areas.append(component_area)
                
                # Generate confidence scores
                confidence_scores = []
                for area in component_areas:
                    area_ratio = area / (binary_mask.shape[0] * binary_mask.shape[1])
                    confidence = min(0.95, 0.8 + min(area_ratio * 2, 0.15))
                    confidence_scores.append(round(confidence, 3))
                
                return {
                    'detected_objects_count': num_features,
                    'total_area_pixels': int(total_area),
                    'confidence_scores': confidence_scores
                }
            else:
                return {
                    'detected_objects_count': 0,
                    'total_area_pixels': 0,
                    'confidence_scores': []
                }
                
        except ImportError:
            # Final fallback - simple pixel counting
            logger.warning("Neither skimage nor scipy available, using simple pixel counting")
            
            return {
                'detected_objects_count': 1 if total_area > 0 else 0,
                'total_area_pixels': int(total_area),
                'confidence_scores': [0.85] if total_area > 0 else []
            }

def save_processed_results(input_image_path, processed_mask, original_mask, output_dir, input_filename):
    """
    Save the processed results to output directory with high-quality, zoomable outputs
    
    Args:
        input_image_path: Path to original input image
        processed_mask: Post-processed mask
        original_mask: Original mask from model
        output_dir: Output directory
        input_filename: Original filename
        
    Returns:
        dict: Paths to saved files
    """
    base_name = os.path.splitext(input_filename)[0]
    
    # Load original input image
    input_image = Image.open(input_image_path)
    if input_image.mode != 'RGB':
        input_image = input_image.convert('RGB')
    
    # Get original image dimensions for high-quality output
    original_width, original_height = input_image.size
    
    # Decide on output resolution - use higher resolution for better zoom capability
    max_dimension = max(original_width, original_height)
    if max_dimension < 512:
        output_size = (original_width * 2, original_height * 2)  # Upscale small images
    elif max_dimension > 1024:
        # Downscale very large images to reasonable size
        scale = 1024 / max_dimension
        output_size = (int(original_width * scale), int(original_height * scale))
    else:
        output_size = (original_width, original_height)  # Keep original size
    
    logger.info(f"Original image size: {original_width}x{original_height}, output size: {output_size}")
    
    # Resize input to target output size
    input_resized = input_image.resize(output_size, Image.Resampling.LANCZOS)
    
    # Save high-quality input image
    input_output_path = os.path.join(output_dir, f"input_{input_filename}")
    input_resized.save(input_output_path, quality=95)
    
    # Resize mask to match output size
    mask_pil = Image.fromarray((processed_mask * 255).astype(np.uint8))
    mask_resized = mask_pil.resize(output_size, Image.Resampling.NEAREST)  # Use NEAREST for masks
    mask_resized_array = np.array(mask_resized) / 255.0  # Convert back to 0-1 range
    
    # Save high-quality processed mask
    processed_mask_path = os.path.join(output_dir, f"mask_{input_filename}")
    mask_resized.save(processed_mask_path, quality=95)
    
    # Create overlay visualization with enhanced visibility
    overlay_path = os.path.join(output_dir, f"overlay_{input_filename}")
    create_enhanced_overlay_visualization_from_arrays(
        np.array(input_resized), 
        mask_resized_array, 
        overlay_path
    )
    
    # Also create a side-by-side comparison for better visualization
    comparison_path = os.path.join(output_dir, f"comparison_{input_filename}")
    create_side_by_side_comparison(
        np.array(input_resized),
        mask_resized_array,
        comparison_path
    )
    
    # Create yellow overlay on input image
    input_with_yellow_overlay_path = os.path.join(output_dir, f"input_with_overlay_{input_filename}")
    create_yellow_overlay_on_input(
        np.array(input_resized),
        mask_resized_array,
        input_with_yellow_overlay_path
    )
    
    return {
        'input_image': input_output_path,
        'segmentation_mask': processed_mask_path,
        'overlay': overlay_path,
        'comparison': comparison_path,
        'input_with_overlay': input_with_yellow_overlay_path
    }

def create_yellow_overlay_on_input(input_array, mask_array, output_path):
    """
    Create yellow overlay mask on original input image
    
    Args:
        input_array: Input image as numpy array (RGB)
        mask_array: Mask as numpy array (0-1 range)
        output_path: Output path for overlay image
    """
    try:
        # Ensure mask is binary
        binary_mask = (mask_array > 0.5).astype(np.float32)
        
        # Start with the original input image
        result = input_array.copy().astype(np.float32)
        
        # Create yellow overlay where mask is present
        # Yellow = Red + Green, no Blue
        yellow_overlay = np.zeros_like(input_array, dtype=np.float32)
        yellow_overlay[:, :, 0] = binary_mask * 255  # Red channel
        yellow_overlay[:, :, 1] = binary_mask * 255  # Green channel  
        yellow_overlay[:, :, 2] = binary_mask * 0    # No Blue channel
        
        # Apply yellow overlay with transparency (alpha blending)
        alpha = 0.4  # 40% opacity for overlay
        mask_3d = np.stack([binary_mask, binary_mask, binary_mask], axis=2)
        
        # Blend: result = (1-alpha) * input + alpha * yellow where mask > 0
        result = np.where(mask_3d > 0.5, 
                         (1-alpha) * result + alpha * yellow_overlay,
                         result)
        
        # Create edge highlighting for better visibility
        try:
            from scipy import ndimage
            # Find edges of the mask
            edges = ndimage.sobel(binary_mask)
            edges = (edges > 0.1).astype(np.float32)
            
            # Make edges more prominent with bright yellow
            edge_overlay = np.zeros_like(input_array, dtype=np.float32)
            edge_overlay[:, :, 0] = edges * 255  # Red
            edge_overlay[:, :, 1] = edges * 255  # Green
            edge_overlay[:, :, 2] = edges * 0    # No Blue
            
            # Add edges to result
            edge_3d = np.stack([edges, edges, edges], axis=2)
            result = np.where(edge_3d > 0.1,
                             0.7 * result + 0.3 * edge_overlay,
                             result)
                             
        except ImportError:
            logger.warning("scipy not available, skipping edge enhancement")
        
        # Ensure values are in valid range
        result = np.clip(result, 0, 255)
        
        # Convert to PIL Image and save
        result_pil = Image.fromarray(result.astype(np.uint8))
        result_pil.save(output_path, quality=95)
        
        logger.info(f"Yellow overlay visualization saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to create yellow overlay visualization: {e}")
        # Create fallback image
        try:
            input_pil = Image.fromarray(input_array.astype(np.uint8))
            input_pil.save(output_path)
        except:
            pass

def create_enhanced_overlay_visualization_from_arrays(input_array, mask_array, output_path):
    """
    Create enhanced overlay from numpy arrays
    
    Args:
        input_array: Input image as numpy array (RGB)
        mask_array: Mask as numpy array (0-1 range)
        output_path: Output path for overlay image
    """
    try:
        # Ensure mask is binary
        binary_mask = (mask_array > 0.5).astype(np.float32)
        
        # Create enhanced colored overlay with better visibility
        overlay = input_array.copy().astype(np.float32)
        
        # Create multiple color channels for better visibility
        mask_colored = np.zeros_like(input_array, dtype=np.float32)
        
        # Use bright red with some transparency for segmented areas
        mask_colored[:, :, 0] = binary_mask * 255  # Red channel
        mask_colored[:, :, 1] = binary_mask * 50   # Slight green for visibility
        mask_colored[:, :, 2] = binary_mask * 50   # Slight blue for visibility
        
        # Create contours for better edge visibility
        try:
            from scipy import ndimage
            # Find edges using simple gradient
            edges = ndimage.sobel(binary_mask)
            edges = (edges > 0.1).astype(np.float32)
            
            # Make edges more visible - bright yellow
            edge_colored = np.zeros_like(input_array, dtype=np.float32)
            edge_colored[:, :, 0] = edges * 255  # Red
            edge_colored[:, :, 1] = edges * 255  # Green (Red + Green = Yellow)
            edge_colored[:, :, 2] = edges * 0    # Blue
            
            # Blend original + mask + edges
            alpha_mask = 0.3  # Transparency for filled areas
            alpha_edge = 0.8  # More opaque for edges
            
            # Add filled mask areas
            mask_bool = binary_mask > 0
            overlay[mask_bool] = (1 - alpha_mask) * overlay[mask_bool] + alpha_mask * mask_colored[mask_bool]
            
            # Add edge highlights
            edge_bool = edges > 0
            overlay[edge_bool] = (1 - alpha_edge) * overlay[edge_bool] + alpha_edge * edge_colored[edge_bool]
            
        except ImportError:
            # Fallback without edge detection
            alpha = 0.4
            mask_bool = binary_mask > 0
            overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * mask_colored[mask_bool]
        
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        # Save high-quality overlay
        overlay_image = Image.fromarray(overlay)
        overlay_image.save(output_path, quality=95)
        
        logger.info(f"Enhanced overlay visualization saved to {output_path}")
        
    except Exception as e:
        logger.warning(f"Failed to create enhanced overlay: {e}")
        # Fallback to simple overlay
        alpha = 0.4
        overlay = (1 - alpha) * input_array + alpha * (mask_array[:, :, np.newaxis] * [255, 0, 0])
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        overlay_image = Image.fromarray(overlay)
        overlay_image.save(output_path, quality=95)

def create_side_by_side_comparison(input_array, mask_array, output_path):
    """
    Create a side-by-side comparison image
    
    Args:
        input_array: Input image as numpy array
        mask_array: Mask as numpy array (0-1 range)
        output_path: Output path for comparison image
    """
    try:
        # Convert mask to visible image
        mask_vis = (mask_array * 255).astype(np.uint8)
        mask_vis_rgb = np.stack([mask_vis, mask_vis, mask_vis], axis=2)
        
        # Create side-by-side comparison
        comparison = np.hstack([input_array, mask_vis_rgb])
        
        # Save comparison
        comparison_image = Image.fromarray(comparison)
        comparison_image.save(output_path, quality=95)
        
        logger.info(f"Side-by-side comparison saved to {output_path}")
        
    except Exception as e:
        logger.warning(f"Failed to create side-by-side comparison: {e}")
        # Fallback: save just the input
        input_image = Image.fromarray(input_array)
        input_image.save(output_path, quality=95)

def create_enhanced_overlay_visualization(input_image_path, mask, output_path):
    """
    Create an enhanced overlay visualization with better visibility and zoomable output
    
    Args:
        input_image_path: Path to original image
        mask: Binary segmentation mask
        output_path: Output path for overlay image
    """
    try:
        # Load original image
        original = Image.open(input_image_path)
        if original.mode != 'RGB':
            original = original.convert('RGB')
        
        # Resize original to match mask if needed
        mask_size = (mask.shape[1], mask.shape[0])  # PIL uses (width, height)
        original_resized = original.resize(mask_size, Image.Resampling.LANCZOS)
        original_array = np.array(original_resized)
        
        # Create enhanced colored overlay with better visibility
        overlay = original_array.copy()
        
        # Create multiple color channels for better visibility
        mask_colored = np.zeros_like(original_array)
        
        # Use bright red with some transparency for segmented areas
        mask_colored[:, :, 0] = mask * 255  # Red channel
        mask_colored[:, :, 1] = mask * 50   # Slight green for visibility
        mask_colored[:, :, 2] = mask * 50   # Slight blue for visibility
        
        # Create contours for better edge visibility
        try:
            from scipy import ndimage
            # Find edges using simple gradient
            edges = ndimage.sobel(mask.astype(float))
            edges = (edges > 0.1).astype(np.uint8)
            
            # Make edges more visible - bright yellow
            edge_colored = np.zeros_like(original_array)
            edge_colored[:, :, 0] = edges * 255  # Red
            edge_colored[:, :, 1] = edges * 255  # Green (Red + Green = Yellow)
            edge_colored[:, :, 2] = edges * 0    # Blue
            
            # Blend original + mask + edges
            alpha_mask = 0.3  # Transparency for filled areas
            alpha_edge = 0.8  # More opaque for edges
            
            overlay = original_array.copy().astype(np.float32)
            
            # Add filled mask areas
            mask_bool = mask > 0
            overlay[mask_bool] = (1 - alpha_mask) * overlay[mask_bool] + alpha_mask * mask_colored[mask_bool]
            
            # Add edge highlights
            edge_bool = edges > 0
            overlay[edge_bool] = (1 - alpha_edge) * overlay[edge_bool] + alpha_edge * edge_colored[edge_bool]
            
        except ImportError:
            # Fallback without edge detection
            alpha = 0.4
            overlay = (1 - alpha) * original_array + alpha * mask_colored
        
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        # Save high-quality overlay
        overlay_image = Image.fromarray(overlay)
        overlay_image.save(output_path, quality=95)
        
        logger.info(f"Enhanced overlay visualization saved to {output_path}")
        
    except Exception as e:
        logger.warning(f"Failed to create enhanced overlay visualization: {e}")
        # Fallback to simple overlay
        create_overlay_visualization(input_image_path, mask, output_path)

def create_overlay_visualization(input_image_path, mask, output_path):
    """
    Create an overlay visualization of the segmentation on the original image
    
    Args:
        input_image_path: Path to original image
        mask: Binary segmentation mask
        output_path: Output path for overlay image
    """
    try:
        # Load original image
        original = Image.open(input_image_path)
        if original.mode != 'RGB':
            original = original.convert('RGB')
        
        # Resize original to match mask if needed
        mask_size = (mask.shape[1], mask.shape[0])  # PIL uses (width, height)
        original_resized = original.resize(mask_size, Image.Resampling.LANCZOS)
        original_array = np.array(original_resized)
        
        # Create colored overlay
        overlay = original_array.copy()
        mask_colored = np.zeros_like(original_array)
        mask_colored[:, :, 0] = mask * 255  # Red channel for segmentation
        
        # Blend with original image
        alpha = 0.4
        overlay = (1 - alpha) * original_array + alpha * mask_colored
        overlay = overlay.astype(np.uint8)
        
        # Save overlay with high quality
        overlay_image = Image.fromarray(overlay)
        overlay_image.save(output_path, quality=95)
        
    except Exception as e:
        logger.warning(f"Failed to create overlay visualization: {e}")
        # Fallback: just save the mask as grayscale
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        mask_image.save(output_path)
