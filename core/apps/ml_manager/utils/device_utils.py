"""
Device detection and management utilities for ML training.
"""

def detect_mps_availability():
    """
    Detect if MPS (Metal Performance Shaders) is available for Apple Silicon.
    Returns a tuple of (is_available, device_info)
    """
    try:
        import torch
        mps_available = torch.backends.mps.is_available() and torch.backends.mps.is_built()
        device_info = {}
        
        if mps_available:
            device_info = {
                'device_type': 'mps',
                'device_name': 'Apple Silicon GPU',
                'backend_built': torch.backends.mps.is_built(),
            }
        
        return mps_available, device_info
    except (ImportError, AttributeError):
        # PyTorch not available or MPS not supported
        return False, {'error': 'MPS not available'}
    except Exception as e:
        return False, {'error': str(e)}

def detect_cuda_availability():
    """
    Detect if CUDA is available for PyTorch operations.
    Returns a tuple of (is_available, device_info)
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device_info = {}
        
        if cuda_available:
            device_info = {
                'cuda_version': torch.version.cuda,
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
                'memory_total': torch.cuda.get_device_properties(0).total_memory if torch.cuda.device_count() > 0 else None,
            }
        
        return cuda_available, device_info
    except ImportError:
        # PyTorch not available
        return False, {'error': 'PyTorch not installed'}
    except Exception as e:
        return False, {'error': str(e)}

def get_device_choices():
    """
    Get available device choices for training.
    Returns a list of tuples suitable for Django ChoiceField.
    """
    choices = [('cpu', 'CPU')]
    
    # Check for MPS (Apple Silicon) first
    mps_available, mps_info = detect_mps_availability()
    if mps_available:
        device_label = f"MPS - {mps_info.get('device_name', 'Apple Silicon GPU')}"
        choices.append(('mps', device_label))
    
    # Check for CUDA
    cuda_available, device_info = detect_cuda_availability()
    if cuda_available and device_info.get('device_count', 0) > 0:
        device_name = device_info.get('device_name', 'CUDA Device')
        memory_gb = None
        if device_info.get('memory_total'):
            memory_gb = round(device_info['memory_total'] / (1024**3), 1)
            device_label = f"CUDA - {device_name} ({memory_gb}GB)"
        else:
            device_label = f"CUDA - {device_name}"
        
        choices.append(('cuda', device_label))
        
        # If multiple CUDA devices are available
        if device_info.get('device_count', 0) > 1:
            for i in range(device_info['device_count']):
                try:
                    import torch
                    device_name = torch.cuda.get_device_name(i)
                    memory_total = torch.cuda.get_device_properties(i).total_memory
                    memory_gb = round(memory_total / (1024**3), 1)
                    choices.append((f'cuda:{i}', f"CUDA:{i} - {device_name} ({memory_gb}GB)"))
                except:
                    choices.append((f'cuda:{i}', f"CUDA:{i}"))
    
    return choices

def get_default_device():
    """
    Get the default device for training.
    Returns 'mps' if available on Apple Silicon, 'cuda' if available, otherwise 'cpu'.
    """
    mps_available, _ = detect_mps_availability()
    if mps_available:
        return 'mps'
    
    cuda_available, _ = detect_cuda_availability()
    if cuda_available:
        return 'cuda'
    
    return 'cpu'

def get_device_info_for_display():
    """
    Get device information for display in the UI.
    Returns a formatted string with device details.
    """
    info_lines = []
    
    # Check MPS first
    mps_available, mps_info = detect_mps_availability()
    if mps_available:
        info_lines.append("✅ MPS Available (Apple Silicon)")
        if mps_info.get('device_name'):
            info_lines.append(f"🎯 GPU: {mps_info['device_name']}")
    else:
        if 'error' not in mps_info or 'MPS not available' not in mps_info.get('error', ''):
            info_lines.append("❌ MPS Not Available")
    
    # Check CUDA
    cuda_available, device_info = detect_cuda_availability()
    if cuda_available:
        info_lines.append("✅ CUDA Available")
        if device_info.get('device_count'):
            info_lines.append(f"📊 {device_info['device_count']} CUDA device(s)")
        if device_info.get('device_name'):
            info_lines.append(f"🎯 Primary: {device_info['device_name']}")
        if device_info.get('memory_total'):
            memory_gb = round(device_info['memory_total'] / (1024**3), 1)
            info_lines.append(f"💾 Memory: {memory_gb}GB")
    else:
        info_lines.append("❌ CUDA Not Available")
        if 'error' in device_info:
            info_lines.append(f"⚠️  {device_info['error']}")
    
    info_lines.append("🖥️  CPU Always Available")
    
    return " | ".join(info_lines)
