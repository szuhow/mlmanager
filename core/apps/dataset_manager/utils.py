# Dataset Manager Utility Classes

import os
import json
import zipfile
import tarfile
import shutil
import tempfile
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from django.conf import settings
from django.utils import timezone
from django.core.files.storage import default_storage
import logging
import torch
import torch.nn as nn
from datetime import datetime

try:
    from monai.losses import DiceLoss as MonaiDiceLoss
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False

logger = logging.getLogger(__name__)

class DatasetProcessor:
    """Handles dataset file processing and analysis"""
    
    SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    SUPPORTED_ARCHIVE_EXTENSIONS = {'.zip', '.tar', '.tar.gz', '.tgz'}
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.temp_dir = None
        self.extracted_path = None
    
    def start_processing(self):
        """Start async dataset processing"""
        try:
            self.dataset.status = 'extracting'
            self.dataset.save(update_fields=['status'])
            
            # Extract files
            self._extract_files()
            
            # Analyze structure
            self.dataset.status = 'analyzing'
            self.dataset.save(update_fields=['status'])
            self._analyze_structure()
            
            # Validate data
            self.dataset.status = 'validating'
            self.dataset.save(update_fields=['status'])
            self._validate_data()
            
            # Generate statistics
            self.dataset.status = 'processing'
            self.dataset.save(update_fields=['status'])
            self._generate_statistics()
            
            # Create sample records
            self._create_samples()
            
            # Mark as ready
            self.dataset.status = 'ready'
            self.dataset.training_ready = True
            self.dataset.save(update_fields=['status', 'training_ready'])
            
            logger.info(f"Dataset {self.dataset.id} processed successfully")
            
        except Exception as e:
            self.dataset.status = 'error'
            self.dataset.error_message = str(e)
            self.dataset.save(update_fields=['status', 'error_message'])
            logger.error(f"Dataset {self.dataset.id} processing failed: {e}")
        
        finally:
            self._cleanup()
    
    def _extract_files(self):
        """Extract uploaded archive"""
        file_path = self.dataset.file_path
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix='dataset_')
        self.extracted_path = os.path.join(self.temp_dir, 'extracted')
        os.makedirs(self.extracted_path, exist_ok=True)
        
        # Extract based on file type
        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(self.extracted_path)
        elif file_path.endswith(('.tar', '.tar.gz', '.tgz')):
            with tarfile.open(file_path, 'r:*') as tar_ref:
                tar_ref.extractall(self.extracted_path)
        else:
            # Single file or folder
            shutil.copy2(file_path, self.extracted_path)
        
        # Update dataset with extracted path
        final_extracted_path = os.path.join(
            settings.MEDIA_ROOT, 
            'datasets', 
            str(self.dataset.id)
        )
        os.makedirs(os.path.dirname(final_extracted_path), exist_ok=True)
        shutil.move(self.extracted_path, final_extracted_path)
        
        self.dataset.extracted_path = final_extracted_path
        self.dataset.save(update_fields=['extracted_path'])
        self.extracted_path = final_extracted_path
    
    def _analyze_structure(self):
        """Analyze dataset structure and detect format"""
        structure = self._scan_directory(self.extracted_path)
        
        # Detect format based on structure
        detected_format = self._detect_format(structure)
        
        # Count files and calculate statistics
        total_files = structure.get('file_count', 0)
        image_files = structure.get('image_count', 0)
        
        self.dataset.detected_structure = structure
        self.dataset.total_samples = image_files if image_files > 0 else total_files
        
        if detected_format:
            self.dataset.format_type = detected_format
        
        self.dataset.save(update_fields=[
            'detected_structure', 'total_samples', 'format_type'
        ])
    
    def _scan_directory(self, path: str) -> Dict[str, Any]:
        """Recursively scan directory structure"""
        structure = {
            'type': 'directory',
            'name': os.path.basename(path),
            'path': path,
            'children': [],
            'file_count': 0,
            'image_count': 0,
            'annotation_count': 0,
            'directory_count': 0
        }
        
        try:
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                
                if os.path.isdir(item_path):
                    # Recursively scan subdirectory
                    child_structure = self._scan_directory(item_path)
                    structure['children'].append(child_structure)
                    structure['file_count'] += child_structure['file_count']
                    structure['image_count'] += child_structure['image_count']
                    structure['annotation_count'] += child_structure['annotation_count']
                    structure['directory_count'] += child_structure['directory_count'] + 1
                
                elif os.path.isfile(item_path):
                    # Analyze file
                    file_info = self._analyze_file(item_path)
                    structure['children'].append(file_info)
                    structure['file_count'] += 1
                    
                    if file_info.get('is_image'):
                        structure['image_count'] += 1
                    elif file_info.get('is_annotation'):
                        structure['annotation_count'] += 1
        
        except PermissionError:
            logger.warning(f"Permission denied accessing {path}")
        
        return structure
    
    def _analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze individual file"""
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()
        file_size = os.path.getsize(file_path)
        
        file_info = {
            'type': 'file',
            'name': file_name,
            'path': file_path,
            'extension': file_ext,
            'size': file_size,
            'is_image': file_ext in self.SUPPORTED_IMAGE_EXTENSIONS,
            'is_annotation': file_ext in {'.json', '.xml', '.txt', '.csv'}
        }
        
        # Get image dimensions if it's an image
        if file_info['is_image']:
            try:
                img = cv2.imread(file_path)
                if img is not None:
                    file_info['dimensions'] = {
                        'height': img.shape[0],
                        'width': img.shape[1],
                        'channels': img.shape[2] if len(img.shape) > 2 else 1
                    }
            except Exception as e:
                logger.warning(f"Could not read image {file_path}: {e}")
        
        return file_info
    
    def _detect_format(self, structure: Dict[str, Any]) -> Optional[str]:
        """Detect dataset format based on structure"""
        # Look for common dataset structures
        
        # COCO format detection
        if self._has_coco_structure(structure):
            return 'coco'
        
        # YOLO format detection
        if self._has_yolo_structure(structure):
            return 'yolo'
        
        # Pascal VOC detection
        if self._has_pascal_voc_structure(structure):
            return 'pascal_voc'
        
        # Simple folder structure (images in folders = classes)
        if self._has_classification_structure(structure):
            return 'folder'
        
        return 'custom'
    
    def _has_coco_structure(self, structure: Dict[str, Any]) -> bool:
        """Check if structure matches COCO format"""
        # Look for annotations.json or similar
        for child in structure.get('children', []):
            if child['type'] == 'file' and 'annotation' in child['name'].lower():
                if child['extension'] == '.json':
                    return True
        return False
    
    def _has_yolo_structure(self, structure: Dict[str, Any]) -> bool:
        """Check if structure matches YOLO format"""
        # Look for .txt files alongside images
        has_images = structure['image_count'] > 0
        has_txt_files = any(
            child['extension'] == '.txt' 
            for child in structure.get('children', [])
            if child['type'] == 'file'
        )
        return has_images and has_txt_files
    
    def _has_pascal_voc_structure(self, structure: Dict[str, Any]) -> bool:
        """Check if structure matches Pascal VOC format"""
        # Look for XML annotation files
        has_images = structure['image_count'] > 0
        has_xml_files = any(
            child['extension'] == '.xml'
            for child in structure.get('children', [])
            if child['type'] == 'file'
        )
        return has_images and has_xml_files
    
    def _has_classification_structure(self, structure: Dict[str, Any]) -> bool:
        """Check if structure matches classification folder format"""
        # Images organized in class folders
        class_folders = [
            child for child in structure.get('children', [])
            if child['type'] == 'directory' and child['image_count'] > 0
        ]
        return len(class_folders) > 1
    
    def _validate_data(self):
        """Validate dataset data quality"""
        validation_results = {
            'total_files_checked': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'errors': [],
            'warnings': []
        }
        
        # Validate images
        for root, dirs, files in os.walk(self.extracted_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                validation_results['total_files_checked'] += 1
                
                if file_ext in self.SUPPORTED_IMAGE_EXTENSIONS:
                    if self._validate_image_file(file_path, validation_results):
                        validation_results['valid_files'] += 1
                    else:
                        validation_results['invalid_files'] += 1
        
        # Calculate quality score
        if validation_results['total_files_checked'] > 0:
            quality_score = validation_results['valid_files'] / validation_results['total_files_checked']
        else:
            quality_score = 0.0
        
        self.dataset.validation_results = validation_results
        self.dataset.quality_score = quality_score
        self.dataset.save(update_fields=['validation_results', 'quality_score'])
    
    def _validate_image_file(self, file_path: str, results: Dict[str, Any]) -> bool:
        """Validate individual image file"""
        try:
            # Try to open and read image
            img = cv2.imread(file_path)
            if img is None:
                results['errors'].append(f"Cannot read image: {file_path}")
                return False
            
            # Check dimensions
            height, width = img.shape[:2]
            if height < 32 or width < 32:
                results['warnings'].append(f"Very small image: {file_path} ({width}x{height})")
            
            if height > 4096 or width > 4096:
                results['warnings'].append(f"Very large image: {file_path} ({width}x{height})")
            
            return True
            
        except Exception as e:
            results['errors'].append(f"Error validating {file_path}: {str(e)}")
            return False
    
    def _generate_statistics(self):
        """Generate comprehensive dataset statistics"""
        stats = {
            'file_type_distribution': {},
            'image_dimensions': [],
            'file_sizes': [],
            'class_distribution': {},
            'total_size_bytes': 0
        }
        
        # Analyze all files
        for root, dirs, files in os.walk(self.extracted_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                file_size = os.path.getsize(file_path)
                
                # Update statistics
                stats['file_type_distribution'][file_ext] = stats['file_type_distribution'].get(file_ext, 0) + 1
                stats['file_sizes'].append(file_size)
                stats['total_size_bytes'] += file_size
                
                # Analyze images
                if file_ext in self.SUPPORTED_IMAGE_EXTENSIONS:
                    try:
                        img = cv2.imread(file_path)
                        if img is not None:
                            height, width = img.shape[:2]
                            stats['image_dimensions'].append({'width': width, 'height': height})
                    except Exception:
                        pass
        
        # Generate class distribution if folder structure suggests classification
        if self.dataset.format_type == 'folder':
            stats['class_distribution'] = self._analyze_class_distribution()
        
        # Calculate summary statistics
        if stats['image_dimensions']:
            widths = [d['width'] for d in stats['image_dimensions']]
            heights = [d['height'] for d in stats['image_dimensions']]
            
            stats['image_stats'] = {
                'avg_width': np.mean(widths),
                'avg_height': np.mean(heights),
                'min_width': np.min(widths),
                'max_width': np.max(widths),
                'min_height': np.min(heights),
                'max_height': np.max(heights)
            }
        
        if stats['file_sizes']:
            stats['file_size_stats'] = {
                'avg_size': np.mean(stats['file_sizes']),
                'min_size': np.min(stats['file_sizes']),
                'max_size': np.max(stats['file_sizes']),
                'total_size': stats['total_size_bytes']
            }
        
        self.dataset.statistics = stats
        self.dataset.class_distribution = stats.get('class_distribution', {})
        self.dataset.save(update_fields=['statistics', 'class_distribution'])
    
    def _analyze_class_distribution(self) -> Dict[str, int]:
        """Analyze class distribution for classification datasets"""
        class_counts = {}
        
        for item in os.listdir(self.extracted_path):
            item_path = os.path.join(self.extracted_path, item)
            if os.path.isdir(item_path):
                # Count images in this class folder
                image_count = 0
                for file in os.listdir(item_path):
                    file_ext = os.path.splitext(file)[1].lower()
                    if file_ext in self.SUPPORTED_IMAGE_EXTENSIONS:
                        image_count += 1
                
                if image_count > 0:
                    class_counts[item] = image_count
        
        return class_counts
    
    def _create_samples(self):
        """Create DatasetSample records for preview"""
        from .models import DatasetSample
        
        sample_index = 0
        max_samples = 1000  # Limit for performance
        
        for root, dirs, files in os.walk(self.extracted_path):
            for file in files:
                if sample_index >= max_samples:
                    break
                
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if file_ext in self.SUPPORTED_IMAGE_EXTENSIONS:
                    relative_path = os.path.relpath(file_path, self.extracted_path)
                    
                    # Determine class from folder structure
                    sample_class = ''
                    if self.dataset.format_type == 'folder':
                        path_parts = relative_path.split(os.sep)
                        if len(path_parts) > 1:
                            sample_class = path_parts[0]
                    
                    # Create sample record
                    sample = DatasetSample.objects.create(
                        dataset=self.dataset,
                        file_path=relative_path,
                        file_name=file,
                        file_size_bytes=os.path.getsize(file_path),
                        file_type=file_ext,
                        sample_index=sample_index,
                        sample_class=sample_class
                    )
                    
                    # Generate thumbnail
                    self._generate_thumbnail(sample, file_path)
                    
                    sample_index += 1
    
    def _generate_thumbnail(self, sample, file_path: str):
        """Generate thumbnail for image sample"""
        try:
            img = cv2.imread(file_path)
            if img is not None:
                # Resize to thumbnail
                thumbnail_size = (150, 150)
                height, width = img.shape[:2]
                
                # Calculate aspect ratio preserving resize
                aspect = width / height
                if aspect > 1:
                    new_width = thumbnail_size[0]
                    new_height = int(thumbnail_size[0] / aspect)
                else:
                    new_height = thumbnail_size[1]
                    new_width = int(thumbnail_size[1] * aspect)
                
                thumbnail = cv2.resize(img, (new_width, new_height))
                
                # Save thumbnail
                thumbnail_dir = os.path.join(
                    settings.MEDIA_ROOT, 
                    'thumbnails', 
                    str(self.dataset.id)
                )
                os.makedirs(thumbnail_dir, exist_ok=True)
                
                thumbnail_path = os.path.join(
                    thumbnail_dir, 
                    f"thumb_{sample.sample_index}.jpg"
                )
                cv2.imwrite(thumbnail_path, thumbnail)
                
                # Update sample with thumbnail path
                sample.thumbnail_path = os.path.relpath(thumbnail_path, settings.MEDIA_ROOT)
                sample.preview_data = {
                    'original_dimensions': {'width': width, 'height': height},
                    'thumbnail_dimensions': {'width': new_width, 'height': new_height}
                }
                sample.save(update_fields=['thumbnail_path', 'preview_data'])
                
        except Exception as e:
            logger.warning(f"Could not generate thumbnail for {file_path}: {e}")
    
    def _cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

class SchemaValidator:
    """Validates data against annotation schemas"""
    
    def __init__(self, schema):
        self.schema = schema
    
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against schema"""
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'validated_fields': {}
        }
        
        schema_def = self.schema.schema_definition
        
        # Validate each field in schema
        for field_name, field_config in schema_def.get('fields', {}).items():
            if field_name in data:
                field_result = self._validate_field(
                    field_name, 
                    data[field_name], 
                    field_config
                )
                results['validated_fields'][field_name] = field_result
                
                if not field_result['valid']:
                    results['valid'] = False
                    results['errors'].extend(field_result['errors'])
            
            elif field_config.get('required', False):
                results['valid'] = False
                results['errors'].append(f"Required field '{field_name}' is missing")
        
        return results
    
    def _validate_field(self, field_name: str, value: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate individual field"""
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        field_type = config.get('type', 'string')
        
        # Type validation
        if field_type == 'string' and not isinstance(value, str):
            result['valid'] = False
            result['errors'].append(f"Field '{field_name}' must be a string")
        
        elif field_type == 'number' and not isinstance(value, (int, float)):
            result['valid'] = False
            result['errors'].append(f"Field '{field_name}' must be a number")
        
        elif field_type == 'boolean' and not isinstance(value, bool):
            result['valid'] = False
            result['errors'].append(f"Field '{field_name}' must be a boolean")
        
        elif field_type == 'array' and not isinstance(value, list):
            result['valid'] = False
            result['errors'].append(f"Field '{field_name}' must be an array")
        
        elif field_type == 'object' and not isinstance(value, dict):
            result['valid'] = False
            result['errors'].append(f"Field '{field_name}' must be an object")
        
        # Additional validations
        if result['valid']:
            # Min/max validation for numbers
            if field_type == 'number':
                if 'min' in config and value < config['min']:
                    result['valid'] = False
                    result['errors'].append(f"Field '{field_name}' must be >= {config['min']}")
                
                if 'max' in config and value > config['max']:
                    result['valid'] = False
                    result['errors'].append(f"Field '{field_name}' must be <= {config['max']}")
            
            # Enum validation
            if 'enum' in config and value not in config['enum']:
                result['valid'] = False
                result['errors'].append(f"Field '{field_name}' must be one of: {config['enum']}")
        
        return result

class PipelineExecutor:
    """Executes data processing pipelines"""
    
    def __init__(self, execution):
        self.execution = execution
        self.pipeline = execution.pipeline
        self.dataset = execution.dataset
    
    def start_execution(self):
        """Start pipeline execution"""
        try:
            self.execution.status = 'running'
            self.execution.started_at = timezone.now()
            self.execution.save(update_fields=['status', 'started_at'])
            
            # Execute pipeline steps
            self._execute_pipeline()
            
            # Mark as completed
            self.execution.status = 'completed'
            self.execution.completed_at = timezone.now()
            self.execution.progress_percentage = 100
            self.execution.save(update_fields=[
                'status', 'completed_at', 'progress_percentage'
            ])
            
        except Exception as e:
            self.execution.status = 'failed'
            self.execution.error_details = {'error': str(e)}
            self.execution.save(update_fields=['status', 'error_details'])
            logger.error(f"Pipeline execution {self.execution.id} failed: {e}")
    
    def _execute_pipeline(self):
        """Execute pipeline steps according to graph"""
        pipeline_graph = self.pipeline.pipeline_graph
        nodes = pipeline_graph.get('nodes', [])
        edges = pipeline_graph.get('edges', [])
        
        # Build execution order
        execution_order = self._build_execution_order(nodes, edges)
        
        # Execute each node
        for i, node_id in enumerate(execution_order):
            node = next(n for n in nodes if n['id'] == node_id)
            
            self.execution.current_step = node.get('name', f'Step {i+1}')
            self.execution.progress_percentage = int((i / len(execution_order)) * 100)
            self.execution.save(update_fields=['current_step', 'progress_percentage'])
            
            self._execute_node(node)
            
            self.execution.add_log_entry(
                'info', 
                f"Completed step: {node.get('name', node_id)}"
            )
    
    def _build_execution_order(self, nodes: List[Dict], edges: List[Dict]) -> List[str]:
        """Build topological execution order from graph"""
        # Simple topological sort
        in_degree = {node['id']: 0 for node in nodes}
        
        for edge in edges:
            in_degree[edge['target']] += 1
        
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            node_id = queue.pop(0)
            result.append(node_id)
            
            # Reduce in-degree for connected nodes
            for edge in edges:
                if edge['source'] == node_id:
                    in_degree[edge['target']] -= 1
                    if in_degree[edge['target']] == 0:
                        queue.append(edge['target'])
        
        return result
    
    def _execute_node(self, node: Dict[str, Any]):
        """Execute individual pipeline node"""
        node_type = node.get('type', 'unknown')
        
        if node_type == 'input':
            # Input node - no processing needed
            pass
        
        elif node_type == 'resize':
            self._execute_resize_node(node)
        
        elif node_type == 'normalize':
            self._execute_normalize_node(node)
        
        elif node_type == 'augment':
            self._execute_augment_node(node)
        
        elif node_type == 'filter':
            self._execute_filter_node(node)
        
        elif node_type == 'output':
            self._execute_output_node(node)
        
        else:
            raise ValueError(f"Unknown node type: {node_type}")
    
    def _execute_resize_node(self, node: Dict[str, Any]):
        """Execute image resize operation"""
        config = node.get('config', {})
        target_size = config.get('size', [224, 224])
        
        # TODO: Implement actual image resizing
        self.execution.add_log_entry(
            'info',
            f"Resizing images to {target_size[0]}x{target_size[1]}"
        )
    
    def _execute_normalize_node(self, node: Dict[str, Any]):
        """Execute normalization operation"""
        config = node.get('config', {})
        mean = config.get('mean', [0.485, 0.456, 0.406])
        std = config.get('std', [0.229, 0.224, 0.225])
        
        # TODO: Implement actual normalization
        self.execution.add_log_entry(
            'info',
            f"Normalizing images with mean={mean}, std={std}"
        )
    
    def _execute_augment_node(self, node: Dict[str, Any]):
        """Execute data augmentation"""
        config = node.get('config', {})
        
        # TODO: Implement augmentation
        self.execution.add_log_entry('info', 'Applying data augmentation')
    
    def _execute_filter_node(self, node: Dict[str, Any]):
        """Execute data filtering"""
        config = node.get('config', {})
        
        # TODO: Implement filtering
        self.execution.add_log_entry('info', 'Filtering dataset')
    
    def _execute_output_node(self, node: Dict[str, Any]):
        """Execute output generation"""
        config = node.get('config', {})
        
        # TODO: Generate processed output
        self.execution.add_log_entry('info', 'Generating processed output')

class MixedLoss(nn.Module):
    """
    Mixed loss function combining Dice loss with Binary Cross Entropy.
    Allows configurable proportions for different loss components.
    """
    
    def __init__(self, dice_weight=0.7, bce_weight=0.3, smooth=1e-6):
        """
        Initialize mixed loss function.
        
        Args:
            dice_weight (float): Weight for Dice loss component (0-1)
            bce_weight (float): Weight for BCE loss component (0-1)
            smooth (float): Smoothing factor for numerical stability
        """
        super(MixedLoss, self).__init__()
        
        # Normalize weights
        total_weight = dice_weight + bce_weight
        self.dice_weight = dice_weight / total_weight
        self.bce_weight = bce_weight / total_weight
        self.smooth = smooth
        
        # Initialize loss functions
        if MONAI_AVAILABLE:
            self.dice_loss = MonaiDiceLoss(sigmoid=True, smooth_nr=smooth, smooth_dr=smooth)
        else:
            # Fallback to custom Dice implementation
            self.dice_loss = self._custom_dice_loss
            
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        logger.info(f"MixedLoss initialized: Dice={self.dice_weight:.2f}, BCE={self.bce_weight:.2f}")
    
    def _custom_dice_loss(self, predictions, targets):
        """Custom Dice loss implementation for fallback."""
        predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        predictions_flat = predictions.view(-1)
        targets_flat = targets.view(-1)
        
        # Calculate Dice coefficient
        intersection = (predictions_flat * targets_flat).sum()
        union = predictions_flat.sum() + targets_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice
    
    def forward(self, predictions, targets):
        """
        Calculate mixed loss.
        
        Args:
            predictions: Model predictions (logits)
            targets: Ground truth targets
            
        Returns:
            torch.Tensor: Combined loss value
        """
        # Calculate individual losses
        dice_loss_val = self.dice_loss(predictions, targets)
        bce_loss_val = self.bce_loss(predictions, targets)
        
        # Combine losses
        mixed_loss = (self.dice_weight * dice_loss_val + 
                     self.bce_weight * bce_loss_val)
        
        return mixed_loss
    
    def get_loss_components(self, predictions, targets):
        """
        Get individual loss components for monitoring.
        
        Returns:
            dict: Dictionary with individual loss values
        """
        with torch.no_grad():
            dice_loss_val = self.dice_loss(predictions, targets)
            bce_loss_val = self.bce_loss(predictions, targets)
            mixed_loss_val = (self.dice_weight * dice_loss_val + 
                            self.bce_weight * bce_loss_val)
            
        return {
            'dice_loss': dice_loss_val.item(),
            'bce_loss': bce_loss_val.item(),
            'mixed_loss': mixed_loss_val.item(),
            'dice_weight': self.dice_weight,
            'bce_weight': self.bce_weight
        }


class EnhancedModelCheckpoint:
    """
    Enhanced model checkpointing with automatic saving, versioning, and metadata.
    """
    
    def __init__(self, checkpoint_dir, model_name, save_freq='best', 
                 max_checkpoints=5, monitor_metric='val_loss', mode='min'):
        """
        Initialize enhanced checkpointing.
        
        Args:
            checkpoint_dir (str): Directory to save checkpoints
            model_name (str): Name prefix for checkpoint files
            save_freq (str): Frequency of saving ('best', 'epoch', 'interval')
            max_checkpoints (int): Maximum number of checkpoints to keep
            monitor_metric (str): Metric to monitor for 'best' mode
            mode (str): 'min' or 'max' for monitored metric
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.save_freq = save_freq
        self.max_checkpoints = max_checkpoints
        self.monitor_metric = monitor_metric
        self.mode = mode
        
        # Tracking variables
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.checkpoint_history = []
        
        logger.info(f"EnhancedModelCheckpoint initialized: {checkpoint_dir}")
        logger.info(f"Monitor: {monitor_metric} ({mode}), Max checkpoints: {max_checkpoints}")
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, metrics, 
                       loss_function=None, training_args=None, model_metadata=None):
        """
        Save model checkpoint with comprehensive metadata.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            epoch (int): Current epoch
            metrics (dict): Training and validation metrics
            loss_function: Loss function instance
            training_args (dict): Training arguments
            model_metadata (dict): Additional model metadata
        """
        current_metric = metrics.get(self.monitor_metric)
        should_save = False
        checkpoint_type = 'regular'
        
        # Determine if checkpoint should be saved
        if self.save_freq == 'best':
            if current_metric is not None:
                if ((self.mode == 'min' and current_metric < self.best_metric) or
                    (self.mode == 'max' and current_metric > self.best_metric)):
                    self.best_metric = current_metric
                    should_save = True
                    checkpoint_type = 'best'
        elif self.save_freq == 'epoch':
            should_save = True
        elif isinstance(self.save_freq, int) and epoch % self.save_freq == 0:
            should_save = True
        
        if should_save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create checkpoint filename
            if checkpoint_type == 'best':
                checkpoint_name = f"{self.model_name}_best_epoch_{epoch+1:03d}_{timestamp}.pth"
            else:
                checkpoint_name = f"{self.model_name}_epoch_{epoch+1:03d}_{timestamp}.pth"
            
            checkpoint_path = self.checkpoint_dir / checkpoint_name
            
            # Prepare comprehensive checkpoint data
            checkpoint_data = {
                # Model and training state
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                
                # Training progress
                'epoch': epoch + 1,
                'metrics': metrics,
                'best_metric': self.best_metric,
                'monitor_metric': self.monitor_metric,
                
                # Model metadata
                'model_metadata': model_metadata or {},
                'training_args': training_args or {},
                
                # Loss function info
                'loss_function_class': loss_function.__class__.__name__ if loss_function else None,
                'loss_function_config': getattr(loss_function, 'get_config', lambda: {})(),
                
                # Checkpoint metadata
                'checkpoint_type': checkpoint_type,
                'timestamp': timestamp,
                'pytorch_version': torch.__version__,
            }
            
            # Add mixed loss specific information
            if hasattr(loss_function, 'get_loss_components'):
                checkpoint_data['loss_components'] = {
                    'dice_weight': loss_function.dice_weight,
                    'bce_weight': loss_function.bce_weight,
                    'smooth': loss_function.smooth
                }
            
            # Save checkpoint
            torch.save(checkpoint_data, checkpoint_path)
            
            # Update checkpoint history
            self.checkpoint_history.append({
                'epoch': epoch + 1,
                'path': str(checkpoint_path),
                'metric_value': current_metric,
                'checkpoint_type': checkpoint_type,
                'timestamp': timestamp
            })
            
            # Clean up old checkpoints
            self._cleanup_checkpoints()
            
            logger.info(f"Checkpoint saved: {checkpoint_name}")
            logger.info(f"Metric ({self.monitor_metric}): {current_metric:.6f}")
            
            return checkpoint_path
        
        return None
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints limit."""
        if len(self.checkpoint_history) > self.max_checkpoints:
            # Sort by epoch (keep most recent)
            sorted_checkpoints = sorted(self.checkpoint_history, 
                                      key=lambda x: x['epoch'], reverse=True)
            
            # Keep best checkpoint and recent ones
            to_remove = []
            best_checkpoints = [cp for cp in sorted_checkpoints 
                              if cp['checkpoint_type'] == 'best']
            regular_checkpoints = [cp for cp in sorted_checkpoints 
                                 if cp['checkpoint_type'] == 'regular']
            
            # Keep all best checkpoints and max_checkpoints-1 regular ones
            if len(regular_checkpoints) > self.max_checkpoints - len(best_checkpoints):
                to_remove = regular_checkpoints[self.max_checkpoints - len(best_checkpoints):]
            
            # Remove old checkpoint files
            for checkpoint in to_remove:
                try:
                    checkpoint_path = Path(checkpoint['path'])
                    if checkpoint_path.exists():
                        checkpoint_path.unlink()
                        logger.info(f"Removed old checkpoint: {checkpoint_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {checkpoint['path']}: {e}")
                
                # Remove from history
                self.checkpoint_history.remove(checkpoint)
    
    def load_checkpoint(self, checkpoint_path, model, optimizer=None, scheduler=None):
        """
        Load checkpoint and restore training state.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
            model: PyTorch model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            
        Returns:
            dict: Checkpoint metadata and metrics
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Restore model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore optimizer state
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore scheduler state
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Update tracking variables
        self.best_metric = checkpoint.get('best_metric', self.best_metric)
        
        logger.info(f"Checkpoint loaded: {checkpoint_path.name}")
        logger.info(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
        logger.info(f"Best metric: {self.best_metric:.6f}")
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'best_metric': checkpoint.get('best_metric'),
            'model_metadata': checkpoint.get('model_metadata', {}),
            'training_args': checkpoint.get('training_args', {}),
            'loss_components': checkpoint.get('loss_components', {})
        }
    
    def get_best_checkpoint(self):
        """Get path to best checkpoint."""
        best_checkpoints = [cp for cp in self.checkpoint_history 
                          if cp['checkpoint_type'] == 'best']
        if best_checkpoints:
            return best_checkpoints[-1]['path']  # Most recent best
        return None
    
    def get_checkpoint_summary(self):
        """Get summary of all checkpoints."""
        summary = {
            'total_checkpoints': len(self.checkpoint_history),
            'best_metric': self.best_metric,
            'monitor_metric': self.monitor_metric,
            'checkpoints': self.checkpoint_history.copy()
        }
        return summary


def create_loss_function(loss_type='dice', **kwargs):
    """
    Factory function to create loss functions with different configurations.
    
    Args:
        loss_type (str): Type of loss ('dice', 'bce', 'mixed', 'crossentropy')
        **kwargs: Additional arguments for loss configuration
        
    Returns:
        torch.nn.Module: Configured loss function
    """
    if loss_type == 'mixed':
        dice_weight = kwargs.get('dice_weight', 0.7)
        bce_weight = kwargs.get('bce_weight', 0.3)
        smooth = kwargs.get('smooth', 1e-6)
        return MixedLoss(dice_weight=dice_weight, bce_weight=bce_weight, smooth=smooth)
    
    elif loss_type == 'dice':
        if MONAI_AVAILABLE:
            sigmoid = kwargs.get('sigmoid', True)
            softmax = kwargs.get('softmax', False)
            smooth = kwargs.get('smooth', 1e-6)
            return MonaiDiceLoss(sigmoid=sigmoid, softmax=softmax, 
                               smooth_nr=smooth, smooth_dr=smooth)
        else:
            raise ImportError("MONAI not available for Dice loss")
    
    elif loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    
    elif loss_type == 'crossentropy':
        return nn.CrossEntropyLoss()
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# =============================================================================
# ENHANCED TRAINING FEATURES INTEGRATION EXAMPLES
# =============================================================================

"""
USAGE EXAMPLES FOR ENHANCED TRAINING FEATURES

1. MIXED LOSS FUNCTION:
   
   # Basic usage with default 70% Dice, 30% BCE
   loss_function = MixedLoss()
   
   # Custom proportions - 80% Dice, 20% BCE
   loss_function = MixedLoss(dice_weight=0.8, bce_weight=0.2)
   
   # Using factory function
   loss_function = create_loss_function('mixed', dice_weight=0.6, bce_weight=0.4)
   
   # In training loop
   for batch in dataloader:
       predictions = model(inputs)
       loss = loss_function(predictions, targets)
       
       # Optional: Log individual components
       components = loss_function.get_loss_components(predictions, targets)
       print(f"Dice: {components['dice_loss']:.4f}, BCE: {components['bce_loss']:.4f}")

2. ENHANCED CHECKPOINTING:
   
   # Setup checkpoint manager
   checkpoint_manager = EnhancedModelCheckpoint(
       checkpoint_dir='./checkpoints',
       model_name='unet_model',
       save_freq='best',           # Save only best models
       max_checkpoints=5,          # Keep 5 checkpoints
       monitor_metric='val_dice',  # Monitor validation Dice score
       mode='max'                  # Higher is better for Dice
   )
   
   # In training loop
   for epoch in range(num_epochs):
       # ... training code ...
       
       metrics = {
           'train_loss': train_loss,
           'val_loss': val_loss,
           'val_dice': val_dice,
           'train_dice': train_dice
       }
       
       # Save checkpoint if criteria met
       checkpoint_path = checkpoint_manager.save_checkpoint(
           model=model,
           optimizer=optimizer,
           scheduler=scheduler,
           epoch=epoch,
           metrics=metrics,
           loss_function=loss_function,
           training_args=training_args,
           model_metadata=model_metadata
       )
       
       if checkpoint_path:
           print(f"Saved checkpoint: {checkpoint_path}")
   
   # Load best checkpoint
   best_path = checkpoint_manager.get_best_checkpoint()
   if best_path:
       checkpoint_data = checkpoint_manager.load_checkpoint(
           best_path, model, optimizer, scheduler
       )

3. COMPLETE INTEGRATION:
   
   # Setup everything at once
   loss_function, checkpoint_manager, config = TrainingHelper.setup_training_with_enhancements(
       model_dir='./models/experiment_1',
       model_name='segmentation_model',
       loss_config={
           'type': 'mixed',
           'dice_weight': 0.75,
           'bce_weight': 0.25,
           'smooth': 1e-6
       },
       checkpoint_config={
           'save_freq': 'best',
           'max_checkpoints': 10,
           'monitor_metric': 'val_dice',
           'mode': 'max'
       }
   )
   
   # Training loop with enhanced features
   for epoch in range(num_epochs):
       model.train()
       train_loss = 0.0
       train_dice = 0.0
       
       for batch_idx, (inputs, targets) in enumerate(train_loader):
           optimizer.zero_grad()
           
           predictions = model(inputs)
           loss = loss_function(predictions, targets)
           
           loss.backward()
           optimizer.step()
           
           train_loss += loss.item()
           
           # Log loss components every N batches
           if batch_idx % 10 == 0:
               TrainingHelper.log_loss_components(
                   loss_function, predictions, targets
               )
       
       # Validation
       model.eval()
       val_loss = 0.0
       val_dice = 0.0
       # ... validation code ...
       
       # Update scheduler
       if scheduler:
           scheduler.step()
       
       # Save checkpoint
       metrics = {
           'train_loss': train_loss / len(train_loader),
           'val_loss': val_loss / len(val_loader),
           'train_dice': train_dice / len(train_loader),
           'val_dice': val_dice / len(val_loader)
       }
       
       checkpoint_manager.save_checkpoint(
           model=model,
           optimizer=optimizer,
           scheduler=scheduler,
           epoch=epoch,
           metrics=metrics,
           loss_function=loss_function,
           training_args={'epochs': num_epochs, 'batch_size': batch_size},
           model_metadata={'architecture': 'UNet', 'input_channels': 3}
       )

INTEGRATION WITH EXISTING TRAINING SCRIPT:

To integrate with ml/training/train.py, modify the loss function setup section:

# Replace existing loss function setup (around line 2300)
if args.loss_type == 'mixed':
    loss_function = create_loss_function(
        'mixed',
        dice_weight=args.dice_weight,
        bce_weight=args.bce_weight
    )
elif args.loss_type == 'dice':
    loss_function = create_loss_function('dice', sigmoid=True)
elif args.loss_type == 'bce':
    loss_function = create_loss_function('bce')
else:
    # Default to mixed loss
    loss_function = create_loss_function('mixed')

# Setup enhanced checkpointing
checkpoint_manager = EnhancedModelCheckpoint(
    checkpoint_dir=os.path.join(model_dir, 'checkpoints'),
    model_name=f'model_{args.model_id}',
    save_freq='best',
    max_checkpoints=args.max_checkpoints,
    monitor_metric='val_dice',
    mode='max'
)

Add these arguments to the argument parser:
parser.add_argument('--loss-type', default='mixed', choices=['dice', 'bce', 'mixed'])
parser.add_argument('--dice-weight', type=float, default=0.7)
parser.add_argument('--bce-weight', type=float, default=0.3)
parser.add_argument('--max-checkpoints', type=int, default=5)
"""

# Enhanced Training Helper - Integration class
class TrainingHelper:
    """
    Enhanced training helper with checkpoint and loss management.
    Provides easy integration for GUI training.
    """
    
    def __init__(self, model_dir: str, model_name: str):
        self.model_dir = model_dir
        self.model_name = model_name
        self.loss_function = None
        self.checkpoint_manager = None
    
    def setup_enhanced_training(self, loss_config=None, checkpoint_config=None):
        """Setup enhanced training with loss and checkpoint management."""
        try:
            from ml.utils.loss_manager import LossManager
            from ml.utils.checkpoint_manager import CheckpointManager
            
            # Setup loss function
            if not loss_config:
                loss_config = {
                    "type": "combined",
                    "dice_weight": 0.7,
                    "bce_weight": 0.3
                }
            
            self.loss_function = LossManager.create_loss_function(loss_config)
            
            # Setup checkpoint manager
            if not checkpoint_config:
                checkpoint_config = {
                    "save_strategy": "best",
                    "max_checkpoints": 5,
                    "monitor_metric": "val_dice",
                    "mode": "max"
                }
            
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=f"{self.model_dir}/checkpoints",
                model_name=self.model_name,
                **checkpoint_config
            )
            
            return True
        except ImportError:
            return False
    
    def get_loss_config_from_form_data(self, form_data):
        """Convert form data to loss configuration."""
        loss_type = form_data.get("loss_function", "combined")
        
        if loss_type == "combined":
            return {
                "type": "combined",
                "dice_weight": form_data.get("dice_weight", 0.7),
                "bce_weight": form_data.get("bce_weight", 0.3)
            }
        else:
            return {"type": loss_type}
    
    def get_checkpoint_config_from_form_data(self, form_data):
        """Convert form data to checkpoint configuration."""
        return {
            "save_strategy": form_data.get("checkpoint_strategy", "best"),
            "max_checkpoints": form_data.get("max_checkpoints", 5),
            "monitor_metric": form_data.get("monitor_metric", "val_dice"),
            "mode": "max" if form_data.get("monitor_metric", "val_dice") in ["val_dice", "val_accuracy"] else "min"
        }

def create_enhanced_training_helper(model_dir: str, model_name: str, form_data: dict = None):
    """Create and setup enhanced training helper from form data."""
    helper = TrainingHelper(model_dir, model_name)
    
    if form_data and form_data.get("use_enhanced_training", True):
        loss_config = helper.get_loss_config_from_form_data(form_data)
        checkpoint_config = helper.get_checkpoint_config_from_form_data(form_data)
        
        if helper.setup_enhanced_training(loss_config, checkpoint_config):
            return helper
    
    return None

