#!/usr/bin/env python
"""
Script to create example ARCADE dataset in Dataset Manager
"""

import os
import sys
import django

# Setup Django
sys.path.append('/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')
django.setup()

from django.contrib.auth.models import User
from core.apps.dataset_manager.models import Dataset, AnnotationSchema, DatasetSample
from django.utils import timezone

def create_arcade_schema():
    """Create annotation schema for ARCADE dataset"""
    
    # Check if schema already exists
    schema, created = AnnotationSchema.objects.get_or_create(
        name="ARCADE Coronary Artery Segmentation",
        defaults={
            'type': 'segmentation',
            'description': 'Multi-class segmentation schema for coronary artery analysis with stenosis detection',
            'schema_definition': {
                "type": "object",
                "properties": {
                    "artery_segments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "segment_id": {"type": "integer", "minimum": 1, "maximum": 26},
                                "segment_name": {"type": "string"},
                                "mask": {"type": "string", "description": "Path to segmentation mask"},
                                "stenosis": {"type": "boolean", "description": "Whether segment has stenosis"}
                            },
                            "required": ["segment_id", "mask"]
                        }
                    },
                    "classes": {
                        "type": "object",
                        "properties": {
                            "background": {"type": "integer", "const": 0},
                            "segments": {"type": "object", "description": "Mapping of segment names to IDs"},
                            "stenosis": {"type": "integer", "const": 26}
                        }
                    }
                }
            },
            'validation_rules': {
                "required": ["artery_segments"],
                "class_count": 27,
                "background_class": 0,
                "stenosis_class": 26
            },
            'example_annotation': {
                "artery_segments": [
                    {
                        "segment_id": 1,
                        "segment_name": "RCA_proximal",
                        "mask": "masks/sample_001_rca_prox.png",
                        "stenosis": False
                    },
                    {
                        "segment_id": 2,
                        "segment_name": "RCA_mid",
                        "mask": "masks/sample_001_rca_mid.png", 
                        "stenosis": True
                    }
                ]
            },
            'is_public': True,
            'created_by': User.objects.get(username='ives')
        }
    )
    
    if created:
        print(f"✅ Created ARCADE annotation schema: {schema.name}")
    else:
        print(f"ℹ️  ARCADE annotation schema already exists: {schema.name}")
    
    return schema

def create_arcade_dataset():
    """Create example ARCADE dataset"""
    
    # Get or create user
    user, created = User.objects.get_or_create(username='ives')
    
    # Get the schema
    schema = create_arcade_schema()
    
    # Create dataset
    dataset, created = Dataset.objects.get_or_create(
        name="ARCADE Coronary Dataset (Example)",
        defaults={
            'description': '''Example dataset from ARCADE (Artery Reconstruction Challenge for Advanced Detection Evaluation).
            
This dataset contains coronary angiography images with multi-class segmentation annotations for:
- 26 coronary artery segments (AHA classification)
- Stenosis detection and localization
- High-resolution masks for precise segmentation training

Dataset characteristics:
- Image format: DICOM/PNG
- Resolution: Variable (typically 512x512 to 1024x1024)
- Annotations: Multi-class segmentation masks
- Classes: Background + 26 artery segments + stenosis markers
- Use cases: Segmentation, classification, stenosis detection''',
            'version': '1.0',
            'format_type': 'custom',
            'annotation_schema': schema,
            'original_filename': 'arcade_example_dataset.zip',
            'file_path': 'datasets/arcade_example/',
            'extracted_path': 'datasets/arcade_example/extracted/',
            'total_samples': 247,  # Realistic number instead of hardcoded thousands
            'file_size_bytes': 1200000000,  # ~1.2GB
            'status': 'ready',
            'is_public': True,
            'created_by': user,
            'class_distribution': {
                'background': 45032847,
                'RCA_proximal': 12450,
                'RCA_mid': 11230,
                'RCA_distal': 9876,
                'LAD_proximal': 13456,
                'LAD_mid': 12789,
                'LAD_distal': 10234,
                'LCX_proximal': 11567,
                'LCX_mid': 9876,
                'stenosis': 2341
            },
            'statistics': {
                'avg_image_size': [768, 768],
                'min_image_size': [512, 512],
                'max_image_size': [1024, 1024],
                'total_pixels': 146126848,
                'stenosis_cases': 89,
                'normal_cases': 158,
                'avg_segments_per_image': 3.2,
                'file_formats': ['dcm', 'png'],
                'annotation_quality': 'expert_reviewed'
            }
        }
    )
    
    if created:
        print(f"✅ Created ARCADE dataset: {dataset.name}")
        print(f"   - Total samples: {dataset.total_samples}")
        print(f"   - File size: {dataset.file_size_bytes / (1024*1024*1024):.1f} GB")
        print(f"   - Status: {dataset.status}")
    else:
        print(f"ℹ️  ARCADE dataset already exists: {dataset.name}")
    
    # Create some sample DatasetSample records
    if created:
        create_sample_records(dataset)
    
    return dataset

def create_sample_records(dataset):
    """Create example sample records for the dataset"""
    
    sample_data = [
        {
            'filename': 'arcade_001.dcm',
            'metadata': {
                'patient_id': 'P001',
                'acquisition_date': '2023-01-15',
                'view': 'RAO_30',
                'contrast': 'iodine',
                'vessel_segments': ['RCA_prox', 'RCA_mid'],
                'stenosis_present': True,
                'stenosis_severity': 'moderate'
            }
        },
        {
            'filename': 'arcade_002.dcm', 
            'metadata': {
                'patient_id': 'P002',
                'acquisition_date': '2023-01-16',
                'view': 'LAO_45',
                'contrast': 'iodine',
                'vessel_segments': ['LAD_prox', 'LAD_mid', 'LCX_prox'],
                'stenosis_present': False,
                'stenosis_severity': 'none'
            }
        },
        {
            'filename': 'arcade_003.dcm',
            'metadata': {
                'patient_id': 'P003', 
                'acquisition_date': '2023-01-17',
                'view': 'Cranial_30',
                'contrast': 'iodine',
                'vessel_segments': ['LAD_prox', 'LAD_mid', 'LAD_distal'],
                'stenosis_present': True,
                'stenosis_severity': 'severe'
            }
        }
    ]
    
    for i, sample_info in enumerate(sample_data):
        sample, created = DatasetSample.objects.get_or_create(
            dataset=dataset,
            file_name=sample_info['filename'],
            defaults={
                'sample_index': i,
                'file_path': f"datasets/arcade_example/images/{sample_info['filename']}",
                'file_type': 'dcm',
                'file_size_bytes': 2048000,  # ~2MB per DICOM
                'sample_class': 'coronary_angiography',
                'annotations': sample_info['metadata'],
                'preview_data': {
                    'mask_path': f"datasets/arcade_example/masks/{sample_info['filename'].replace('.dcm', '_mask.png')}",
                    'segments_annotated': sample_info['metadata']['vessel_segments'],
                    'annotation_quality': 'expert',
                    'annotation_time': '2023-01-20'
                },
                'quality_score': 0.95,
                'annotation_confidence': 0.98,
                'is_valid': True
            }
        )
        
        if created:
            print(f"   ✅ Created sample: {sample.file_name}")

def create_additional_schemas():
    """Create additional useful schemas"""
    
    user = User.objects.get(username='ives')
    
    # Classification schema
    classification_schema, created = AnnotationSchema.objects.get_or_create(
        name="Coronary Artery Classification",
        defaults={
            'type': 'classification',
            'description': 'Binary and multi-class classification for coronary artery analysis',
            'schema_definition': {
                "type": "object",
                "properties": {
                    "stenosis_presence": {"type": "boolean"},
                    "stenosis_severity": {"type": "string", "enum": ["none", "mild", "moderate", "severe"]},
                    "vessel_type": {"type": "string", "enum": ["RCA", "LAD", "LCX"]},
                    "image_quality": {"type": "string", "enum": ["excellent", "good", "fair", "poor"]}
                }
            },
            'validation_rules': {
                "required": ["stenosis_presence"],
                "classes": ["no_stenosis", "mild_stenosis", "moderate_stenosis", "severe_stenosis"]
            },
            'example_annotation': {
                "stenosis_presence": True,
                "stenosis_severity": "moderate", 
                "vessel_type": "LAD",
                "image_quality": "good"
            },
            'is_public': True,
            'created_by': user
        }
    )
    
    # Detection schema
    detection_schema, created = AnnotationSchema.objects.get_or_create(
        name="Stenosis Detection (Bounding Boxes)",
        defaults={
            'type': 'detection',
            'description': 'Object detection schema for stenosis localization with bounding boxes',
            'schema_definition': {
                "type": "object",
                "properties": {
                    "detections": {
                        "type": "array",
                        "items": {
                            "type": "object", 
                            "properties": {
                                "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                                "class": {"type": "string"},
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                "severity": {"type": "string", "enum": ["mild", "moderate", "severe"]}
                            },
                            "required": ["bbox", "class"]
                        }
                    }
                }
            },
            'validation_rules': {
                "classes": ["stenosis", "normal_vessel"],
                "bbox_format": "xyxy",
                "coordinate_system": "image_relative"
            },
            'example_annotation': {
                "detections": [
                    {
                        "bbox": [245, 156, 298, 203],
                        "class": "stenosis",
                        "confidence": 0.89,
                        "severity": "moderate"
                    }
                ]
            },
            'is_public': True,
            'created_by': user
        }
    )
    
    print(f"✅ Created additional schemas")

if __name__ == "__main__":
    print("🚀 Creating ARCADE example dataset...")
    
    try:
        # Create schemas
        create_additional_schemas()
        
        # Create main dataset
        dataset = create_arcade_dataset()
        
        print("\n🎉 ARCADE example dataset created successfully!")
        print(f"📊 Dataset ID: {dataset.id}")
        print(f"📁 Dataset name: {dataset.name}")
        print(f"🔢 Total samples: {dataset.total_samples}")
        print(f"✅ Status: {dataset.status}")
        print("\n💡 You can now view it in the Dataset Manager at: http://localhost:8000/datasets/")
        
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
