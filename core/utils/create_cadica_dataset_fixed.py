#!/usr/bin/env python
"""
CADICA Dataset Integration with ML Manager - FIXED VERSION
Complete integration including bounding boxes, clinical features, and ML-ready structure
"""

import os
import sys
import glob
import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

# Django setup
sys.path.append('/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')

import django
django.setup()

from core.apps.dataset_manager.models import Dataset, DatasetSample, AnnotationSchema
from core.apps.ml_manager.models import MLModel

class CADICAMLDatasetCreator:
    def __init__(self, cadica_path='/app/data/datasets/cadica'):
        self.cadica_path = cadica_path
        self.metadata_file = os.path.join(cadica_path, 'metadata.xlsx')
        self.analysis_results = {}
        
    def create_comprehensive_dataset(self):
        """Create comprehensive CADICA dataset with all features"""
        print("🏥 Creating comprehensive CADICA dataset for ML Manager...")
        
        # Analyze dataset structure
        self.analyze_dataset_structure()
        
        # Load clinical metadata
        self.load_clinical_metadata()
        
        # Analyze bounding boxes and annotations
        self.analyze_bounding_boxes()
        
        # Create annotation schema
        schema = self.create_annotation_schema()
        
        # Create dataset entry
        dataset = self.create_dataset_entry(schema)
        
        return dataset
    
    def analyze_dataset_structure(self):
        """Analyze overall structure of CADICA dataset"""
        print("   🔍 Analyzing dataset structure...")
        
        selected_videos_path = os.path.join(self.cadica_path, 'selectedVideos')
        non_selected_videos_path = os.path.join(self.cadica_path, 'nonSelectedVideos')
        
        # Count patients, videos, and images
        patient_dirs = glob.glob(os.path.join(selected_videos_path, 'p*'))
        total_patients = len(patient_dirs)
        
        total_videos = 0
        total_images = 0
        lesion_videos = 0
        
        for patient_dir in patient_dirs:
            # Count video directories
            video_dirs = glob.glob(os.path.join(patient_dir, 'v*'))
            total_videos += len(video_dirs)
            
            # Count images
            for video_dir in video_dirs:
                input_dir = os.path.join(video_dir, 'input')
                if os.path.exists(input_dir):
                    images = glob.glob(os.path.join(input_dir, '*.png'))
                    total_images += len(images)
                
                # Check if has lesions
                gt_dir = os.path.join(video_dir, 'groundtruth')
                if os.path.exists(gt_dir):
                    lesion_videos += 1
        
        self.structure_info = {
            'total_patients': total_patients,
            'total_videos': total_videos,
            'total_images': total_images,
            'lesion_videos': lesion_videos,
            'non_lesion_videos': total_videos - lesion_videos
        }
        
        print(f"   📊 Found {total_patients} patients, {total_videos} videos, {total_images} images")
        print(f"   🏷️  {lesion_videos} videos with lesions, {total_videos - lesion_videos} without")
    
    def load_clinical_metadata(self):
        """Load and analyze clinical metadata"""
        print("   🏥 Loading clinical metadata...")
        
        if not os.path.exists(self.metadata_file):
            print(f"   ⚠️  Metadata file not found: {self.metadata_file}")
            self.clinical_metadata = None
            self.clinical_features = None
            return
        
        try:
            # Load metadata
            self.clinical_metadata = pd.read_excel(self.metadata_file)
            
            # Analyze clinical features
            age_values = pd.to_numeric(self.clinical_metadata['Age (years)'], errors='coerce').dropna()
            
            self.clinical_features = {
                'age_stats': {
                    'mean': float(age_values.mean()),
                    'std': float(age_values.std()),
                    'min': int(age_values.min()),
                    'max': int(age_values.max())
                },
                'sex_distribution': self.clinical_metadata['Sex'].value_counts().to_dict(),
                'diabetes_prevalence': float(self.clinical_metadata['Diabetes mellitus'].mean()),
                'hypertension_prevalence': float(self.clinical_metadata['High blood pressure'].mean()),
                'dyslipidemia_prevalence': float(self.clinical_metadata['Dyslipidemia'].mean()),
                'smoking_prevalence': float(self.clinical_metadata['Smoker'].mean())
            }
            
            print(f"   📊 Loaded metadata for {len(self.clinical_metadata)} patients")
            print(f"   👥 Age range: {self.clinical_features['age_stats']['min']}-{self.clinical_features['age_stats']['max']} years")
            
        except Exception as e:
            print(f"   ⚠️  Error loading metadata: {e}")
            self.clinical_metadata = None
            self.clinical_features = None
    
    def analyze_bounding_boxes(self):
        """Analyze bounding box annotations"""
        print("   🏷️  Analyzing bounding box annotations...")
        
        selected_videos_path = os.path.join(self.cadica_path, 'selectedVideos')
        
        all_annotations = []
        class_counts = Counter()
        bbox_sizes = []
        
        patient_dirs = glob.glob(os.path.join(selected_videos_path, 'p*'))
        
        for patient_dir in patient_dirs:
            patient_id = os.path.basename(patient_dir)
            
            video_dirs = glob.glob(os.path.join(patient_dir, 'v*'))
            for video_dir in video_dirs:
                video_id = os.path.basename(video_dir)
                gt_dir = os.path.join(video_dir, 'groundtruth')
                
                if os.path.exists(gt_dir):
                    # Find ground truth files
                    bbox_files = glob.glob(os.path.join(gt_dir, f'{patient_id}_{video_id}_*.txt'))
                    
                    for bbox_file in bbox_files:
                        frame_num = os.path.basename(bbox_file).split('_')[-1].replace('.txt', '')
                        
                        try:
                            with open(bbox_file, 'r') as f:
                                lines = f.readlines()
                                
                            for line in lines:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    x, y, w, h = map(float, parts[:4])
                                    stenosis_class = parts[4] if len(parts) > 4 else 'unknown'
                                    
                                    area = w * h
                                    bbox_sizes.append(area)
                                    class_counts[stenosis_class] += 1
                                    
                                    all_annotations.append({
                                        'patient_id': patient_id,
                                        'video_id': video_id,
                                        'frame': frame_num,
                                        'x': x, 'y': y, 'w': w, 'h': h,
                                        'area': area,
                                        'stenosis_class': stenosis_class,
                                        'severity_level': self._map_stenosis_to_severity(stenosis_class)
                                    })
                        
                        except Exception as e:
                            print(f"   ⚠️  Error reading {bbox_file}: {e}")
        
        self.bbox_analysis = {
            'total_annotations': len(all_annotations),
            'class_distribution': dict(class_counts),
            'average_bbox_size': float(np.mean(bbox_sizes)) if bbox_sizes else 0,
            'bbox_size_std': float(np.std(bbox_sizes)) if bbox_sizes else 0,
            'annotations_per_patient': len(all_annotations) / self.structure_info['total_patients'] if self.structure_info['total_patients'] > 0 else 0
        }
        
        self.all_annotations = all_annotations
        
        print(f"   📊 Found {len(all_annotations)} bounding box annotations")
        print(f"   🏷️  Classes: {list(class_counts.keys())}")
        print(f"   📈 Average {self.bbox_analysis['annotations_per_patient']:.1f} annotations per patient")
    
    def _map_stenosis_to_severity(self, stenosis_class):
        """Map stenosis class to severity level (0-7)"""
        severity_map = {
            'p0_20': 1,    # Minimal stenosis
            'p20_50': 2,   # Mild stenosis  
            'p50_70': 3,   # Moderate stenosis
            'p70_90': 4,   # Severe stenosis
            'p90_98': 5,   # Critical stenosis
            'p99': 6,      # Near-total occlusion
            'p100': 7      # Total occlusion
        }
        return severity_map.get(stenosis_class, 0)
    
    def create_annotation_schema(self):
        """Create comprehensive annotation schema"""
        print("   📋 Creating annotation schema...")
        
        schema_config = {
            "type": "object",
            "properties": {
                "bounding_boxes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number", "description": "Top-left x coordinate"},
                            "y": {"type": "number", "description": "Top-left y coordinate"},
                            "width": {"type": "number", "description": "Bounding box width"},
                            "height": {"type": "number", "description": "Bounding box height"},
                            "stenosis_class": {"type": "string", "description": "Stenosis severity class"},
                            "severity_level": {"type": "integer", "minimum": 0, "maximum": 7},
                            "area": {"type": "number", "description": "Bounding box area"},
                            "clinical_significance": {"type": "string", "description": "Clinical interpretation"}
                        },
                        "required": ["x", "y", "width", "height", "stenosis_class", "severity_level"]
                    }
                },
                "clinical_data": {
                    "type": "object",
                    "properties": {
                        "age": {"type": "integer", "minimum": 0, "maximum": 120},
                        "sex": {"type": "string", "enum": ["M", "F"]},
                        "vessels_affected": {"type": "integer", "minimum": 0, "maximum": 3},
                        "max_stenosis_degree": {"type": "integer", "minimum": 0, "maximum": 100},
                        "risk_factors": {
                            "type": "object",
                            "properties": {
                                "diabetes": {"type": "boolean"},
                                "hypertension": {"type": "boolean"},
                                "dyslipidemia": {"type": "boolean"},
                                "smoking": {"type": "boolean"}
                            }
                        }
                    }
                },
                "image_metadata": {
                    "type": "object", 
                    "properties": {
                        "patient_id": {"type": "string"},
                        "video_id": {"type": "string"},
                        "frame_number": {"type": "string"},
                        "has_lesions": {"type": "boolean"},
                        "lesion_count": {"type": "integer", "minimum": 0}
                    }
                }
            },
            "required": ["image_metadata"]
        }
        
        schema, created = AnnotationSchema.objects.get_or_create(
            name="CADICA Coronary Stenosis Annotations",
            defaults={
                'type': 'detection',
                'schema_definition': schema_config,
                'description': 'Comprehensive schema for CADICA coronary angiography annotations including bounding boxes, stenosis classification, and clinical data',
                'created_by_id': 1  # Use admin user
            }
        )
        
        if created:
            print(f"   ✅ Created new annotation schema: {schema.name}")
        else:
            print(f"   ♻️  Using existing annotation schema: {schema.name}")
        
        return schema
    
    def create_dataset_entry(self, annotation_schema):
        """Create dataset entry in ML Manager"""
        print("   📊 Creating dataset entry...")
        
        # Calculate dataset size
        dataset_size = self._calculate_dataset_size()
        
        # Calculate quality score
        quality_score = self._calculate_quality_score()
        
        # Create dataset
        dataset, created = Dataset.objects.get_or_create(
            name="CADICA Coronary Stenosis Detection Dataset",
            defaults={
                'description': self._create_dataset_description(),
                'version': '1.0',
                'original_filename': 'cadica_dataset',
                'file_path': 'datasets/cadica',
                'extracted_path': 'datasets/cadica',
                'format_type': 'custom',
                'annotation_schema': annotation_schema,
                'quality_score': quality_score,
                'file_size_bytes': dataset_size,
                'total_samples': self.structure_info.get('total_images', 0),
                'status': 'ready',
                'detected_structure': self._create_structure_info(),
                'statistics': self._create_statistics(),
                'class_distribution': self.bbox_analysis.get('class_distribution', {}),
                'created_by_id': 1,  # Admin user
                'is_public': True
            }
        )
        
        if created:
            print(f"   ✅ Created dataset: {dataset.name}")
        else:
            print(f"   ♻️  Using existing dataset: {dataset.name}")
            # Update fields
            dataset.detected_structure = self._create_structure_info()
            dataset.statistics = self._create_statistics()
            dataset.quality_score = quality_score
            dataset.file_size_bytes = dataset_size
            dataset.total_samples = self.structure_info.get('total_images', 0)
            dataset.class_distribution = self.bbox_analysis.get('class_distribution', {})
            dataset.save()
        
        return dataset
    
    def _calculate_dataset_size(self):
        """Calculate total dataset size"""
        total_size = 0
        
        # Calculate size of all PNG files
        selected_videos_path = os.path.join(self.cadica_path, 'selectedVideos')
        for root, dirs, files in os.walk(selected_videos_path):
            for file in files:
                if file.endswith('.png'):
                    file_path = os.path.join(root, file)
                    try:
                        total_size += os.path.getsize(file_path)
                    except:
                        pass
        
        return total_size
    
    def _calculate_quality_score(self):
        """Calculate dataset quality score"""
        score = 0.0
        
        # Expert annotations (40%)
        score += 0.4
        
        # Annotation completeness (20%)
        if self.bbox_analysis['total_annotations'] > 0:
            score += 0.2
        
        # Clinical metadata availability (20%)
        if self.clinical_metadata is not None:
            score += 0.2
        
        # Class balance (10%)
        if self.bbox_analysis['class_distribution']:
            classes = list(self.bbox_analysis['class_distribution'].values())
            balance = 1 - (max(classes) - min(classes)) / sum(classes) if sum(classes) > 0 else 0
            score += balance * 0.1
        
        # Dataset size (10%)
        if self.structure_info['total_images'] > 10000:
            score += 0.1
        elif self.structure_info['total_images'] > 5000:
            score += 0.05
        
        return min(score, 1.0)
    
    def _create_dataset_description(self):
        """Create comprehensive dataset description"""
        return f"""
        CADICA (Coronary Angiography Dataset for Stenosis Detection) - Comprehensive ML Dataset
        
        🏥 **Medical Dataset for AI Development**
        The CADICA dataset is an annotated Invasive Coronary Angiography (ICA) dataset containing {self.structure_info['total_patients']} patients with expert-labeled stenosis annotations. This dataset is specifically designed for computer-aided diagnosis of coronary artery disease severity.
        
        📊 **Dataset Statistics:**
        • Patients: {self.structure_info['total_patients']}
        • Videos: {self.structure_info['total_videos']} 
        • Images: {self.structure_info['total_images']}
        • Annotations: {self.bbox_analysis['total_annotations']} bounding boxes
        • Expert-labeled stenosis classifications
        
        🏷️  **Annotation Classes:**
        • p0_20: Minimal stenosis (0-20%) - No intervention required
        • p20_50: Mild stenosis (20-50%) - Medical therapy
        • p50_70: Moderate stenosis (50-70%) - Consider intervention
        • p70_90: Severe stenosis (70-90%) - Revascularization indicated
        • p90_98: Critical stenosis (90-98%) - Urgent intervention
        • p99: Near-total occlusion (99%) - Emergent intervention
        • p100: Total occlusion (100%) - Immediate revascularization
        
        🎯 **ML Applications:**
        • Automated stenosis detection and classification
        • Clinical decision support systems
        • Medical AI research and development
        • Physician training and education tools
        
        📚 **Clinical Relevance:**
        This dataset bridges the gap between computer vision and clinical cardiology, enabling development of AI systems that can assist cardiologists in accurate stenosis assessment and treatment planning.
        """
    
    def _create_dataset_metadata(self):
        """Create comprehensive dataset metadata"""
        metadata = {
            'dataset_info': {
                'version': '1.0',
                'creation_date': '2024',
                'data_source': 'CADICA - Coronary Angiography Dataset',
                'expert_annotations': True,
                'medical_domain': 'Interventional Cardiology'
            },
            'structure': self.structure_info,
            'annotations': self.bbox_analysis,
            'stenosis_classes': {
                'p0_20': {'range': '0-20%', 'severity': 'Minimal', 'treatment': 'Medical therapy'},
                'p20_50': {'range': '20-50%', 'severity': 'Mild', 'treatment': 'Medical therapy'},
                'p50_70': {'range': '50-70%', 'severity': 'Moderate', 'treatment': 'Consider intervention'},
                'p70_90': {'range': '70-90%', 'severity': 'Severe', 'treatment': 'Revascularization indicated'},
                'p90_98': {'range': '90-98%', 'severity': 'Critical', 'treatment': 'Urgent intervention'},
                'p99': {'range': '99%', 'severity': 'Near-total occlusion', 'treatment': 'Emergent intervention'},
                'p100': {'range': '100%', 'severity': 'Total occlusion', 'treatment': 'Immediate revascularization'}
            },
            'ml_applications': [
                'Stenosis detection',
                'Severity classification', 
                'Clinical decision support',
                'Medical AI research',
                'Physician training'
            ]
        }
        
        if self.clinical_features:
            metadata['clinical_features'] = self.clinical_features
        
        return metadata
    
    def create_comprehensive_samples(self, dataset, max_samples=500):
        """Create comprehensive sample entries with full annotation data"""
        print(f"\n🖼️  Creating comprehensive sample entries (max: {max_samples})...")
        
        selected_videos_path = os.path.join(self.cadica_path, 'selectedVideos')
        
        # Get all image files
        all_image_files = []
        for root, dirs, files in os.walk(selected_videos_path):
            for file in files:
                if file.endswith('.png'):
                    all_image_files.append(os.path.join(root, file))
        
        # Create samples
        sample_count = 0
        
        for idx, image_path in enumerate(all_image_files[:max_samples]):
            try:
                # Extract metadata from path
                rel_path = os.path.relpath(image_path, selected_videos_path)
                path_parts = rel_path.split(os.sep)
                
                if len(path_parts) >= 3:
                    patient_id = path_parts[0]
                    video_id = path_parts[1]
                    filename = path_parts[3]  # skip 'input' folder
                    
                    # Get frame number from filename
                    frame_num = filename.replace('.png', '').split('_')[-1]
                    
                    # Get annotations for this frame
                    frame_annotations = [ann for ann in self.all_annotations 
                                       if ann['patient_id'] == patient_id and 
                                          ann['video_id'] == video_id and 
                                          ann['frame'] == frame_num]
                    
                    # Get clinical data
                    clinical_data = self._get_clinical_data_for_patient(patient_id)
                    
                    # Create sample entry with unique index
                    self._create_sample_entry(dataset, image_path, frame_annotations, clinical_data, 
                                            patient_id, video_id, frame_num, idx)
                    
                    sample_count += 1
                    
                    if sample_count % 100 == 0:
                        print(f"   📊 Created {sample_count} samples...")
                        
            except Exception as e:
                print(f"   ⚠️  Error creating sample data for {image_path}: {e}")
        
        print(f"   ✅ Created {sample_count} comprehensive sample entries")
        return sample_count
    
    def _create_sample_entry(self, dataset, image_path, annotations, clinical_data, 
                           patient_id, video_id, frame_num, sample_index):
        """Create individual sample entry"""
        
        # Prepare bounding boxes
        bboxes = []
        for ann in annotations:
            bboxes.append({
                'x': float(ann['x']),
                'y': float(ann['y']), 
                'width': float(ann['w']),
                'height': float(ann['h']),
                'stenosis_class': ann['stenosis_class'],
                'severity_level': int(ann['severity_level']),
                'area': float(ann['area']),
                'clinical_significance': self._get_clinical_significance(ann['stenosis_class'])
            })
        
        # Create comprehensive annotation data
        annotation_data = {
            'bounding_boxes': bboxes,
            'image_metadata': {
                'patient_id': patient_id,
                'video_id': video_id,
                'frame_number': frame_num,
                'has_lesions': len(bboxes) > 0,
                'lesion_count': len(bboxes)
            }
        }
        
        # Add clinical data if available
        if clinical_data:
            annotation_data['clinical_data'] = clinical_data
        
        # Calculate scores
        complexity_score = self._calculate_complexity_score(bboxes, clinical_data)
        quality_score = self._calculate_sample_quality_score(bboxes, clinical_data)
        
        # Create sample
        sample, created = DatasetSample.objects.get_or_create(
            dataset=dataset,
            file_path=image_path,
            defaults={
                'file_name': os.path.basename(image_path),
                'file_type': 'image/png',
                'sample_index': sample_index,
                'annotations': annotation_data,
                'complexity_score': complexity_score,
                'quality_score': quality_score,
                'preview_data': {
                    'patient_id': patient_id,
                    'video_id': video_id,
                    'frame_number': frame_num,
                    'stenosis_count': len(bboxes),
                    'max_severity': max([bbox['severity_level'] for bbox in bboxes]) if bboxes else 0,
                    'clinical_summary': self._get_clinical_summary(clinical_data),
                    'treatment_urgency': self._assess_treatment_urgency(bboxes, clinical_data)
                },
                'sample_class': f"severity_{max([bbox['severity_level'] for bbox in bboxes]) if bboxes else 0}"
            }
        )
        
        return sample
    
    def _get_clinical_significance(self, stenosis_class):
        """Get clinical significance for stenosis class"""
        significance_map = {
            'p0_20': 'Minimal stenosis - no intervention',
            'p20_50': 'Mild stenosis - medical therapy',
            'p50_70': 'Moderate stenosis - consider intervention', 
            'p70_90': 'Severe stenosis - revascularization indicated',
            'p90_98': 'Critical stenosis - urgent intervention',
            'p99': 'Near-total occlusion - emergent intervention',
            'p100': 'Total occlusion - immediate revascularization'
        }
        return significance_map.get(stenosis_class, 'Unknown significance')
    
    def _get_clinical_data_for_patient(self, patient_id):
        """Get clinical data for specific patient"""
        if self.clinical_metadata is None:
            return {}
            
        try:
            # Extract patient number from ID (e.g., 'p1' -> 1)
            patient_num = int(patient_id.replace('p', ''))
            
            # Find patient in metadata
            patient_row = self.clinical_metadata[self.clinical_metadata['Patient ID'] == patient_num]
            
            if len(patient_row) > 0:
                row = patient_row.iloc[0]
                return {
                    'age': int(row['Age (years)']) if pd.notna(row['Age (years)']) else None,
                    'sex': str(row['Sex']) if pd.notna(row['Sex']) else None,
                    'vessels_affected': int(row['Number of vessels affected']) if pd.notna(row['Number of vessels affected']) else 0,
                    'max_stenosis_degree': self._parse_stenosis_degree(row['Maximum degree of the coronary artery involvement']) if pd.notna(row['Maximum degree of the coronary artery involvement']) else 0,
                    'diabetes': bool(row['Diabetes mellitus']) if pd.notna(row['Diabetes mellitus']) else False,
                    'hypertension': bool(row['High blood pressure']) if pd.notna(row['High blood pressure']) else False,
                    'dyslipidemia': bool(row['Dyslipidemia']) if pd.notna(row['Dyslipidemia']) else False,
                    'smoker': bool(row['Smoker']) if pd.notna(row['Smoker']) else False,
                    'risk_factors': {
                        'diabetes': bool(row['Diabetes mellitus']) if pd.notna(row['Diabetes mellitus']) else False,
                        'hypertension': bool(row['High blood pressure']) if pd.notna(row['High blood pressure']) else False,
                        'dyslipidemia': bool(row['Dyslipidemia']) if pd.notna(row['Dyslipidemia']) else False,
                        'smoking': bool(row['Smoker']) if pd.notna(row['Smoker']) else False
                    }
                }
        except Exception as e:
            print(f"   ⚠️  Error getting clinical data for {patient_id}: {e}")
        
        return {}
    
    def _parse_stenosis_degree(self, value):
        """Parse stenosis degree from various formats"""
        if pd.isna(value):
            return 0
        
        value_str = str(value).strip()
        
        # Handle percentage formats
        if value_str.startswith('>'):
            # '>70%' -> 75 (assume middle of range)
            number = value_str.replace('>', '').replace('%', '')
            try:
                return int(float(number)) + 5  # Add 5 to represent "greater than"
            except:
                return 0
        elif value_str.startswith('<'):
            # '<50%' -> 45 (assume middle of range)
            number = value_str.replace('<', '').replace('%', '')
            try:
                return int(float(number)) - 5  # Subtract 5 to represent "less than"
            except:
                return 0
        elif '%' in value_str:
            # '75%' -> 75
            try:
                return int(float(value_str.replace('%', '')))
            except:
                return 0
        else:
            # Direct number
            try:
                return int(float(value_str))
            except:
                return 0
    
    def _calculate_complexity_score(self, bboxes, clinical_data):
        """Calculate sample complexity score"""
        score = 0.0
        
        # Number of lesions (40%)
        if bboxes:
            lesion_score = min(len(bboxes) / 5.0, 1.0)  # Normalize to max 5 lesions
            score += lesion_score * 0.4
        
        # Stenosis severity (30%)
        if bboxes:
            max_severity = max([bbox['severity_level'] for bbox in bboxes])
            severity_score = max_severity / 7.0  # Normalize to 7 levels
            score += severity_score * 0.3
        
        # Clinical complexity (20%)
        if clinical_data:
            clinical_score = 0
            clinical_score += min(clinical_data.get('vessels_affected', 0) / 3.0, 1.0) * 0.5
            clinical_score += clinical_data.get('age', 0) / 100.0 * 0.3
            risk_factors = sum(clinical_data.get('risk_factors', {}).values()) / 4.0 * 0.2
            clinical_score += risk_factors
            score += min(clinical_score, 1.0) * 0.2
        
        # Lesion size variability (10%)
        if len(bboxes) > 1:
            areas = [bbox['area'] for bbox in bboxes]
            variability = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0
            score += min(variability, 1.0) * 0.1
        
        return min(score, 1.0)
    
    def _calculate_sample_quality_score(self, bboxes, clinical_data):
        """Calculate sample quality score"""
        score = 0.8  # Base quality for expert annotations
        
        # Has annotations (20%)
        if bboxes:
            score += 0.2
        
        # Clinical data completeness (bonus)
        if clinical_data:
            score = min(score + 0.1, 1.0)
        
        return score
    
    def _get_clinical_summary(self, clinical_data):
        """Get brief clinical summary"""
        if not clinical_data:
            return "No clinical data"
        
        age = clinical_data.get('age', 'Unknown')
        sex = clinical_data.get('sex', 'Unknown')
        vessels = clinical_data.get('vessels_affected', 0)
        
        risk_factors = []
        if clinical_data.get('diabetes'): risk_factors.append('DM')
        if clinical_data.get('hypertension'): risk_factors.append('HTN')
        if clinical_data.get('dyslipidemia'): risk_factors.append('DLP')
        if clinical_data.get('smoker'): risk_factors.append('Smoking')
        
        summary = f"{age}y {sex}, {vessels} vessels"
        if risk_factors:
            summary += f", RF: {'+'.join(risk_factors)}"
        
        return summary
    
    def _assess_treatment_urgency(self, bboxes, clinical_data):
        """Assess treatment urgency based on stenosis and clinical data"""
        if not bboxes:
            return "routine"
        
        max_severity = max([bbox['severity_level'] for bbox in bboxes])
        vessels_affected = clinical_data.get('vessels_affected', 1) if clinical_data else 1
        
        # Critical stenosis or multi-vessel disease
        if max_severity >= 6 or (max_severity >= 4 and vessels_affected >= 2):
            return "urgent"
        elif max_severity >= 4:
            return "semi_urgent"
        else:
            return "routine"
    
    def print_integration_summary(self, dataset, sample_count):
        """Print comprehensive integration summary"""
        print("\n" + "="*80)
        print("🏥 CADICA ML Manager Integration - Complete Summary")
        print("="*80)
        
        print(f"\n✅ **Dataset Successfully Created:**")
        print(f"   📊 Dataset ID: {dataset.id}")
        print(f"   📋 Name: {dataset.name}")
        print(f"   📈 Quality Score: {dataset.quality_score:.2f}/1.00")
        print(f"   📁 Total Size: {self._format_bytes(dataset.file_size_bytes)}")
        
        print(f"\n📊 **Comprehensive Statistics:**")
        print(f"   👥 Patients: {self.structure_info['total_patients']}")
        print(f"   🎥 Videos: {self.structure_info['total_videos']}")
        print(f"   🖼️  Images: {self.structure_info['total_images']}")
        print(f"   🏷️  Bounding Box Annotations: {self.bbox_analysis['total_annotations']}")
        print(f"   🖼️  Sample Entries: {sample_count}")
        print(f"   📊 Annotation Coverage: {(self.bbox_analysis['total_annotations'] / self.structure_info['total_images']):.1%}")
        
        print(f"\n🏥 **Clinical Features:**")
        if self.clinical_features:
            print(f"   📊 Patient Demographics: Age, Sex, Comorbidities")
            print(f"   💊 Risk Factors: Diabetes, Hypertension, Dyslipidemia, Smoking")
            print(f"   🏥 Clinical Severity: Vessel involvement, Max stenosis degree")
            print(f"   📈 Age Range: {self.clinical_features['age_stats']['min']}-{self.clinical_features['age_stats']['max']} years")
        
        print(f"\n🏷️  **Stenosis Classification:**")
        for stenosis_class, count in self.bbox_analysis['class_distribution'].items():
            percentage = (count / self.bbox_analysis['total_annotations']) * 100
            clinical_desc = self._get_clinical_significance(stenosis_class)
            print(f"   • {stenosis_class}: {count} annotations ({percentage:.1f}%) - {clinical_desc}")
        
        print(f"\n🎯 **ML Applications Ready:**")
        print(f"   🔍 Automated stenosis detection")
        print(f"   📊 Severity classification (7 classes)")
        print(f"   🏥 Clinical decision support")
        print(f"   📚 Medical AI research")
        print(f"   👩‍⚕️ Physician training tools")
        
        print(f"\n🚀 **Next Steps:**")
        print(f"   1. Use training templates for model development")
        print(f"   2. Validate on clinical datasets")
        print(f"   3. Integrate with clinical workflows")
        print(f"   4. Pursue medical device certification")
    
    def _format_bytes(self, size_bytes):
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def _create_structure_info(self):
        """Create structure information for dataset"""
        return {
            'total_patients': self.structure_info.get('total_patients', 0),
            'total_videos': self.structure_info.get('total_videos', 0),
            'total_images': self.structure_info.get('total_images', 0),
            'lesion_videos': self.structure_info.get('lesion_videos', 0),
            'non_lesion_videos': self.structure_info.get('non_lesion_videos', 0),
            'annotations': self.bbox_analysis.get('total_annotations', 0),
            'clinical_features': self.clinical_features is not None,
            'annotation_coverage': (self.bbox_analysis.get('total_annotations', 0) / 
                                  max(1, self.structure_info.get('total_images', 1))),
            'class_distribution': self.bbox_analysis.get('class_distribution', {})
        }
    
    def _create_statistics(self):
        """Create comprehensive statistics for dataset"""
        stats = {
            'images': {
                'total': self.structure_info.get('total_images', 0),
                'with_annotations': self.bbox_analysis.get('total_annotations', 0) > 0,
                'average_per_patient': (self.structure_info.get('total_images', 0) / 
                                      max(1, self.structure_info.get('total_patients', 1)))
            },
            'annotations': {
                'total': self.bbox_analysis.get('total_annotations', 0),
                'average_per_image': (self.bbox_analysis.get('total_annotations', 0) / 
                                    max(1, self.structure_info.get('total_images', 1))),
                'average_bbox_size': self.bbox_analysis.get('average_bbox_size', 0),
                'bbox_size_std': self.bbox_analysis.get('bbox_size_std', 0)
            },
            'classes': self.bbox_analysis.get('class_distribution', {}),
            'quality': {
                'overall_score': self._calculate_quality_score(),
                'annotation_coverage': (self.bbox_analysis.get('total_annotations', 0) / 
                                      max(1, self.structure_info.get('total_images', 1)))
            }
        }
        
        if self.clinical_features:
            stats['clinical'] = self.clinical_features
            
        return stats
    
def main():
    """Main execution function"""
    creator = CADICAMLDatasetCreator()
    
    print("🏥 Starting CADICA ML Manager Dataset Integration...")
    print("   Creating comprehensive dataset with bounding boxes and clinical features...")
    
    # Create comprehensive dataset
    dataset = creator.create_comprehensive_dataset()
    
    if dataset:
        # Create sample entries
        sample_count = creator.create_comprehensive_samples(dataset, max_samples=500)
        
        # Print comprehensive summary
        creator.print_integration_summary(dataset, sample_count)
        
        print(f"\n🎉 CADICA dataset successfully integrated into ML Manager!")
        print(f"   Ready for medical AI development and clinical research! 🏥")
        
        return dataset
    else:
        print("❌ Failed to create CADICA dataset")
        return None

if __name__ == "__main__":
    main()
