#!/usr/bin/env python
"""
CADICA Dataset Analyzer and Importer
Analyzes the CADICA dataset and creates entries in Dataset Manager
"""

import os
import sys
import glob
import json
import pandas as pd
from collections import Counter, defaultdict

# Django setup
sys.path.append('/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')

import django
django.setup()

from core.apps.dataset_manager.models import Dataset, DatasetSample, AnnotationSchema

class CADICAAnalyzer:
    def __init__(self, cadica_path='/app/data/datasets/cadica'):
        self.cadica_path = cadica_path
        self.metadata_file = os.path.join(cadica_path, 'metadata.xlsx')
        self.analysis_results = {}
    
    def analyze_dataset(self):
        """Comprehensive analysis of CADICA dataset"""
        print("🔍 Analyzing CADICA dataset...")
        
        # Load metadata
        if os.path.exists(self.metadata_file):
            self.metadata = pd.read_excel(self.metadata_file)
            print(f"📊 Loaded metadata for {len(self.metadata)} patients")
        else:
            print("❌ Metadata file not found")
            return False
        
        # Analyze structure
        self.analyze_structure()
        self.analyze_annotations()
        self.analyze_clinical_data()
        
        return True
    
    def analyze_structure(self):
        """Analyze dataset structure"""
        print("\n🗂️  Analyzing dataset structure...")
        
        # Count patients
        patients = glob.glob(f'{self.cadica_path}/selectedVideos/p*')
        patient_ids = [os.path.basename(p).replace('p', '') for p in patients]
        
        # Count videos and images
        videos = glob.glob(f'{self.cadica_path}/selectedVideos/p*/v*')
        images = glob.glob(f'{self.cadica_path}/selectedVideos/p*/v*/input/*.png')
        annotations = glob.glob(f'{self.cadica_path}/selectedVideos/p*/v*/groundtruth/*.txt')
        
        self.analysis_results['structure'] = {
            'total_patients': len(patients),
            'patient_ids': sorted([int(pid) for pid in patient_ids]),
            'total_videos': len(videos),
            'total_images': len(images),
            'total_annotation_files': len(annotations)
        }
        
        print(f"   Patients: {len(patients)}")
        print(f"   Videos: {len(videos)}")
        print(f"   Images: {len(images)}")
        print(f"   Annotation files: {len(annotations)}")
    
    def analyze_annotations(self):
        """Analyze annotation data"""
        print("\n🏷️  Analyzing annotations...")
        
        class_counter = Counter()
        patient_lesions = defaultdict(int)
        video_lesions = defaultdict(int)
        bbox_stats = []
        
        annotation_files = glob.glob(f'{self.cadica_path}/selectedVideos/p*/v*/groundtruth/*.txt')
        
        for ann_file in annotation_files:
            # Extract patient and video info
            basename = os.path.basename(ann_file)
            parts = basename.split('_')
            if len(parts) >= 3:
                patient_id = parts[0]
                video_id = f"{parts[0]}_{parts[1]}"
                
                try:
                    with open(ann_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            line = line.strip()
                            if line:
                                parts = line.split()
                                if len(parts) >= 5:
                                    x, y, w, h = map(int, parts[:4])
                                    class_name = parts[4]
                                    
                                    class_counter[class_name] += 1
                                    patient_lesions[patient_id] += 1
                                    video_lesions[video_id] += 1
                                    
                                    bbox_stats.append({
                                        'x': x, 'y': y, 'w': w, 'h': h,
                                        'area': w * h,
                                        'class': class_name
                                    })
                except Exception as e:
                    print(f"   Warning: Could not process {ann_file}: {e}")
        
        # Calculate statistics
        areas = [bbox['area'] for bbox in bbox_stats]
        widths = [bbox['w'] for bbox in bbox_stats]
        heights = [bbox['h'] for bbox in bbox_stats]
        
        self.analysis_results['annotations'] = {
            'classes': dict(class_counter),
            'total_bboxes': len(bbox_stats),
            'patients_with_lesions': len(patient_lesions),
            'videos_with_lesions': len(video_lesions),
            'bbox_stats': {
                'avg_area': sum(areas) / len(areas) if areas else 0,
                'avg_width': sum(widths) / len(widths) if widths else 0,
                'avg_height': sum(heights) / len(heights) if heights else 0,
                'min_area': min(areas) if areas else 0,
                'max_area': max(areas) if areas else 0
            }
        }
        
        print(f"   Classes found: {list(class_counter.keys())}")
        print(f"   Total bounding boxes: {len(bbox_stats)}")
        print(f"   Patients with lesions: {len(patient_lesions)}")
        print(f"   Videos with lesions: {len(video_lesions)}")
    
    def analyze_clinical_data(self):
        """Analyze clinical metadata"""
        print("\n🏥 Analyzing clinical data...")
        
        # Basic demographics
        age_stats = self.metadata['Age (years)'].describe()
        sex_dist = self.metadata['Sex'].value_counts()
        
        # Clinical conditions
        diabetes = int(self.metadata['Diabetes mellitus'].sum())
        dyslipidemia = int(self.metadata['Dyslipidemia'].sum())
        hypertension = int(self.metadata['High blood pressure'].sum())
        smokers = int(self.metadata['Smoker'].sum())
        
        # Severity assessment
        vessel_involvement = self.metadata['Number of vessels affected'].value_counts()
        max_degree = self.metadata['Maximum degree of the coronary artery involvement'].value_counts()
        
        self.analysis_results['clinical'] = {
            'demographics': {
                'age_mean': float(age_stats['mean']),
                'age_std': float(age_stats['std']),
                'age_min': float(age_stats['min']),
                'age_max': float(age_stats['max']),
                'sex_distribution': {k: int(v) for k, v in sex_dist.items()}
            },
            'conditions': {
                'diabetes': diabetes,
                'dyslipidemia': dyslipidemia,
                'hypertension': hypertension,
                'smokers': smokers
            },
            'severity': {
                'vessel_involvement': {str(k): int(v) for k, v in vessel_involvement.items()},
                'max_degree': {str(k): int(v) for k, v in max_degree.items()}
            }
        }
        
        print(f"   Age: {age_stats['mean']:.1f} ± {age_stats['std']:.1f} years")
        print(f"   Sex: {dict(sex_dist)}")
        print(f"   Diabetes: {diabetes}/{len(self.metadata)} ({diabetes/len(self.metadata)*100:.1f}%)")
        print(f"   Hypertension: {hypertension}/{len(self.metadata)} ({hypertension/len(self.metadata)*100:.1f}%)")
    
    def create_annotation_schema(self):
        """Create annotation schema for CADICA dataset"""
        print("\n📝 Creating annotation schema...")
        
        classes = self.analysis_results.get('annotations', {}).get('classes', {})
        
        # Create class definitions
        class_definitions = []
        for i, (class_name, count) in enumerate(classes.items()):
            class_definitions.append({
                'id': i,
                'name': class_name,
                'color': f'#{hash(class_name) % 0xFFFFFF:06x}',  # Generate color from hash
                'description': f'CADICA lesion class {class_name} ({count} instances)'
            })
        
        schema_config = {
            'type': 'object_detection',
            'format': 'bbox',
            'coordinate_system': 'pixel',
            'classes': class_definitions,
            'bbox_format': 'xywh',  # x, y, width, height
            'metadata': {
                'dataset': 'CADICA',
                'task': 'coronary_lesion_detection',
                'description': 'Invasive Coronary Angiography lesion detection'
            }
        }
        
        # Create or update schema
        schema, created = AnnotationSchema.objects.get_or_create(
            name='CADICA Lesion Detection',
            defaults={
                'type': 'detection',
                'description': 'CADICA dataset annotation schema for coronary lesion detection',
                'schema_definition': schema_config,
                'is_public': True,
                'created_by_id': 1
            }
        )
        
        if created:
            print(f"   ✅ Created new annotation schema: {schema.name}")
        else:
            schema.schema_definition = schema_config
            schema.save()
            print(f"   🔄 Updated existing annotation schema: {schema.name}")
        
        return schema
    
    def create_dataset_entry(self):
        """Create dataset entry in Dataset Manager"""
        print("\n📁 Creating dataset entry...")
        
        # Create annotation schema first
        schema = self.create_annotation_schema()
        
        # Calculate total size
        total_size = 0
        for root, dirs, files in os.walk(self.cadica_path):
            for file in files:
                filepath = os.path.join(root, file)
                try:
                    total_size += os.path.getsize(filepath)
                except:
                    pass
        
        # Create dataset
        dataset, created = Dataset.objects.get_or_create(
            name='CADICA Coronary Angiography Dataset',
            defaults={
                'description': 'CADICA: Annotated Invasive Coronary Angiography dataset with 42 patients, lesion detection annotations, and clinical metadata',
                'version': '1.0',
                'original_filename': 'cadica',
                'format_type': 'custom',
                'annotation_schema': schema,
                'file_path': 'datasets/cadica',
                'file_size_bytes': total_size,
                'total_samples': self.analysis_results.get('structure', {}).get('total_images', 0),
                'status': 'ready',
                'detected_structure': self.analysis_results,
                'statistics': {
                    'patients': self.analysis_results.get('structure', {}).get('total_patients', 0),
                    'videos': self.analysis_results.get('structure', {}).get('total_videos', 0),
                    'images': self.analysis_results.get('structure', {}).get('total_images', 0),
                    'annotations': self.analysis_results.get('annotations', {}).get('total_bboxes', 0),
                    'classes': list(self.analysis_results.get('annotations', {}).get('classes', {}).keys())
                },
                'is_public': True,
                'created_by_id': 1  # Assume admin user
            }
        )
        
        if created:
            print(f"   ✅ Created new dataset: {dataset.name}")
        else:
            # Update existing dataset
            dataset.detected_structure = self.analysis_results
            dataset.total_samples = self.analysis_results.get('structure', {}).get('total_images', 0)
            dataset.file_size_bytes = total_size
            dataset.save()
            print(f"   🔄 Updated existing dataset: {dataset.name}")
        
        return dataset
    
    def create_sample_entries(self, dataset, limit=100):
        """Create sample entries for preview (limited to avoid overwhelming DB)"""
        print(f"\n🖼️  Creating sample entries (limit: {limit})...")
        
        # Clear existing samples
        DatasetSample.objects.filter(dataset=dataset).delete()
        
        # Get image files
        image_files = glob.glob(f'{self.cadica_path}/selectedVideos/p*/v*/input/*.png')
        
        # Limit samples for performance
        image_files = image_files[:limit]
        
        samples_to_create = []
        for i, img_path in enumerate(image_files):
            try:
                # Extract info from path
                rel_path = os.path.relpath(img_path, self.cadica_path)
                filename = os.path.basename(img_path)
                
                # Extract patient and video info
                path_parts = rel_path.split(os.sep)
                if len(path_parts) >= 3:
                    patient_id = path_parts[1]  # p11
                    video_id = path_parts[2]    # v22
                    
                    # Check if there's a corresponding annotation
                    ann_file = img_path.replace('/input/', '/groundtruth/').replace('.png', '.txt')
                    has_annotation = os.path.exists(ann_file)
                    
                    # Get annotation class if available
                    sample_class = ""
                    if has_annotation:
                        try:
                            with open(ann_file, 'r') as f:
                                first_line = f.readline().strip()
                                if first_line:
                                    parts = first_line.split()
                                    if len(parts) >= 5:
                                        sample_class = parts[4]
                        except:
                            pass
                    
                    sample = DatasetSample(
                        dataset=dataset,
                        file_name=filename,
                        file_path=rel_path,
                        file_size_bytes=os.path.getsize(img_path),
                        file_type='.png',
                        sample_index=i,
                        sample_class=sample_class,
                        annotations={'has_lesion': has_annotation, 'patient': patient_id, 'video': video_id},
                        is_valid=True,
                        preview_data={
                            'patient_id': patient_id,
                            'video_id': video_id,
                            'has_annotation': has_annotation
                        }
                    )
                    samples_to_create.append(sample)
            
            except Exception as e:
                print(f"   Warning: Could not process {img_path}: {e}")
        
        # Bulk create
        DatasetSample.objects.bulk_create(samples_to_create, batch_size=50)
        print(f"   ✅ Created {len(samples_to_create)} sample entries")
        
        return len(samples_to_create)
    
    def print_summary(self):
        """Print analysis summary"""
        print("\n" + "="*60)
        print("📊 CADICA Dataset Analysis Summary")
        print("="*60)
        
        structure = self.analysis_results.get('structure', {})
        annotations = self.analysis_results.get('annotations', {})
        clinical = self.analysis_results.get('clinical', {})
        
        print(f"🗂️  Structure:")
        print(f"   • Patients: {structure.get('total_patients', 0)}")
        print(f"   • Videos: {structure.get('total_videos', 0)}")
        print(f"   • Images: {structure.get('total_images', 0)}")
        print(f"   • Annotation files: {structure.get('total_annotation_files', 0)}")
        
        print(f"\n🏷️  Annotations:")
        print(f"   • Total bounding boxes: {annotations.get('total_bboxes', 0)}")
        print(f"   • Patients with lesions: {annotations.get('patients_with_lesions', 0)}")
        print(f"   • Videos with lesions: {annotations.get('videos_with_lesions', 0)}")
        classes = annotations.get('classes', {})
        for class_name, count in classes.items():
            print(f"   • {class_name}: {count} instances")
        
        print(f"\n🏥 Clinical Data:")
        demographics = clinical.get('demographics', {})
        conditions = clinical.get('conditions', {})
        print(f"   • Age: {demographics.get('age_mean', 0):.1f} ± {demographics.get('age_std', 0):.1f} years")
        print(f"   • Sex: {demographics.get('sex_distribution', {})}")
        print(f"   • Diabetes: {conditions.get('diabetes', 0)}")
        print(f"   • Hypertension: {conditions.get('hypertension', 0)}")
        print(f"   • Smokers: {conditions.get('smokers', 0)}")

def main():
    analyzer = CADICAAnalyzer()
    
    if analyzer.analyze_dataset():
        analyzer.print_summary()
        
        # Create entries in database
        dataset = analyzer.create_dataset_entry()
        sample_count = analyzer.create_sample_entries(dataset, limit=200)
        
        print(f"\n✅ CADICA dataset successfully added to Dataset Manager!")
        print(f"   Dataset ID: {dataset.id}")
        print(f"   Samples created: {sample_count}")
        print(f"   Schema: {dataset.annotation_schema.name}")
    else:
        print("❌ Failed to analyze CADICA dataset")

if __name__ == "__main__":
    main()
