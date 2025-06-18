#!/usr/bin/env python
"""
Skrypt do dodania przykładowych próbek do datasetu ARCADE
"""

import os
import sys
import django

# Dodaj ścieżkę projektu
sys.path.append('/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')
django.setup()

from core.apps.dataset_manager.models import Dataset, DatasetSample
import random

def create_sample_data():
    """Tworzy przykładowe próbki dla datasetu ARCADE"""
    
    try:
        # Znajdź dataset ARCADE
        arcade_dataset = Dataset.objects.get(name="ARCADE Coronary Dataset (Example)")
        print(f"📊 Znaleziono dataset: {arcade_dataset.name}")
        
        # Usuń istniejące próbki
        DatasetSample.objects.filter(dataset=arcade_dataset).delete()
        print("🗑️  Usunięto istniejące próbki")
        
        # Lista przykładowych klas z ARCADE
        arcade_classes = [
            'background', 'lad_healthy', 'lad_stenosis', 'lcx_healthy', 'lcx_stenosis',
            'rca_healthy', 'rca_stenosis', 'catheter', 'guidewire', 'balloon',
            'stent', 'contrast_agent'
        ]
        
        sample_types = ['image', 'annotation']
        
        # Tworzenie przykładowych próbek
        samples_to_create = []
        
        for i in range(1, 501):  # 500 próbek
            sample_type = random.choice(sample_types)
            class_label = random.choice(arcade_classes) if random.random() > 0.2 else None
            
            if sample_type == 'image':
                file_name = f"image_{i:04d}.jpg"
                file_extension = '.jpg'
            else:
                file_name = f"annotation_{i:04d}.json"
                file_extension = '.json'
            
            sample = DatasetSample(
                dataset=arcade_dataset,
                file_name=file_name,
                file_path=f"arcade_samples/{file_name}",
                file_size_bytes=random.randint(50000, 500000),  # 50KB - 500KB
                file_type=file_extension,
                sample_index=i,
                sample_class=class_label or "",  # Pusty string zamiast None
                annotations={"type": sample_type, "classes": [class_label] if class_label else []},
                annotation_confidence=random.uniform(0.7, 1.0) if class_label else None,
                is_valid=True,
                preview_data={
                    'width': random.randint(256, 1024) if sample_type == 'image' else None,
                    'height': random.randint(256, 1024) if sample_type == 'image' else None,
                    'channels': 3 if sample_type == 'image' else None,
                    'format': 'JPEG' if sample_type == 'image' else 'JSON'
                },
                quality_score=random.uniform(0.6, 1.0),
                complexity_score=random.uniform(0.3, 0.9)
            )
            samples_to_create.append(sample)
        
        # Bulk create dla wydajności
        DatasetSample.objects.bulk_create(samples_to_create, batch_size=100)
        
        # Aktualizuj statystyki datasetu
        total_samples = DatasetSample.objects.filter(dataset=arcade_dataset).count()
        arcade_dataset.total_samples = total_samples
        arcade_dataset.save()
        
        print(f"✅ Utworzono {total_samples} próbek dla datasetu ARCADE")
        
        # Pokaż statystyki
        image_count = DatasetSample.objects.filter(dataset=arcade_dataset, file_type__in=['.jpg', '.jpeg', '.png']).count()
        annotation_count = DatasetSample.objects.filter(dataset=arcade_dataset, file_type__in=['.json', '.xml']).count()
        annotated_images = DatasetSample.objects.filter(dataset=arcade_dataset).exclude(sample_class="").count()
        
        print(f"📊 Statystyki:")
        print(f"   - Obrazy: {image_count}")
        print(f"   - Adnotacje: {annotation_count}")
        print(f"   - Próbki z klasami: {annotated_images}")
        print(f"   - Klasy: {len(arcade_classes)}")
        
        # Pokaż rozkład klas
        class_distribution = {}
        for sample in DatasetSample.objects.filter(dataset=arcade_dataset).exclude(sample_class=""):
            class_distribution[sample.sample_class] = class_distribution.get(sample.sample_class, 0) + 1
        
        print(f"📈 Rozkład klas:")
        for class_name, count in sorted(class_distribution.items()):
            print(f"   - {class_name}: {count}")
            
    except Dataset.DoesNotExist:
        print("❌ Dataset ARCADE nie został znaleziony. Uruchom najpierw skrypt tworzenia datasetu.")
    except Exception as e:
        print(f"❌ Błąd: {e}")
        raise

if __name__ == "__main__":
    create_sample_data()
