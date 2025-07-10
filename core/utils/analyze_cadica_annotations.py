#!/usr/bin/env python
"""
Szczegółowa analiza struktury adnotacji datasetu CADICA
"""

import os
import sys
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import numpy as np

# Django setup
sys.path.append('/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')

import django
django.setup()

class CADICAAnnotationAnalyzer:
    def __init__(self, cadica_path='/app/data/datasets/cadica'):
        self.cadica_path = cadica_path
        self.analysis_results = {}
    
    def analyze_annotation_structure(self):
        """Szczegółowa analiza struktury adnotacji"""
        print("🔍 Analizuję strukturę adnotacji CADICA...")
        
        # Znajdź wszystkie pliki adnotacji
        annotation_files = glob.glob(f'{self.cadica_path}/selectedVideos/p*/v*/groundtruth/*.txt')
        
        # Statystyki adnotacji
        bbox_data = []
        class_distribution = Counter()
        patient_stats = defaultdict(lambda: {'videos': set(), 'annotations': 0, 'classes': set()})
        video_stats = defaultdict(lambda: {'annotations': 0, 'classes': set()})
        frame_stats = defaultdict(int)
        
        print(f"📁 Znaleziono {len(annotation_files)} plików adnotacji")
        
        for ann_file in annotation_files:
            # Wyciągnij informacje o pacjencie i video
            rel_path = os.path.relpath(ann_file, f'{self.cadica_path}/selectedVideos')
            path_parts = rel_path.split(os.sep)
            
            if len(path_parts) >= 4:
                patient_id = path_parts[0]  # p11
                video_id = path_parts[1]    # v22
                filename = path_parts[3]    # p11_v22_00034.txt
                
                # Sprawdź czy to plik groundtruth table czy pojedyncza klatka
                if 'groundTruthTable' in filename:
                    continue
                
                # Wyciągnij numer klatki
                if filename.endswith('.txt'):
                    try:
                        frame_parts = filename.replace('.txt', '').split('_')
                        if len(frame_parts) >= 3:
                            frame_num = frame_parts[2]
                            frame_stats[f"{patient_id}_{video_id}"] += 1
                    except:
                        pass
                
                # Przeanalizuj zawartość pliku
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
                                    
                                    # Dane bounding box
                                    bbox_info = {
                                        'patient': patient_id,
                                        'video': video_id,
                                        'frame': filename,
                                        'x': x, 'y': y, 'w': w, 'h': h,
                                        'area': w * h,
                                        'aspect_ratio': w / h if h > 0 else 0,
                                        'class': class_name
                                    }
                                    bbox_data.append(bbox_info)
                                    
                                    # Statystyki
                                    class_distribution[class_name] += 1
                                    patient_stats[patient_id]['videos'].add(video_id)
                                    patient_stats[patient_id]['annotations'] += 1
                                    patient_stats[patient_id]['classes'].add(class_name)
                                    video_stats[f"{patient_id}_{video_id}"]['annotations'] += 1
                                    video_stats[f"{patient_id}_{video_id}"]['classes'].add(class_name)
                
                except Exception as e:
                    print(f"   ⚠️  Błąd przy przetwarzaniu {ann_file}: {e}")
        
        self.bbox_data = bbox_data
        self.class_distribution = class_distribution
        self.patient_stats = patient_stats
        self.video_stats = video_stats
        self.frame_stats = frame_stats
        
        print(f"✅ Przeanalizowano {len(bbox_data)} adnotacji")
        return True
    
    def analyze_class_meanings(self):
        """Analiza znaczenia klas stenozy"""
        print("\n🏷️  Analiza klas stenozy:")
        
        class_meanings = {
            'p0_20': 'Minimalna stenoza (0-20%)',
            'p20_50': 'Łagodna stenoza (20-50%)',
            'p50_70': 'Umiarkowana stenoza (50-70%)', 
            'p70_90': 'Znaczna stenoza (70-90%)',
            'p90_98': 'Ciężka stenoza (90-98%)',
            'p99': 'Krytyczna stenoza (99%)',
            'p100': 'Całkowite zamknięcie (100%)'
        }
        
        print("📊 Rozkład stopni stenozy:")
        total_annotations = sum(self.class_distribution.values())
        
        for class_name, count in sorted(self.class_distribution.items()):
            meaning = class_meanings.get(class_name, 'Nieznana kategoria')
            percentage = (count / total_annotations) * 100
            print(f"   • {class_name}: {count:4d} ({percentage:5.1f}%) - {meaning}")
        
        return class_meanings
    
    def analyze_bbox_statistics(self):
        """Analiza statystyk bounding boxes"""
        print("\n📏 Statystyki bounding boxes:")
        
        if not self.bbox_data:
            print("   Brak danych bbox")
            return
        
        # Konwersja do numpy arrays dla analizy
        areas = np.array([bbox['area'] for bbox in self.bbox_data])
        widths = np.array([bbox['w'] for bbox in self.bbox_data])
        heights = np.array([bbox['h'] for bbox in self.bbox_data])
        aspect_ratios = np.array([bbox['aspect_ratio'] for bbox in self.bbox_data if bbox['aspect_ratio'] > 0])
        
        print(f"   📦 Obszary (px²):")
        print(f"      Średni: {np.mean(areas):.1f}")
        print(f"      Mediana: {np.median(areas):.1f}")
        print(f"      Min/Max: {np.min(areas):.0f} / {np.max(areas):.0f}")
        print(f"      Odchylenie std: {np.std(areas):.1f}")
        
        print(f"   📐 Wymiary (px):")
        print(f"      Szerokość - śr: {np.mean(widths):.1f}, med: {np.median(widths):.1f}")
        print(f"      Wysokość - śr: {np.mean(heights):.1f}, med: {np.median(heights):.1f}")
        
        print(f"   📏 Proporcje (w/h):")
        print(f"      Średnia: {np.mean(aspect_ratios):.2f}")
        print(f"      Mediana: {np.median(aspect_ratios):.2f}")
        print(f"      Min/Max: {np.min(aspect_ratios):.2f} / {np.max(aspect_ratios):.2f}")
        
        # Analiza rozkładu wielkości dla każdej klasy
        print(f"\n   📊 Średnie obszary według klas:")
        class_areas = defaultdict(list)
        for bbox in self.bbox_data:
            class_areas[bbox['class']].append(bbox['area'])
        
        for class_name in sorted(class_areas.keys()):
            areas_for_class = np.array(class_areas[class_name])
            print(f"      {class_name}: {np.mean(areas_for_class):.1f} ± {np.std(areas_for_class):.1f} px²")
    
    def analyze_patient_distribution(self):
        """Analiza rozkładu adnotacji między pacjentami"""
        print("\n👥 Rozkład adnotacji między pacjentami:")
        
        annotations_per_patient = [stats['annotations'] for stats in self.patient_stats.values()]
        videos_per_patient = [len(stats['videos']) for stats in self.patient_stats.values()]
        classes_per_patient = [len(stats['classes']) for stats in self.patient_stats.values()]
        
        print(f"   📊 Adnotacje na pacjenta:")
        print(f"      Średnio: {np.mean(annotations_per_patient):.1f}")
        print(f"      Mediana: {np.median(annotations_per_patient):.1f}")
        print(f"      Min/Max: {min(annotations_per_patient)} / {max(annotations_per_patient)}")
        
        print(f"   🎥 Video na pacjenta:")
        print(f"      Średnio: {np.mean(videos_per_patient):.1f}")
        print(f"      Mediana: {np.median(videos_per_patient):.1f}")
        print(f"      Min/Max: {min(videos_per_patient)} / {max(videos_per_patient)}")
        
        print(f"   🏷️  Klas na pacjenta:")
        print(f"      Średnio: {np.mean(classes_per_patient):.1f}")
        print(f"      Mediana: {np.median(classes_per_patient):.1f}")
        print(f"      Min/Max: {min(classes_per_patient)} / {max(classes_per_patient)}")
        
        # Top 5 pacjentów z najwięcej adnotacjami
        print(f"\n   🔝 Top 5 pacjentów (liczba adnotacji):")
        sorted_patients = sorted(self.patient_stats.items(), 
                               key=lambda x: x[1]['annotations'], reverse=True)
        for i, (patient_id, stats) in enumerate(sorted_patients[:5]):
            print(f"      {i+1}. {patient_id}: {stats['annotations']} adnotacji, "
                  f"{len(stats['videos'])} video, {len(stats['classes'])} klas")
    
    def analyze_video_distribution(self):
        """Analiza rozkładu adnotacji między video"""
        print("\n🎥 Rozkład adnotacji między video:")
        
        annotations_per_video = [stats['annotations'] for stats in self.video_stats.values()]
        classes_per_video = [len(stats['classes']) for stats in self.video_stats.values()]
        
        print(f"   📊 Adnotacje na video:")
        print(f"      Średnio: {np.mean(annotations_per_video):.1f}")
        print(f"      Mediana: {np.median(annotations_per_video):.1f}")
        print(f"      Min/Max: {min(annotations_per_video)} / {max(annotations_per_video)}")
        
        print(f"   🏷️  Klas na video:")
        print(f"      Średnio: {np.mean(classes_per_video):.1f}")
        print(f"      Mediana: {np.median(classes_per_video):.1f}")
        print(f"      Min/Max: {min(classes_per_video)} / {max(classes_per_video)}")
        
        # Video z wieloma klasami (złożone przypadki)
        multi_class_videos = {video_id: stats for video_id, stats in self.video_stats.items() 
                             if len(stats['classes']) > 3}
        print(f"\n   🔬 Video z wieloma klasami stenozy ({len(multi_class_videos)}):")
        for video_id, stats in sorted(multi_class_videos.items(), 
                                    key=lambda x: len(x[1]['classes']), reverse=True)[:5]:
            classes_str = ', '.join(sorted(stats['classes']))
            print(f"      {video_id}: {len(stats['classes'])} klas ({classes_str})")
    
    def analyze_frame_annotation_density(self):
        """Analiza gęstości adnotacji w klatkach"""
        print("\n🖼️  Gęstość adnotacji w klatkach:")
        
        frames_per_video = list(self.frame_stats.values())
        
        print(f"   📊 Klatki z adnotacjami na video:")
        print(f"      Średnio: {np.mean(frames_per_video):.1f}")
        print(f"      Mediana: {np.median(frames_per_video):.1f}")
        print(f"      Min/Max: {min(frames_per_video)} / {max(frames_per_video)}")
        
        # Video z największą liczbą zanotowanych klatek
        print(f"\n   🎯 Top 5 video (liczba zanotowanych klatek):")
        sorted_videos = sorted(self.frame_stats.items(), key=lambda x: x[1], reverse=True)
        for i, (video_id, frame_count) in enumerate(sorted_videos[:5]):
            annotations_count = self.video_stats[video_id]['annotations']
            classes_count = len(self.video_stats[video_id]['classes'])
            print(f"      {i+1}. {video_id}: {frame_count} klatek, "
                  f"{annotations_count} adnotacji, {classes_count} klas")
    
    def analyze_coordinate_patterns(self):
        """Analiza wzorców lokalizacji adnotacji"""
        print("\n📍 Wzorce lokalizacji adnotacji:")
        
        if not self.bbox_data:
            return
        
        # Analiza pozycji centrum bbox względem obrazu (zakładamy typowy rozmiar angio)
        # Typowe obrazy angio to około 512x512 lub 1024x1024
        assumed_img_size = 512
        
        center_x = [(bbox['x'] + bbox['w']/2) / assumed_img_size for bbox in self.bbox_data]
        center_y = [(bbox['y'] + bbox['h']/2) / assumed_img_size for bbox in self.bbox_data]
        
        print(f"   📊 Pozycje centrum (znormalizowane do 0-1):")
        print(f"      X (lewo-prawo): śr={np.mean(center_x):.3f}, std={np.std(center_x):.3f}")
        print(f"      Y (góra-dół): śr={np.mean(center_y):.3f}, std={np.std(center_y):.3f}")
        
        # Analiza czy stenozy występują w konkretnych regionach dla różnych klas
        print(f"\n   🗺️  Rozkład przestrzenny według klas:")
        class_positions = defaultdict(lambda: {'x': [], 'y': []})
        
        for bbox in self.bbox_data:
            norm_x = (bbox['x'] + bbox['w']/2) / assumed_img_size
            norm_y = (bbox['y'] + bbox['h']/2) / assumed_img_size
            class_positions[bbox['class']]['x'].append(norm_x)
            class_positions[bbox['class']]['y'].append(norm_y)
        
        for class_name in sorted(class_positions.keys()):
            x_vals = class_positions[class_name]['x']
            y_vals = class_positions[class_name]['y']
            if x_vals and y_vals:
                print(f"      {class_name}: X={np.mean(x_vals):.3f}±{np.std(x_vals):.3f}, "
                      f"Y={np.mean(y_vals):.3f}±{np.std(y_vals):.3f}")
    
    def generate_summary_report(self):
        """Generuje podsumowanie analizy"""
        print("\n" + "="*80)
        print("📋 PODSUMOWANIE ANALIZY ADNOTACJI CADICA")
        print("="*80)
        
        total_annotations = len(self.bbox_data)
        total_patients = len(self.patient_stats)
        total_videos = len(self.video_stats)
        total_classes = len(self.class_distribution)
        
        print(f"📊 Podstawowe statystyki:")
        print(f"   • Łączna liczba adnotacji: {total_annotations}")
        print(f"   • Liczba pacjentów: {total_patients}")
        print(f"   • Liczba video: {total_videos}")
        print(f"   • Liczba klas stenozy: {total_classes}")
        
        print(f"\n🏥 Charakterystyka kliniczna:")
        print(f"   • Rozkład stopni stenozy od minimalnej (0-20%) do całkowitego zamknięcia (100%)")
        print(f"   • Najczęstsza: {max(self.class_distribution, key=self.class_distribution.get)} "
              f"({max(self.class_distribution.values())} przypadków)")
        print(f"   • Najrzadsza: {min(self.class_distribution, key=self.class_distribution.get)} "
              f"({min(self.class_distribution.values())} przypadków)")
        
        if self.bbox_data:
            areas = [bbox['area'] for bbox in self.bbox_data]
            print(f"\n📏 Charakterystyka adnotacji:")
            print(f"   • Średni rozmiar adnotacji: {np.mean(areas):.0f} px²")
            print(f"   • Rozrzut wielkości: {np.std(areas):.0f} px² (std)")
            print(f"   • Zakres: {min(areas)}-{max(areas)} px²")
        
        annotations_per_patient = [stats['annotations'] for stats in self.patient_stats.values()]
        print(f"\n👥 Charakterystyka rozkładu:")
        print(f"   • Średnio {np.mean(annotations_per_patient):.1f} adnotacji na pacjenta")
        print(f"   • Zakres: {min(annotations_per_patient)}-{max(annotations_per_patient)} adnotacji na pacjenta")
        
        print(f"\n✅ Dataset CADICA nadaje się do:")
        print(f"   • Treningu modeli detekcji stenozy (object detection)")
        print(f"   • Klasyfikacji stopnia stenozy (7 klas)")
        print(f"   • Analizy rozkładu przestrzennego zmian")
        print(f"   • Badań związanych z angioplastyką wieńcową")

def main():
    analyzer = CADICAAnnotationAnalyzer()
    
    if analyzer.analyze_annotation_structure():
        analyzer.analyze_class_meanings()
        analyzer.analyze_bbox_statistics()
        analyzer.analyze_patient_distribution()
        analyzer.analyze_video_distribution()
        analyzer.analyze_frame_annotation_density()
        analyzer.analyze_coordinate_patterns()
        analyzer.generate_summary_report()
    else:
        print("❌ Błąd podczas analizy adnotacji")

if __name__ == "__main__":
    main()
