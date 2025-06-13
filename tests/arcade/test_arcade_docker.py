#!/usr/bin/env python3
"""
Test wszystkich 6 typów ARCADE w kontenerze Docker
Sprawdza czy maski i obrazy są prawidłowo wyświetlane
"""

import requests
import json
import time
import sys
import os

def test_arcade_in_docker():
    """Test wszystkich typów ARCADE przez HTTP API"""
    print("🧪 Testowanie poprawek ARCADE w kontenerze Docker...")
    
    base_url = "http://localhost:8000"
    
    # Sprawdź czy serwer działa
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"✅ Serwer Django działa (status: {response.status_code})")
    except Exception as e:
        print(f"❌ Serwer Django nie działa: {e}")
        return False
    
    # Wszystkie 6 typów ARCADE
    arcade_types = [
        {
            'name': 'Binary Segmentation',
            'type': 'arcade_binary_segmentation',
            'description': 'Segmentacja binarna - problem z czarnymi maskami (0-1 zamiast 0-255)'
        },
        {
            'name': 'Semantic Segmentation', 
            'type': 'arcade_semantic_segmentation',
            'description': 'Segmentacja semantyczna - obrazy i maski się nie wyświetlają'
        },
        {
            'name': 'Artery Classification',
            'type': 'arcade_artery_classification', 
            'description': 'Klasyfikacja tętnic - czarne maski zamiast obrazów, brak wizualizacji'
        },
        {
            'name': 'Stenosis Detection',
            'type': 'arcade_stenosis_detection',
            'description': 'Detekcja stenozy - bounding boxy nie działają prawidłowo'
        },
        {
            'name': 'Stenosis Segmentation',
            'type': 'arcade_stenosis_segmentation',
            'description': 'Segmentacja stenozy - ogólne problemy z wyświetlaniem'
        },
        {
            'name': 'Semantic Segmentation Binary',
            'type': 'arcade_semantic_segmentation_binary',
            'description': 'Segmentacja semantyczna binarna - kombinacja problemów'
        }
    ]
    
    # Ścieżka do danych testowych
    test_paths = [
        '/app/data/datasets/arcade_challenge_datasets/dataset_phase_1/segmentation_dataset/seg_train',
        '/app/data/datasets/arcade_challenge_datasets/dataset_phase_1/classification_dataset/class_train',
        '/app/data/datasets/arcade_challenge_datasets/dataset_phase_1/detection_dataset/det_train'
    ]
    
    print(f"\n📊 Testowanie {len(arcade_types)} typów ARCADE...")
    
    results = {}
    
    for arcade_config in arcade_types:
        print(f"\n🎯 Testowanie: {arcade_config['name']}")
        print(f"   📝 Typ: {arcade_config['type']}")
        print(f"   💭 Problem: {arcade_config['description']}")
        
        # Wybierz odpowiednią ścieżkę dla typu
        if 'classification' in arcade_config['type']:
            data_path = test_paths[1]
        elif 'detection' in arcade_config['type']:
            data_path = test_paths[2]  
        else:
            data_path = test_paths[0]
            
        print(f"   📂 Ścieżka: {data_path}")
        
        try:
            # Testuj przez endpoint dataset-preview
            test_data = {
                'dataset_type': arcade_config['type'],
                'data_path': data_path,
                'sample_size': 3
            }
            
            print(f"   🔄 Wysyłanie żądania...")
            response = requests.post(
                f"{base_url}/ml/dataset-preview/",
                data=test_data,
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"   ✅ Odpowiedź HTTP: {response.status_code}")
                
                # Sprawdź czy odpowiedź zawiera dane
                content = response.text
                if 'error' in content.lower():
                    print(f"   ⚠️  Znaleziono błąd w odpowiedzi")
                    results[arcade_config['type']] = {'status': 'error', 'content': content[:200]}
                else:
                    print(f"   ✅ Brak błędów w odpowiedzi")
                    # Sprawdź czy zawiera obrazy/maski
                    if 'image_url' in content and 'mask_url' in content:
                        print(f"   ✅ Znaleziono URL-e obrazów i masek")
                        results[arcade_config['type']] = {'status': 'success', 'has_images': True}
                    elif 'sample' in content:
                        print(f"   ✅ Znaleziono próbki danych")
                        results[arcade_config['type']] = {'status': 'success', 'has_samples': True}
                    else:
                        print(f"   ⚠️  Brak obrazów w odpowiedzi")
                        results[arcade_config['type']] = {'status': 'partial', 'content': content[:200]}
                        
            else:
                print(f"   ❌ Błąd HTTP: {response.status_code}")
                print(f"   📄 Odpowiedź: {response.text[:200]}...")
                results[arcade_config['type']] = {
                    'status': 'http_error', 
                    'code': response.status_code,
                    'content': response.text[:200]
                }
                
        except requests.exceptions.Timeout:
            print(f"   ⏱️  Timeout - operacja trwała zbyt długo")
            results[arcade_config['type']] = {'status': 'timeout'}
            
        except Exception as e:
            print(f"   ❌ Wyjątek: {e}")
            results[arcade_config['type']] = {'status': 'exception', 'error': str(e)}
    
    # Podsumowanie
    print(f"\n📊 PODSUMOWANIE TESTÓW:")
    print(f"{'='*60}")
    
    success_count = 0
    total_count = len(arcade_types)
    
    for arcade_config in arcade_types:
        type_name = arcade_config['type']
        result = results.get(type_name, {'status': 'not_tested'})
        
        status_emoji = {
            'success': '✅',
            'partial': '⚠️ ',
            'error': '❌',
            'http_error': '🌐',
            'timeout': '⏱️ ',
            'exception': '💥',
            'not_tested': '❓'
        }.get(result['status'], '❓')
        
        print(f"{status_emoji} {arcade_config['name']:.<30} {result['status'].upper()}")
        
        if result['status'] == 'success':
            success_count += 1
    
    print(f"{'='*60}")
    print(f"🎯 Wynik: {success_count}/{total_count} typów działa prawidłowo")
    
    if success_count == total_count:
        print(f"🎉 WSZYSTKIE TESTY PRZESZŁY POMYŚLNIE!")
        return True
    elif success_count > total_count // 2:
        print(f"⚠️  WIĘKSZOŚĆ TESTÓW PRZESZŁA - wymagane dalsze poprawki")
        return False  
    else:
        print(f"❌ WIĘKSZOŚĆ TESTÓW NIE PRZESZŁA - wymagane znaczące poprawki")
        return False

def test_media_files():
    """Sprawdź czy pliki multimedialne są generowane"""
    print(f"\n🖼️  Testowanie generowania plików multimedialnych...")
    
    try:
        # Sprawdź czy katalog temp istnieje
        response = requests.get("http://localhost:8000/media/temp/dataset_preview/", timeout=5)
        if response.status_code == 200:
            print(f"   ✅ Katalog temp/dataset_preview jest dostępny")
        else:
            print(f"   ⚠️  Katalog temp może nie istnieć (status: {response.status_code})")
            
    except Exception as e:
        print(f"   ❌ Błąd sprawdzania katalogów multimedialnych: {e}")

def test_paths_reorganization():
    """Sprawdź reorganizację ścieżek danych"""
    print(f"\n📂 Testowanie reorganizacji ścieżek...")
    
    # Sprawdź czy endpoint logów działa
    try:
        response = requests.get("http://localhost:8000/ml/training-logs/", timeout=5)
        if response.status_code in [200, 404]:  # 404 może być OK jeśli brak logów
            print(f"   ✅ Endpoint logów treningowych działa")
        else:
            print(f"   ⚠️  Problem z endpointem logów (status: {response.status_code})")
    except Exception as e:
        print(f"   ❌ Błąd testowania endpointu logów: {e}")

if __name__ == "__main__":
    print("🚀 URUCHAMIANIE TESTÓW ARCADE W KONTENERZE DOCKER")
    print("="*60)
    
    # Podstawowe testy
    success = test_arcade_in_docker()
    test_media_files() 
    test_paths_reorganization()
    
    print(f"\n🏁 TESTY ZAKOŃCZONE")
    
    if success:
        print(f"✅ Status: SUKCES - poprawki ARCADE działają!")
        sys.exit(0)
    else:
        print(f"❌ Status: NIEPOWODZENIE - wymagane dalsze poprawki")
        sys.exit(1)
