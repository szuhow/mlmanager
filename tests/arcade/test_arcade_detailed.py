#!/usr/bin/env python3
"""
Szczegółowy test ARCADE - sprawdza dokładne błędy
"""

import requests

def test_arcade_detailed():
    """Test z dokładnymi błędami"""
    print("🔍 Szczegółowy test ARCADE...")
    
    base_url = "http://localhost:8000"
    
    # Test jednego typu z pełną odpowiedzią
    test_data = {
        'dataset_type': 'arcade_binary_segmentation',
        'data_path': '/app/data/datasets',
        'sample_size': 2
    }
    
    print(f"🎯 Testowanie arcade_binary_segmentation...")
    
    try:
        response = requests.post(
            f"{base_url}/ml/dataset-preview/",
            data=test_data,
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        content = response.text
        print(f"Długość odpowiedzi: {len(content)} znaków")
        
        # Sprawdź błędy
        if 'error' in content.lower():
            print("❌ ZNALEZIONO BŁĘDY:")
            # Znajdź wszystkie wystąpienia error
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'error' in line.lower():
                    # Wyświetl kilka linii kontekstu
                    start = max(0, i-2)
                    end = min(len(lines), i+3)
                    print(f"   Linie {start}-{end}:")
                    for j in range(start, end):
                        marker = ">>> " if j == i else "    "
                        print(f"{marker}{j}: {lines[j][:100]}")
                    print()
        
        # Sprawdź obecność obrazów
        if 'image_url' in content:
            print("✅ Znaleziono image_url")
        else:
            print("❌ Brak image_url")
            
        if 'mask_url' in content:
            print("✅ Znaleziono mask_url")
        else:
            print("❌ Brak mask_url")
            
        # Sprawdź komunikaty o załadowaniu danych
        if 'loaded' in content.lower():
            print("✅ Znaleziono komunikaty o załadowaniu")
        else:
            print("❌ Brak komunikatów o załadowaniu")
        
        # Pokazuj fragment odpowiedzi
        print(f"\n📄 Fragment odpowiedzi (pierwsze 500 znaków):")
        print(content[:500])
        print(f"...")
        print(f"📄 Fragment odpowiedzi (ostatnie 500 znaków):")
        print(content[-500:])
        
    except Exception as e:
        print(f"❌ Wyjątek: {e}")

if __name__ == "__main__":
    test_arcade_detailed()
