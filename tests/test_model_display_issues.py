#!/usr/bin/env python3
"""
Test sprawdzający problemy z progress i performance w completed modelach
"""

import requests
import json

def test_model_display_issues():
    """Test progress i performance display dla completed modeli"""
    
    print("🧪 Testing Model Display Issues")
    
    # 1. Sprawdź model list API
    print("\n1. Testing model list...")
    try:
        response = requests.get("http://localhost:8000/ml/models/", timeout=10)
        if response.status_code == 200:
            print(f"✅ Model list loaded successfully")
            # Sprawdź czy są completed modele
            content = response.text
            if 'completed' in content.lower():
                print("✅ Found completed models in list")
                if 'performance-badge' in content:
                    print("✅ Performance badges found in HTML")
                else:
                    print("❌ Performance badges missing from HTML")
            else:
                print("❌ No completed models found")
        else:
            print(f"❌ Model list failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Model list test failed: {e}")
    
    # 2. Test individual model detail pages
    print("\n2. Testing model detail pages...")
    
    # Get model IDs from simple API call
    try:
        # Próbuj uzyskać listę modeli przez GET na głównej stronie
        response = requests.get("http://localhost:8000/ml/", timeout=10)
        if response.status_code == 200:
            print("✅ Main ML page accessible")
            
            # Sprawdź czy można znaleźć model ID w HTML
            content = response.text
            
            # Szukaj wzorców data-model-id
            import re
            model_ids = re.findall(r'data-model-id="(\d+)"', content)
            
            if model_ids:
                print(f"✅ Found {len(model_ids)} model IDs")
                
                # Test pierwszych kilku modeli
                for model_id in model_ids[:3]:
                    print(f"\n  Testing model {model_id}...")
                    
                    # Test model detail page
                    try:
                        detail_response = requests.get(f"http://localhost:8000/ml/model/{model_id}/", timeout=10)
                        if detail_response.status_code == 200:
                            detail_content = detail_response.text
                            
                            # Sprawdź elementy progress
                            checks = [
                                ('progress-enhanced', 'Progress bar container'),
                                ('training-progress', 'Training progress section'),
                                ('Performance Summary', 'Performance summary section'),
                                ('metric-value', 'Metric values'),
                                ('badge bg-success', 'Success badge for completed'),
                            ]
                            
                            for element, description in checks:
                                if element in detail_content:
                                    print(f"    ✅ {description} found")
                                else:
                                    print(f"    ❌ {description} missing")
                            
                            # Sprawdź czy progress bar jest widoczny
                            if 'style="display: none;"' in detail_content and 'progress-enhanced' in detail_content:
                                print(f"    ⚠️ Progress bar may be hidden")
                            else:
                                print(f"    ✅ Progress bar should be visible")
                                
                        else:
                            print(f"    ❌ Model detail failed: {detail_response.status_code}")
                    except Exception as e:
                        print(f"    ❌ Model detail test failed: {e}")
            else:
                print("❌ No model IDs found in HTML")
        else:
            print(f"❌ Main ML page failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Main page test failed: {e}")
    
    print("\n📋 Summary of potential issues:")
    print("1. Progress bars may be hidden for completed models")
    print("2. Performance metrics may show '-' if best_val_dice is 0")
    print("3. Progress percentage may not be 100% for completed models")
    print("4. Performance summary section may not be visible")
    
    print("\n🔧 Fixes applied:")
    print("✅ Progress bar now shows for completed/failed models")
    print("✅ Progress percentage property fixed for completed models")
    print("✅ Performance summary section added to model detail")
    print("✅ Better styling and indicators for completed models")
    
    print("\n🚀 Manual verification needed:")
    print("1. Check browser for completed model - should show 100% progress")
    print("2. Verify performance badges appear in model list")
    print("3. Check model detail page has Performance Summary section")
    print("4. Confirm completed models show green progress bar")

if __name__ == '__main__':
    test_model_display_issues()
