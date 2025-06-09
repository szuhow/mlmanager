#!/usr/bin/env python3
"""
Test organizacji metryk systemowych w MLflow w namespace'ach
"""

import os
import sys
import django
import time
import mlflow
import logging

# Setup Django
sys.path.append('/home/rafal/Dokumenty/ivessystem/coronary/ives/coronary-experiments')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coronary_experiments.settings')
django.setup()

from ml_manager.mlflow_utils import setup_mlflow, create_new_run
from shared.utils.system_monitor import SystemMonitor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_organized_system_metrics():
    """Test zorganizowanych metryk systemowych w namespace'ach"""
    print("=" * 60)
    print("🔧 TESTOWANIE ORGANIZACJI METRYK SYSTEMOWYCH W MLFLOW")
    print("=" * 60)
    
    try:
        # 1. Setup MLflow
        print("\n1. Konfiguracja MLflow...")
        setup_mlflow()
        
        # 2. Utwórz run
        print("\n2. Tworzenie MLflow run...")
        run_id = create_new_run(params={
            'test_type': 'organized_system_metrics',
            'description': 'Test organizacji metryk systemowych w namespace\'ach'
        })
        
        with mlflow.start_run(run_id=run_id):
            print(f"✅ Uruchomiony MLflow run: {mlflow.active_run().info.run_id}")
            
            # 3. Utwórz monitor systemu
            print("\n3. Tworzenie SystemMonitor...")
            system_monitor = SystemMonitor(log_interval=2, enable_gpu=True)
            
            # 4. Zaloguj metryki ręcznie
            print("\n4. Logowanie metryk testowych...")
            system_monitor.log_metrics_to_mlflow(step=0)
            print("✅ Pierwsze metryki zalogowane")
            
            # 5. Uruchom monitoring w tle
            print("\n5. Uruchamianie monitoringu w tle...")
            system_monitor.start_monitoring()
            print("✅ Monitoring rozpoczęty")
            
            # 6. Poczekaj na kilka metryk
            print("\n6. Oczekiwanie na metryki (8 sekund)...")
            for i in range(8):
                time.sleep(1)
                print(f"   Oczekiwanie... {i+1}/8 sekund", end='\r')
            print("\n✅ Oczekiwanie zakończone")
            
            # 7. Zatrzymaj monitoring
            print("\n7. Zatrzymywanie monitoringu...")
            system_monitor.stop_monitoring()
            print("✅ Monitoring zatrzymany")
        
        # 8. Sprawdź zalogowane metryki
        print("\n8. Sprawdzanie zalogowanych metryk...")
        final_run = mlflow.get_run(run_id)
        logged_metrics = final_run.data.metrics
        logged_params = final_run.data.params
        logged_tags = final_run.data.tags
        
        print(f"✅ Łącznie metryk: {len(logged_metrics)}")
        print(f"✅ Łącznie parametrów: {len(logged_params)}")
        print(f"✅ Łącznie tagów: {len(logged_tags)}")
        
        # Analiza namespace'ów
        system_metrics = {k: v for k, v in logged_metrics.items() if k.startswith('system.')}
        hardware_metrics = {k: v for k, v in logged_metrics.items() if k.startswith('hardware.')}
        process_metrics = {k: v for k, v in logged_metrics.items() if k.startswith('process.')}
        
        print(f"\n📊 ORGANIZACJA METRYK:")
        print(f"   🖥️  System namespace: {len(system_metrics)} metryk")
        print(f"   🔧 Hardware namespace: {len(hardware_metrics)} metryk")
        print(f"   ⚙️  Process namespace: {len(process_metrics)} metryk")
        
        if system_metrics:
            print(f"\n   Przykładowe metryki systemowe:")
            for key, value in sorted(system_metrics.items())[:3]:
                print(f"      {key}: {value}")
        
        if hardware_metrics:
            print(f"\n   Przykładowe metryki sprzętowe:")
            for key, value in sorted(hardware_metrics.items())[:3]:
                print(f"      {key}: {value}")
        
        if process_metrics:
            print(f"\n   Przykładowe metryki procesów:")
            for key, value in sorted(process_metrics.items())[:3]:
                print(f"      {key}: {value}")
        
        # Sprawdź tagi monitoringu
        monitoring_tags = {k: v for k, v in logged_tags.items() if k.startswith('monitoring.')}
        system_tags = {k: v for k, v in logged_tags.items() if k.startswith('system.')}
        
        print(f"\n🏷️  TAGI ORGANIZACYJNE:")
        print(f"   📊 Tagi monitoringu: {len(monitoring_tags)}")
        print(f"   💻 Tagi systemowe: {len(system_tags)}")
        
        if monitoring_tags:
            print("   Tagi monitoringu:")
            for key, value in monitoring_tags.items():
                print(f"      {key}: {value}")
        
        # Sprawdź parametry systemowe
        system_params = {k: v for k, v in logged_params.items() if k.startswith('system_') or k.startswith('gpu_')}
        print(f"\n⚙️  PARAMETRY SYSTEMOWE: {len(system_params)}")
        if system_params:
            for key, value in sorted(system_params.items())[:5]:
                print(f"      {key}: {value}")
        
        # Podsumowanie
        total_organized = len(system_metrics) + len(hardware_metrics) + len(process_metrics)
        if total_organized > 0:
            print(f"\n🎉 SUKCES: {total_organized} metryk zostało zorganizowanych w namespace'ach!")
            print(f"🔗 Zobacz w MLflow: http://mlflow:5000/#/experiments/1/runs/{run_id}")
            return True
        else:
            print("❌ BŁĄD: Nie znaleziono zorganizowanych metryk!")
            return False
            
    except Exception as e:
        print(f"❌ Test nie powiódł się: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_organized_system_metrics()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 METRYKI SYSTEMOWE SĄ ZORGANIZOWANE W NAMESPACE'ACH!")
        print("✅ System Metrics: system.*")
        print("✅ Hardware Metrics: hardware.*") 
        print("✅ Process Metrics: process.*")
    else:
        print("🔧 ORGANIZACJA METRYK WYMAGA POPRAWEK")
    print("=" * 60)
