#!/usr/bin/env python3
"""
Test script to verify MLflow experiment fixes are working correctly
"""

import os
import sys
import django

# Setup Django
sys.path.append('/home/rafal/Dokumenty/ivessystem/coronary/ives/coronary-experiments')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.testing')
django.setup()

from ml_manager.mlflow_utils import setup_mlflow
import mlflow
from django.conf import settings

def test_mlflow_experiment_fix():
    """Test that MLflow experiments are set up correctly"""
    print('=' * 60)
    print('🧪 TESTING MLFLOW EXPERIMENT SETUP FIXES')
    print('=' * 60)
    
    print(f'Django MLFLOW_EXPERIMENT_NAME: {settings.MLFLOW_EXPERIMENT_NAME}')
    print(f'Django MLFLOW_TRACKING_URI: {settings.MLFLOW_TRACKING_URI}')

    # Test setup_mlflow function
    print('\n1. Testing setup_mlflow() function...')
    try:
        setup_mlflow()
        print('✅ setup_mlflow() executed successfully')
    except Exception as e:
        print(f'❌ setup_mlflow() failed: {e}')
        return False

    # Check current experiment
    try:
        experiment = mlflow.get_experiment_by_name(settings.MLFLOW_EXPERIMENT_NAME)
        if experiment:
            print(f'✅ Experiment "{settings.MLFLOW_EXPERIMENT_NAME}" exists with ID: {experiment.experiment_id}')
        else:
            print(f'❌ Experiment "{settings.MLFLOW_EXPERIMENT_NAME}" not found')
            return False
    except Exception as e:
        print(f'❌ Error checking experiment: {e}')
        return False

    # Check MLflow client and list all experiments
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        print(f'\n2. All experiments:')
        for exp in experiments:
            marker = '🎯' if exp.name == settings.MLFLOW_EXPERIMENT_NAME else '  '
            print(f'   {marker} {exp.name} (ID: {exp.experiment_id})')
    except Exception as e:
        print(f'❌ Error listing experiments: {e}')
        return False

    # Test creating a new run
    print(f'\n3. Testing run creation in experiment "{settings.MLFLOW_EXPERIMENT_NAME}"...')
    try:
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run() as run:
            run_info = run.info
            print(f'✅ Created test run: {run_info.run_id}')
            print(f'   Experiment ID: {run_info.experiment_id}')
            
            # Log some test parameters
            mlflow.log_param('test_param', 'test_value')
            mlflow.log_metric('test_metric', 0.5)
            
        print('✅ Run completed successfully')
        
        # Verify the run is in the correct experiment
        run_details = client.get_run(run_info.run_id)
        exp_id = run_details.info.experiment_id
        exp = client.get_experiment(exp_id)
        
        if exp.name == settings.MLFLOW_EXPERIMENT_NAME:
            print(f'✅ Run is in correct experiment: "{exp.name}" (ID: {exp_id})')
            return True
        else:
            print(f'❌ Run is in wrong experiment: "{exp.name}" (expected "{settings.MLFLOW_EXPERIMENT_NAME}")')
            return False
        
    except Exception as e:
        print(f'❌ Error creating run: {e}')
        return False

if __name__ == '__main__':
    success = test_mlflow_experiment_fix()
    
    print('\n' + '=' * 60)
    if success:
        print('🎉 SUCCESS: MLflow experiment setup is working correctly!')
        print('✅ Training runs will now go to "coronary-experiments" experiment')
    else:
        print('❌ FAILED: MLflow experiment setup needs further debugging')
    print('=' * 60)
    
    sys.exit(0 if success else 1)
