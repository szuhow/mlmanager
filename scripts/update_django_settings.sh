#!/bin/bash
# Script to update all Django settings references to the new modular structure

echo "🔧 Updating Django settings references across the codebase..."

# Files to update with their patterns
files_to_update=(
    "tests/e2e/test_comprehensive_mlflow_fixes.py"
    "tests/e2e/test_complete_training_workflow.py"
    "tests/fixtures/debug_device_detection.py"
    "tests/unit/test_mlflow_fixes_final.py"
    "tests/unit/test_unet_creation.py"
    "tests/unit/test_comprehensive_mlflow_fixes.py"
    "tests/unit/test_javascript_fixes.py"
    "tests/integration/test_mlflow_fixes_final.py"
    "tests/unit/test_mlflow_experiment_fix.py"
)

# Update Django settings module references
for file in "${files_to_update[@]}"; do
    if [ -f "$file" ]; then
        echo "📝 Updating $file..."
        
        # Update settings module references
        sed -i '' "s/coronary_experiments\.settings/core.config.settings.testing/g" "$file"
        sed -i '' "s/core\.coronary_experiments\.settings/core.config.settings.testing/g" "$file"
        
        # Update import paths
        sed -i '' "s/from ml_manager\.models/from core.apps.ml_manager.models/g" "$file"
        sed -i '' "s/from shared\.utils/from ml.utils.utils/g" "$file"
        sed -i '' "s/from shared\.train/from ml.training.train/g" "$file"
        
        echo "✅ Updated $file"
    else
        echo "⚠️  File not found: $file"
    fi
done

echo "🎉 Django settings update completed!"
