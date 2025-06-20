from django.core.management.base import BaseCommand
from ...models import MLModel
import psutil


class Command(BaseCommand):
    help = 'Validate and correct orphaned training statuses after container restart'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Only show what would be changed without making actual changes',
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Show detailed process information',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        verbose = options['verbose']
        
        self.stdout.write("🔍 Validating training statuses...")
        
        # Find models with potentially orphaned training status
        orphaned_models = MLModel.objects.filter(status__in=['training', 'loading'])
        
        if not orphaned_models.exists():
            self.stdout.write(self.style.SUCCESS("✅ No potentially orphaned training models found"))
            return
        
        self.stdout.write(f"🔍 Found {orphaned_models.count()} models with training/loading status:")
        
        # Grace period for recently created models
        from django.utils import timezone
        from datetime import timedelta
        grace_period = timedelta(minutes=5)
        now = timezone.now()
        
        corrected_count = 0
        for model in orphaned_models:
            self.stdout.write(f"\n📋 Model {model.id}: {model.name}")
            self.stdout.write(f"   Status: {model.status}")
            self.stdout.write(f"   Created: {model.created_at}")
            self.stdout.write(f"   MLflow Run: {model.mlflow_run_id or 'None'}")
            
            # Check model age
            model_age = now - model.created_at
            if model_age < grace_period:
                self.stdout.write(self.style.WARNING(f"   ⏰ Model created {model_age.total_seconds():.1f}s ago - skipping (grace period)"))
                continue
            
            if self.is_training_process_active(model, verbose):
                self.stdout.write(self.style.SUCCESS("   ✅ Training process is ACTIVE"))
            else:
                self.stdout.write(self.style.WARNING("   ❌ Training process is NOT ACTIVE"))
                
                if dry_run:
                    self.stdout.write(self.style.WARNING(f"   🔧 [DRY RUN] Would change status from '{model.status}' to 'failed'"))
                    corrected_count += 1
                else:
                    old_status = model.status
                    model.status = 'failed'
                    model.save()
                    corrected_count += 1
                    self.stdout.write(self.style.SUCCESS(f"   🔧 CORRECTED: '{old_status}' → 'failed' (age: {model_age.total_seconds():.1f}s)"))
        
        if corrected_count > 0:
            if dry_run:
                self.stdout.write(f"\n🎯 [DRY RUN] Would correct {corrected_count} orphaned training statuses")
            else:
                self.stdout.write(f"\n🎯 Corrected {corrected_count} orphaned training statuses")
        else:
            self.stdout.write("\n✅ All training statuses are valid")

    def is_training_process_active(self, model, verbose=False):
        """Check if a training process is actually running for this model"""
        try:
            found_processes = []
            
            # Check if there are any Python processes running train.py with this model ID
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'python' in proc.info['name'].lower():
                        cmdline = proc.info['cmdline']
                        if (cmdline and 
                            any('train.py' in str(arg) for arg in cmdline) and
                            any(f'--model-id={model.id}' in str(arg) for arg in cmdline)):
                            found_processes.append({
                                'pid': proc.info['pid'],
                                'cmdline': ' '.join(cmdline)
                            })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if found_processes:
                if verbose:
                    self.stdout.write("   🔍 Found active training processes:")
                    for proc_info in found_processes:
                        self.stdout.write(f"      PID {proc_info['pid']}: {proc_info['cmdline'][:100]}...")
                return True
            
            # Check MLflow run status if available
            if model.mlflow_run_id:
                try:
                    import mlflow
                    run_info = mlflow.get_run(model.mlflow_run_id)
                    if verbose:
                        self.stdout.write(f"   🔍 MLflow run status: {run_info.info.status}")
                    
                    # If MLflow run is still RUNNING but no process found, it's likely orphaned
                    if run_info.info.status == 'RUNNING':
                        if verbose:
                            self.stdout.write("   ⚠️  MLflow run is RUNNING but no training process found")
                        return False  # Orphaned MLflow run
                except Exception as e:
                    if verbose:
                        self.stdout.write(f"   ⚠️  Could not check MLflow run: {e}")
            
            return False
            
        except Exception as e:
            # If we can't determine, assume it's active to be safe
            self.stdout.write(self.style.WARNING(f"   ⚠️  Could not verify training process: {e}"))
            return True  # Err on the side of caution
