"""
Management command to check direct training system status.
"""
from django.core.management.base import BaseCommand
from core.apps.ml_manager.utils.direct_training import training_manager
from core.apps.ml_manager.models import MLModel


class Command(BaseCommand):
    help = 'Check direct training system status'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--model-id',
            type=int,
            help='Check specific model training status'
        )
        parser.add_argument(
            '--stop-all',
            action='store_true',
            help='Stop any active training'
        )
    
    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('🚀 Direct Training System Status'))
        self.stdout.write('=' * 50)
        
        # Check global status
        is_active = training_manager.is_training_active()
        active_model_id = training_manager.get_active_training_model_id()
        
        if is_active:
            self.stdout.write(
                self.style.WARNING(f'⚡ Training ACTIVE for model {active_model_id}')
            )
            
            try:
                model = MLModel.objects.get(id=active_model_id)
                self.stdout.write(f'   Model name: {model.name}')
                self.stdout.write(f'   Status: {model.status}')
                self.stdout.write(f'   Current epoch: {model.current_epoch}/{model.total_epochs}')
                
                status = training_manager.get_training_status(active_model_id)
                if 'elapsed_time' in status:
                    elapsed = status['elapsed_time']
                    self.stdout.write(f'   Elapsed time: {elapsed:.1f} seconds')
                    
            except MLModel.DoesNotExist:
                self.stdout.write(
                    self.style.ERROR(f'   ❌ Model {active_model_id} not found in database')
                )
        else:
            self.stdout.write(self.style.SUCCESS('✅ No active training'))
        
        # Check specific model if requested
        if options['model_id']:
            model_id = options['model_id']
            self.stdout.write(f'\n📊 Model {model_id} specific status:')
            
            try:
                model = MLModel.objects.get(id=model_id)
                self.stdout.write(f'   Model name: {model.name}')
                self.stdout.write(f'   Database status: {model.status}')
                
                status = training_manager.get_training_status(model_id)
                if status['active']:
                    self.stdout.write(self.style.WARNING('   ⚡ Training manager shows ACTIVE'))
                else:
                    self.stdout.write('   💤 Training manager shows INACTIVE')
                    self.stdout.write(f'   Message: {status.get("message", "N/A")}')
                    
            except MLModel.DoesNotExist:
                self.stdout.write(
                    self.style.ERROR(f'   ❌ Model {model_id} not found')
                )
        
        # Stop all training if requested
        if options['stop_all']:
            if is_active:
                self.stdout.write(f'\n🛑 Stopping training for model {active_model_id}...')
                result = training_manager.stop_training()
                if result['success']:
                    self.stdout.write(self.style.SUCCESS('   ✅ Training stopped successfully'))
                else:
                    self.stdout.write(
                        self.style.ERROR(f'   ❌ Failed to stop training: {result["error"]}')
                    )
            else:
                self.stdout.write('\n💤 No active training to stop')
        
        # Show system info
        self.stdout.write('\n🔧 System Information:')
        self.stdout.write(f'   Training manager instance: {id(training_manager)}')
        self.stdout.write(f'   Thread active: {training_manager.training_thread is not None and training_manager.training_thread.is_alive() if training_manager.training_thread else False}')
        
        # Show database models in training state
        training_models = MLModel.objects.filter(status='training')
        if training_models.exists():
            self.stdout.write(f'\n📚 Models marked as training in database:')
            for model in training_models:
                self.stdout.write(f'   - Model {model.id}: {model.name} (epochs: {model.current_epoch}/{model.total_epochs})')
        else:
            self.stdout.write('\n📚 No models marked as training in database')
        
        self.stdout.write('\n' + '=' * 50)
        self.stdout.write(self.style.SUCCESS('🎯 Direct Training System Check Complete'))
