from django.core.management.base import BaseCommand
from django.conf import settings
from ...models import MLModel
import os
import shutil
import mlflow


class Command(BaseCommand):
    help = 'Clean up orphaned MLflow files and artifacts not associated with any model'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Only show what would be deleted without actually deleting',
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Delete files without confirmation',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        force = options['force']
        
        self.stdout.write("🔍 Searching for orphaned MLflow files...")
        
        # Get all MLflow run IDs from database
        active_runs = set(MLModel.objects.exclude(mlflow_run_id__isnull=True).values_list('mlflow_run_id', flat=True))
        self.stdout.write(f"📊 Found {len(active_runs)} active MLflow runs in database")
        
        # Scan MLflow directory for all runs
        mlflow_dir = os.path.join(settings.BASE_DIR, 'data', 'mlflow')
        if not os.path.exists(mlflow_dir):
            self.stdout.write(self.style.WARNING("❌ MLflow directory not found"))
            return
        
        all_runs = []
        orphaned_runs = []
        total_size = 0
        
        for item in os.listdir(mlflow_dir):
            item_path = os.path.join(mlflow_dir, item)
            if os.path.isdir(item_path) and len(item) == 32:  # MLflow run ID format
                all_runs.append(item)
                
                # Calculate size
                dir_size = self.get_directory_size(item_path)
                total_size += dir_size
                
                if item not in active_runs:
                    orphaned_runs.append({
                        'run_id': item,
                        'path': item_path,
                        'size': dir_size
                    })
        
        self.stdout.write(f"📊 Found {len(all_runs)} total MLflow run directories")
        self.stdout.write(f"🗑️  Found {len(orphaned_runs)} orphaned runs")
        self.stdout.write(f"💾 Total MLflow storage: {self.format_size(total_size)}")
        
        if not orphaned_runs:
            self.stdout.write(self.style.SUCCESS("✅ No orphaned files found!"))
            return
        
        # Calculate potential savings
        orphaned_size = sum(run['size'] for run in orphaned_runs)
        self.stdout.write(f"💰 Potential space savings: {self.format_size(orphaned_size)}")
        
        # Show detailed list
        self.stdout.write("\\n📋 Orphaned runs:")
        for run in orphaned_runs[:10]:  # Show first 10
            self.stdout.write(f"   🗑️  {run['run_id']} - {self.format_size(run['size'])}")
        
        if len(orphaned_runs) > 10:
            self.stdout.write(f"   ... and {len(orphaned_runs) - 10} more")
        
        if dry_run:
            self.stdout.write(f"\\n🔧 [DRY RUN] Would delete {len(orphaned_runs)} orphaned runs")
            self.stdout.write(f"🔧 [DRY RUN] Would free up {self.format_size(orphaned_size)}")
            return
        
        # Confirm deletion
        if not force:
            confirm = input(f"\\n❓ Delete {len(orphaned_runs)} orphaned runs? (y/N): ")
            if confirm.lower() != 'y':
                self.stdout.write("❌ Deletion cancelled")
                return
        
        # Delete orphaned runs
        deleted_count = 0
        deleted_size = 0
        
        for run in orphaned_runs:
            try:
                shutil.rmtree(run['path'])
                deleted_count += 1
                deleted_size += run['size']
                self.stdout.write(f"✅ Deleted: {run['run_id']}")
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"❌ Failed to delete {run['run_id']}: {e}"))
        
        self.stdout.write(f"\\n🎉 Cleanup completed!")
        self.stdout.write(f"✅ Deleted {deleted_count} orphaned runs")
        self.stdout.write(f"💾 Freed up {self.format_size(deleted_size)}")
        
        # Show remaining storage
        remaining_size = total_size - deleted_size
        self.stdout.write(f"📊 Remaining MLflow storage: {self.format_size(remaining_size)}")

    def get_directory_size(self, path):
        """Calculate total size of directory"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except (OSError, IOError):
                        pass
        except (OSError, IOError):
            pass
        return total_size

    def format_size(self, size_bytes):
        """Format bytes to human readable format"""
        if size_bytes == 0:
            return "0 B"
        
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
