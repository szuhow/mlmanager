#!/usr/bin/env python3
"""
Management command to re-analyze existing datasets and update their task type detection
"""

from django.core.management.base import BaseCommand
from core.apps.dataset_manager.models import Dataset
from core.apps.dataset_manager.utils import DatasetProcessor
import os


class Command(BaseCommand):
    help = 'Re-analyze existing datasets to update task type detection'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dataset-id',
            type=int,
            help='Re-analyze specific dataset by ID',
        )
        parser.add_argument(
            '--dataset-name',
            type=str,
            help='Re-analyze specific dataset by name (partial match)',
        )
        parser.add_argument(
            '--all',
            action='store_true',
            help='Re-analyze all datasets',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be updated without making changes',
        )

    def handle(self, *args, **options):
        datasets = Dataset.objects.all()
        
        # Filter datasets based on options
        if options['dataset_id']:
            datasets = datasets.filter(id=options['dataset_id'])
        elif options['dataset_name']:
            datasets = datasets.filter(name__icontains=options['dataset_name'])
        elif not options['all']:
            self.stdout.write(
                self.style.ERROR('Please specify --dataset-id, --dataset-name, or --all')
            )
            return
        
        if not datasets.exists():
            self.stdout.write(self.style.WARNING('No datasets found matching criteria'))
            return
        
        self.stdout.write(f'Found {datasets.count()} dataset(s) to analyze')
        
        updated_count = 0
        error_count = 0
        
        for dataset in datasets:
            try:
                self.stdout.write(f'\\nAnalyzing dataset: {dataset.name} (ID: {dataset.id})')
                
                # Check if extracted path exists
                if not dataset.extracted_path or not os.path.exists(dataset.extracted_path):
                    self.stdout.write(
                        self.style.WARNING(f'  Skipping - extracted path not found: {dataset.extracted_path}')
                    )
                    continue
                
                # Create processor and re-analyze
                processor = DatasetProcessor(dataset)
                
                # Store old values for comparison
                old_format = dataset.format_type
                old_structure = dataset.detected_structure.copy() if dataset.detected_structure else {}
                old_task_type = old_structure.get('task_type', 'unknown')
                
                if not options['dry_run']:
                    # Re-analyze the dataset
                    processor._analyze_structure()
                    dataset.refresh_from_db()
                
                # Show changes
                new_format = dataset.format_type
                new_structure = dataset.detected_structure or {}
                new_task_type = new_structure.get('task_type', 'unknown')
                
                self.stdout.write(f'  Format: {old_format} -> {new_format}')
                self.stdout.write(f'  Task Type: {old_task_type} -> {new_task_type}')
                
                if old_task_type != new_task_type or old_format != new_format:
                    if options['dry_run']:
                        self.stdout.write(self.style.SUCCESS('  Would be updated'))
                    else:
                        self.stdout.write(self.style.SUCCESS('  Updated'))
                        updated_count += 1
                else:
                    self.stdout.write('  No changes needed')
                
                # Show structure details
                if new_structure:
                    self.stdout.write(f'  Image count: {new_structure.get("image_count", 0)}')
                    self.stdout.write(f'  File count: {new_structure.get("file_count", 0)}')
                    
                    # Show detected folders
                    children = new_structure.get('children', [])
                    directories = [c for c in children if c.get('type') == 'directory']
                    if directories:
                        self.stdout.write('  Directories found:')
                        for d in directories[:5]:  # Show first 5
                            self.stdout.write(f'    - {d.get("name", "unknown")} ({d.get("image_count", 0)} images)')
                        if len(directories) > 5:
                            self.stdout.write(f'    ... and {len(directories) - 5} more')
                
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'  Error analyzing dataset {dataset.name}: {str(e)}')
                )
                error_count += 1
                continue
        
        # Summary
        self.stdout.write('\\n' + '='*50)
        if options['dry_run']:
            self.stdout.write(self.style.SUCCESS(f'DRY RUN: {updated_count} dataset(s) would be updated'))
        else:
            self.stdout.write(self.style.SUCCESS(f'Successfully updated {updated_count} dataset(s)'))
        
        if error_count > 0:
            self.stdout.write(self.style.ERROR(f'{error_count} dataset(s) had errors'))
        
        self.stdout.write('\\nRe-analysis complete!')
