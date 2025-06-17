# Dataset Manager Signals

from django.db.models.signals import post_save, pre_delete
from django.dispatch import receiver
from django.contrib.auth.models import User
import logging
import os
from .models import Dataset, DatasetSample, PipelineExecution
from .utils import DatasetProcessor

logger = logging.getLogger(__name__)

@receiver(post_save, sender=Dataset)
def dataset_post_save(sender, instance, created, **kwargs):
    """Handle dataset creation and updates"""
    if created:
        logger.info(f"New dataset created: {instance.name} by {instance.created_by}")
        
        # Start automatic processing if file is uploaded
        if instance.file_path and instance.status == 'uploaded':
            try:
                # Initialize processor
                processor = DatasetProcessor(instance)
                
                # Start background processing
                from django.db import transaction
                with transaction.atomic():
                    instance.status = 'processing'
                    instance.save(update_fields=['status'])
                
                # This would typically be run in a background task (Celery)
                # For now, we'll just log it
                logger.info(f"Starting processing for dataset: {instance.name}")
                
            except Exception as e:
                logger.error(f"Error starting dataset processing: {str(e)}")
                instance.status = 'failed'
                instance.error_message = str(e)
                instance.save(update_fields=['status', 'error_message'])

@receiver(pre_delete, sender=Dataset)
def dataset_pre_delete(sender, instance, **kwargs):
    """Clean up files before dataset deletion"""
    try:
        # Remove dataset files
        if instance.file_path and os.path.exists(instance.file_path):
            os.remove(instance.file_path)
            logger.info(f"Removed dataset file: {instance.file_path}")
        
        # Remove extracted files directory
        if instance.extracted_path and os.path.exists(instance.extracted_path):
            import shutil
            shutil.rmtree(instance.extracted_path)
            logger.info(f"Removed extracted directory: {instance.extracted_path}")
            
    except Exception as e:
        logger.error(f"Error cleaning up dataset files: {str(e)}")

@receiver(post_save, sender=DatasetSample)
def dataset_sample_post_save(sender, instance, created, **kwargs):
    """Handle dataset sample creation"""
    if created:
        logger.debug(f"New sample created for dataset {instance.dataset.name}: {instance.file_path}")
        
        # Update dataset sample count
        dataset = instance.dataset
        dataset.sample_count = dataset.datasetsample_set.count()
        dataset.save(update_fields=['sample_count'])

@receiver(pre_delete, sender=DatasetSample)
def dataset_sample_pre_delete(sender, instance, **kwargs):
    """Clean up sample files before deletion"""
    try:
        # Remove sample file
        if instance.file_path and os.path.exists(instance.file_path):
            os.remove(instance.file_path)
            
        # Remove thumbnail
        if instance.thumbnail_path and os.path.exists(instance.thumbnail_path):
            os.remove(instance.thumbnail_path)
            
        logger.debug(f"Cleaned up sample files for: {instance.file_path}")
        
    except Exception as e:
        logger.error(f"Error cleaning up sample files: {str(e)}")

@receiver(post_save, sender=PipelineExecution)
def pipeline_execution_post_save(sender, instance, created, **kwargs):
    """Handle pipeline execution status changes"""
    if created:
        logger.info(f"New pipeline execution started: {instance.pipeline.name} by {instance.executed_by}")
    
    # Log status changes
    if not created:
        logger.info(f"Pipeline execution {instance.id} status changed to: {instance.status}")
        
        # Send notifications for completion/failure
        if instance.status in ['completed', 'failed']:
            # This is where you would send notifications to users
            # For example, via email, websocket, or push notifications
            logger.info(f"Pipeline execution {instance.id} finished with status: {instance.status}")

@receiver(pre_delete, sender=PipelineExecution)
def pipeline_execution_pre_delete(sender, instance, **kwargs):
    """Clean up execution artifacts before deletion"""
    try:
        # Remove execution logs if they exist as files
        if hasattr(instance, 'log_file_path') and instance.log_file_path:
            if os.path.exists(instance.log_file_path):
                os.remove(instance.log_file_path)
                logger.info(f"Removed execution log file: {instance.log_file_path}")
                
    except Exception as e:
        logger.error(f"Error cleaning up execution files: {str(e)}")

# Custom signal for dataset processing completion
from django.dispatch import Signal

dataset_processing_complete = Signal()
dataset_processing_failed = Signal()

@receiver(dataset_processing_complete)
def handle_dataset_processing_complete(sender, dataset, **kwargs):
    """Handle successful dataset processing completion"""
    logger.info(f"Dataset processing completed successfully: {dataset.name}")
    
    # Update dataset status
    dataset.status = 'ready'
    dataset.save(update_fields=['status'])
    
    # You could add more logic here, such as:
    # - Sending notifications
    # - Triggering automatic model training
    # - Updating related models

@receiver(dataset_processing_failed)
def handle_dataset_processing_failed(sender, dataset, error, **kwargs):
    """Handle failed dataset processing"""
    logger.error(f"Dataset processing failed: {dataset.name} - {error}")
    
    # Update dataset status
    dataset.status = 'failed'
    dataset.error_message = str(error)
    dataset.save(update_fields=['status', 'error_message'])
