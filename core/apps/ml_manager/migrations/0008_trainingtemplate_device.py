# Generated by Django 5.2.2 on 2025-06-09 19:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ml_manager', '0007_trainingtemplate_resolution'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainingtemplate',
            name='device',
            field=models.CharField(choices=[('auto', 'Auto (CUDA if available, else CPU)'), ('cpu', 'CPU'), ('cuda', 'CUDA')], default='auto', help_text='Device to use for training. Auto will use CUDA if available.', max_length=20),
        ),
    ]
