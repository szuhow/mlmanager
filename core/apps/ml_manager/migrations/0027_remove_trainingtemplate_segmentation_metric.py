# Generated manually on 2025-07-14

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ml_manager', '0026_mlmodel_primary_metric_name_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='trainingtemplate',
            name='segmentation_metric',
        ),
    ]
