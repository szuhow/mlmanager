# Generated by Django 5.2.2 on 2025-06-10 12:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ml_manager', '0010_mlmodel_current_batch_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainingtemplate',
            name='elastic_alpha',
            field=models.FloatField(default=34.0, help_text='Elastic transformation strength'),
        ),
        migrations.AddField(
            model_name='trainingtemplate',
            name='elastic_sigma',
            field=models.FloatField(default=4.0, help_text='Elastic transformation smoothness'),
        ),
        migrations.AddField(
            model_name='trainingtemplate',
            name='flip_probability',
            field=models.FloatField(default=0.5, help_text='Probability of applying flip (0.0-1.0)'),
        ),
        migrations.AddField(
            model_name='trainingtemplate',
            name='intensity_range',
            field=models.FloatField(default=0.2, help_text='Intensity variation range (±range)'),
        ),
        migrations.AddField(
            model_name='trainingtemplate',
            name='noise_std',
            field=models.FloatField(default=0.01, help_text='Standard deviation of Gaussian noise'),
        ),
        migrations.AddField(
            model_name='trainingtemplate',
            name='rotation_range',
            field=models.IntegerField(default=30, help_text='Maximum rotation angle in degrees (±range)'),
        ),
        migrations.AddField(
            model_name='trainingtemplate',
            name='scale_range_max',
            field=models.FloatField(default=1.2, help_text='Maximum scale factor'),
        ),
        migrations.AddField(
            model_name='trainingtemplate',
            name='scale_range_min',
            field=models.FloatField(default=0.8, help_text='Minimum scale factor'),
        ),
        migrations.AddField(
            model_name='trainingtemplate',
            name='use_elastic_transform',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='trainingtemplate',
            name='use_gaussian_noise',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='trainingtemplate',
            name='use_random_crop',
            field=models.BooleanField(default=False),
        ),
    ]
