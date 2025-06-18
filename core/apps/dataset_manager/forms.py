# Dataset Manager Forms

from django import forms
from django.contrib.auth.models import User
from django.db.models import Q
from .models import DataPipeline, AnnotationSchema, Dataset

class DataPipelineForm(forms.ModelForm):
    """Form for creating and editing data pipelines"""
    
    class Meta:
        model = DataPipeline
        fields = [
            'name', 'description', 'version', 'output_format',
            'input_formats', 'default_parameters', 'required_parameters',
            'is_template', 'is_public'
        ]
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter pipeline name'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Describe what this pipeline does...'
            }),
            'version': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '1.0'
            }),
            'output_format': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g., processed_dataset, csv, json'
            }),
            'is_template': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            }),
            'is_public': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            }),
        }
    
    # JSON fields as text areas for easier editing
    input_formats = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': '["zip", "folder", "csv"]'
        }),
        help_text="JSON array of supported input formats",
        required=False
    )
    
    default_parameters = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 5,
            'placeholder': '{"batch_size": 32, "quality": "high"}'
        }),
        help_text="JSON object with default parameters",
        required=False
    )
    
    required_parameters = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': '["input_path", "output_path"]'
        }),
        help_text="JSON array of required parameter names",
        required=False
    )
    
    # Pipeline graph as a hidden field (will be populated by visual editor)
    pipeline_graph = forms.CharField(
        widget=forms.HiddenInput(),
        required=False
    )
    
    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        
        # Pre-populate JSON fields if editing
        if self.instance.pk:
            self.fields['input_formats'].initial = str(self.instance.input_formats) if self.instance.input_formats else '[]'
            self.fields['default_parameters'].initial = str(self.instance.default_parameters) if self.instance.default_parameters else '{}'
            self.fields['required_parameters'].initial = str(self.instance.required_parameters) if self.instance.required_parameters else '[]'
            self.fields['pipeline_graph'].initial = str(self.instance.pipeline_graph) if self.instance.pipeline_graph else '{}'
    
    def clean_input_formats(self):
        """Validate and parse input_formats JSON"""
        data = self.cleaned_data['input_formats']
        if not data.strip():
            return []
        try:
            import json
            parsed = json.loads(data)
            if not isinstance(parsed, list):
                raise forms.ValidationError("Input formats must be a JSON array")
            return parsed
        except json.JSONDecodeError:
            raise forms.ValidationError("Invalid JSON format for input formats")
    
    def clean_default_parameters(self):
        """Validate and parse default_parameters JSON"""
        data = self.cleaned_data['default_parameters']
        if not data.strip():
            return {}
        try:
            import json
            parsed = json.loads(data)
            if not isinstance(parsed, dict):
                raise forms.ValidationError("Default parameters must be a JSON object")
            return parsed
        except json.JSONDecodeError:
            raise forms.ValidationError("Invalid JSON format for default parameters")
    
    def clean_required_parameters(self):
        """Validate and parse required_parameters JSON"""
        data = self.cleaned_data['required_parameters']
        if not data.strip():
            return []
        try:
            import json
            parsed = json.loads(data)
            if not isinstance(parsed, list):
                raise forms.ValidationError("Required parameters must be a JSON array")
            return parsed
        except json.JSONDecodeError:
            raise forms.ValidationError("Invalid JSON format for required parameters")
    
    def clean_pipeline_graph(self):
        """Validate and parse pipeline_graph JSON"""
        data = self.cleaned_data['pipeline_graph']
        if not data.strip():
            return {'nodes': [], 'edges': []}
        try:
            import json
            parsed = json.loads(data)
            if not isinstance(parsed, dict):
                raise forms.ValidationError("Pipeline graph must be a JSON object")
            # Ensure required structure
            if 'nodes' not in parsed:
                parsed['nodes'] = []
            if 'edges' not in parsed:
                parsed['edges'] = []
            return parsed
        except json.JSONDecodeError:
            raise forms.ValidationError("Invalid JSON format for pipeline graph")
    
    def save(self, commit=True):
        """Save the pipeline with the current user as creator"""
        pipeline = super().save(commit=False)
        if self.user and not pipeline.created_by_id:
            pipeline.created_by = self.user
        if commit:
            pipeline.save()
        return pipeline

class AnnotationSchemaForm(forms.ModelForm):
    """Form for creating and editing annotation schemas"""
    
    class Meta:
        model = AnnotationSchema
        fields = [
            'name', 'type', 'description', 'is_public'
        ]
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter schema name'
            }),
            'type': forms.Select(attrs={
                'class': 'form-control'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Describe this annotation schema...'
            }),
            'is_public': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            }),
        }
    
    # JSON fields as text areas
    schema_definition = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 8,
            'placeholder': '{"properties": {"class": {"type": "string"}}}'
        }),
        help_text="JSON schema definition",
        required=False
    )
    
    validation_rules = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 5,
            'placeholder': '{"required": ["class"], "min_length": 1}'
        }),
        help_text="JSON validation rules",
        required=False
    )
    
    example_annotation = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 5,
            'placeholder': '{"class": "example_class", "confidence": 0.95}'
        }),
        help_text="Example annotation for preview",
        required=False
    )
    
    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        
        # Pre-populate JSON fields if editing
        if self.instance.pk:
            self.fields['schema_definition'].initial = str(self.instance.schema_definition) if self.instance.schema_definition else '{}'
            self.fields['validation_rules'].initial = str(self.instance.validation_rules) if self.instance.validation_rules else '{}'
            self.fields['example_annotation'].initial = str(self.instance.example_annotation) if self.instance.example_annotation else '{}'
    
    def clean_schema_definition(self):
        """Validate and parse schema_definition JSON"""
        data = self.cleaned_data['schema_definition']
        if not data.strip():
            return {}
        try:
            import json
            parsed = json.loads(data)
            if not isinstance(parsed, dict):
                raise forms.ValidationError("Schema definition must be a JSON object")
            return parsed
        except json.JSONDecodeError:
            raise forms.ValidationError("Invalid JSON format for schema definition")
    
    def clean_validation_rules(self):
        """Validate and parse validation_rules JSON"""
        data = self.cleaned_data['validation_rules']
        if not data.strip():
            return {}
        try:
            import json
            parsed = json.loads(data)  
            if not isinstance(parsed, dict):
                raise forms.ValidationError("Validation rules must be a JSON object")
            return parsed
        except json.JSONDecodeError:
            raise forms.ValidationError("Invalid JSON format for validation rules")
    
    def clean_example_annotation(self):
        """Validate and parse example_annotation JSON"""
        data = self.cleaned_data['example_annotation']
        if not data.strip():
            return {}
        try:
            import json
            parsed = json.loads(data)
            return parsed
        except json.JSONDecodeError:
            raise forms.ValidationError("Invalid JSON format for example annotation")
    
    def save(self, commit=True):
        """Save the schema with the current user as creator"""
        schema = super().save(commit=False)
        if self.user and not schema.created_by_id:
            schema.created_by = self.user
        if commit:
            schema.save()
        return schema

class DatasetUploadForm(forms.ModelForm):
    """Form for uploading and creating datasets"""
    
    class Meta:
        model = Dataset
        fields = [
            'name', 'description', 'version', 'format_type', 
            'annotation_schema'
        ]
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter dataset name'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Describe this dataset...'
            }),
            'version': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '1.0'
            }),
            'format_type': forms.Select(attrs={
                'class': 'form-control'
            }),
            'annotation_schema': forms.Select(attrs={
                'class': 'form-control'
            }),
        }
    
    # File upload field
    dataset_file = forms.FileField(
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.zip,.tar,.tar.gz,.csv'
        }),
        help_text="Upload ZIP, TAR, or CSV file containing your dataset"
    )
    
    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        
        # Filter schemas to user's schemas and public ones
        if self.user:
            self.fields['annotation_schema'].queryset = AnnotationSchema.objects.filter(
                Q(created_by=self.user) | Q(is_public=True)
            ).distinct()
        
        # Make annotation_schema optional
        self.fields['annotation_schema'].required = False
        self.fields['annotation_schema'].empty_label = "Detect automatically"
    
    def save(self, commit=True):
        """Save the dataset with the current user as creator"""
        dataset = super().save(commit=False)
        if self.user and not dataset.created_by_id:
            dataset.created_by = self.user
        
        # Set initial status
        dataset.status = 'uploading'
        
        if commit:
            dataset.save()
        return dataset
