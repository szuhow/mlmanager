import mlflow
import os

# Try to import Django settings, fall back to environment variables
try:
    from django.conf import settings
    DJANGO_AVAILABLE = True
    def get_mlflow_tracking_uri():
        return settings.MLFLOW_TRACKING_URI
    def get_mlflow_experiment_name():
        return settings.MLFLOW_EXPERIMENT_NAME
except ImportError:
    DJANGO_AVAILABLE = False
    def get_mlflow_tracking_uri():
        return os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db')
    def get_mlflow_experiment_name():
        return os.getenv('MLFLOW_EXPERIMENT_NAME', 'coronary-segmentation')

def setup_mlflow():
    """Setup MLflow tracking"""
    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    
    # Get or create the experiment
    experiment_name = get_mlflow_experiment_name()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        if DJANGO_AVAILABLE:
            from django.conf import settings
            artifact_location = settings.MLFLOW_ARTIFACT_ROOT
        else:
            artifact_location = os.getenv('MLFLOW_ARTIFACT_ROOT', './mlruns')
        mlflow.create_experiment(
            experiment_name,
            artifact_location=artifact_location
        )
    return mlflow.set_experiment(experiment_name)

def get_run(run_id):
    """Get MLflow run by ID"""
    return mlflow.get_run(run_id)

def log_params_and_metrics(run_id, params=None, metrics=None):
    """Log parameters and metrics to an existing run"""
    with mlflow.start_run(run_id=run_id):
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)

def create_new_run(params=None, metrics=None):
    """Create a new MLflow run and log initial parameters and metrics"""
    setup_mlflow()
    with mlflow.start_run() as run:
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        return run.info.run_id

def register_model(run_id, model_name, model_description=None, tags=None):
    """Register a model in MLflow Model Registry"""
    try:
        client = mlflow.tracking.MlflowClient()
        # Ensure the registered model exists, create if not
        try:
            client.get_registered_model(model_name)
        except Exception:
            client.create_registered_model(model_name)
        # Get the model URI from the run
        model_uri = f"runs:/{run_id}/model"
        # Register the model version
        # MLflow <2.0 does not support 'description' in create_model_version, so fallback if needed
        try:
            model_version = client.create_model_version(
                name=model_name,
                source=model_uri,
                description=model_description,
                tags=tags
            )
        except TypeError:
            # Fallback for older MLflow: remove description if not supported
            model_version = client.create_model_version(
                name=model_name,
                source=model_uri,
                tags=tags
            )
            # Update description after creation
            if model_description:
                client.update_model_version(model_name, model_version.version, description=model_description)
        return {
            'name': model_version.name,
            'version': model_version.version,
            'run_id': model_version.run_id,
            'status': model_version.status,
            'creation_timestamp': model_version.creation_timestamp
        }
    except Exception as e:
        print(f"Error registering model: {e}")
        return None

def transition_model_stage(model_name, version, stage, archive_existing_versions=False):
    """Transition a model version to a new stage"""
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Transition to new stage
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing_versions
        )
        
        return True
    except Exception as e:
        print(f"Error transitioning model stage: {e}")
        return False

def get_registered_model_info(model_name):
    """Get information about a registered model"""
    try:
        client = mlflow.tracking.MlflowClient()
        model = client.get_registered_model(model_name)
        
        # Get latest versions
        latest_versions = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])
        
        return {
            'name': model.name,
            'description': model.description,
            'creation_timestamp': model.creation_timestamp,
            'last_updated_timestamp': model.last_updated_timestamp,
            'latest_versions': [
                {
                    'version': v.version,
                    'stage': v.current_stage,
                    'run_id': v.run_id,
                    'status': v.status,
                    'creation_timestamp': v.creation_timestamp
                }
                for v in latest_versions
            ]
        }
    except Exception as e:
        print(f"Error getting registered model info: {e}")
        return None

def list_registered_models():
    """List all registered models"""
    try:
        client = mlflow.tracking.MlflowClient()
        return client.list_registered_models()
    except Exception as e:
        print(f"Error listing registered models: {e}")
        return []

def get_model_version_details(model_name, version):
    """Get detailed information about a specific model version"""
    try:
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_model_version(model_name, version)
        
        return {
            'name': model_version.name,
            'version': model_version.version,
            'creation_timestamp': model_version.creation_timestamp,
            'last_updated_timestamp': model_version.last_updated_timestamp,
            'description': model_version.description,
            'user_id': model_version.user_id,
            'current_stage': model_version.current_stage,
            'source': model_version.source,
            'run_id': model_version.run_id,
            'status': model_version.status,
            'tags': model_version.tags
        }
    except Exception as e:
        print(f"Error getting model version details: {e}")
        return None

def update_model_description(model_name, description):
    """Update the description of a registered model"""
    try:
        client = mlflow.tracking.MlflowClient()
        client.update_registered_model(model_name, description=description)
        return True
    except Exception as e:
        print(f"Error updating model description: {e}")
        return False

def update_model_version_description(model_name, version, description):
    """Update the description of a specific model version"""
    try:
        client = mlflow.tracking.MlflowClient()
        client.update_model_version(model_name, version, description=description)
        return True
    except Exception as e:
        print(f"Error updating model version description: {e}")
        return False
