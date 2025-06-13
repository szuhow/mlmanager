import mlflow
import os

# Try to import Django settings, fall back to environment variables
try:
    from django.conf import settings
    DJANGO_AVAILABLE = True
    def get_mlflow_tracking_uri():
        try:
            return settings.MLFLOW_TRACKING_URI
        except AttributeError:
            return os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
    def get_mlflow_experiment_name():
        try:
            return settings.MLFLOW_EXPERIMENT_NAME
        except AttributeError:
            return os.getenv('MLFLOW_EXPERIMENT_NAME', 'coronary-experiments')
except ImportError:
    DJANGO_AVAILABLE = False
    def get_mlflow_tracking_uri():
        return os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
    def get_mlflow_experiment_name():
        return os.getenv('MLFLOW_EXPERIMENT_NAME', 'coronary-experiments')

def setup_mlflow():
    """Setup MLflow tracking"""
    try:
        tracking_uri = get_mlflow_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)
        print(f"[MLFLOW] Set tracking URI to: {tracking_uri}")
    except Exception as e:
        print(f"[MLFLOW] Failed to setup MLflow experiment: {e}")
        # Fallback to environment variable
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
        mlflow.set_tracking_uri(tracking_uri)
        print(f"[MLFLOW] Fallback tracking URI: {tracking_uri}")
    
    # Get or create the experiment
    try:
        experiment_name = get_mlflow_experiment_name()
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            if DJANGO_AVAILABLE:
                try:
                    from django.conf import settings
                    artifact_location = settings.MLFLOW_ARTIFACT_ROOT
                except:
                    artifact_location = os.getenv('MLFLOW_ARTIFACT_ROOT', './mlruns')
            else:
                artifact_location = os.getenv('MLFLOW_ARTIFACT_ROOT', './mlruns')
            mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location
            )
        return mlflow.set_experiment(experiment_name)
    except Exception as e:
        print(f"[MLFLOW] Failed to setup experiment: {e}")
        # Use default experiment
        return mlflow.set_experiment("Default")

def setup_mlflow_experiment(experiment_name=None):
    """Setup MLflow experiment with optional custom name"""
    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    
    # Use custom name or default
    if experiment_name is None:
        experiment_name = get_mlflow_experiment_name()
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        if DJANGO_AVAILABLE:
            from django.conf import settings
            artifact_location = settings.MLFLOW_ARTIFACT_ROOT
        else:
            artifact_location = os.getenv('MLFLOW_ARTIFACT_ROOT', './mlruns')
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=artifact_location
        )
    else:
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(experiment_name)
    return experiment_id

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
    # Start run but don't use context manager - let training subprocess manage lifecycle
    run = mlflow.start_run()
    try:
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        return run.info.run_id
    except Exception as e:
        # If there's an error, end the run to clean up
        mlflow.end_run()
        raise e

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

def get_mlflow_ui_url(run_id=None, experiment_name=None):
    """Generate MLflow UI URL for a specific run or experiment"""
    try:
        tracking_uri = get_mlflow_tracking_uri()
        
        # Parse the tracking URI to get the base URL
        if tracking_uri.startswith('http'):
            # For Docker environment, convert internal network URL to external access URL
            if 'mlflow:5000' in tracking_uri:
                # In Docker, MLflow UI is accessible via localhost:5000 from the host
                base_url = 'http://localhost:5000'
            else:
                base_url = tracking_uri
        else:
            # For file-based tracking, assume local MLflow server
            base_url = 'http://localhost:5000'
        
        if run_id:
            # URL for specific run
            try:
                client = mlflow.tracking.MlflowClient()
                run = client.get_run(run_id)
                experiment_id = run.info.experiment_id
                return f"{base_url}/#/experiments/{experiment_id}/runs/{run_id}"
            except Exception as e:
                print(f"Error getting run info for URL generation: {e}")
                return f"{base_url}/#/experiments"
        
        elif experiment_name:
            # URL for experiment
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment:
                    return f"{base_url}/#/experiments/{experiment.experiment_id}"
                else:
                    return f"{base_url}/#/experiments"
            except Exception as e:
                print(f"Error getting experiment info for URL generation: {e}")
                return f"{base_url}/#/experiments"
        
        else:
            # Default to experiments list
            return f"{base_url}/#/experiments"
            
    except Exception as e:
        print(f"Error generating MLflow URL: {e}")
        return None

def generate_mlflow_ui_url(run_id=None, experiment_name=None):
    """Alias for get_mlflow_ui_url for backward compatibility"""
    return get_mlflow_ui_url(run_id=run_id, experiment_name=experiment_name)
