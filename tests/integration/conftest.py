"""
Configuration for integration tests
"""
import pytest
import docker
import os

@pytest.fixture(scope="session")
def docker_client():
    """Docker client for integration tests"""
    return docker.from_env()

@pytest.fixture(scope="session") 
def mlflow_server():
    """MLflow server fixture for integration tests"""
    # Setup code for MLflow test server
    yield "http://localhost:5000"
    # Teardown code

@pytest.fixture
def test_container_config():
    """Container configuration for tests"""
    return {
        'image': 'test_image',
        'ports': {'8000/tcp': None},
        'environment': {
            'DJANGO_SETTINGS_MODULE': 'core.config.settings.testing'
        }
    }
