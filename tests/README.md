# Tests Directory Structure

This directory contains all test files organized by category and testing level.

## Directory Structure

### `/arcade/`
Tests specific to ARCADE dataset functionality:
- Dataset loading and processing
- Mask generation and caching
- COCO annotation handling
- Binary/semantic/stenosis segmentation tests

### `/mlflow/`
Tests for MLflow integration:
- Experiment tracking
- Artifact management
- Model registration
- Path handling and configuration

### `/ui/`
User interface and frontend tests:
- Dropdown functionality
- Safari browser compatibility
- JavaScript components
- HTML template tests

### `/training/`
Machine learning training and model tests:
- Training pipeline tests
- MONAI transforms
- Model callbacks
- Quick training verification

### `/integration/`
Integration tests that span multiple components:
- End-to-end workflows
- Container-based tests
- Cross-component interactions
- System-level testing

### `/unit/`
Unit tests for individual components:
- Forms validation
- Registry functions
- Individual fixes verification
- Component isolation tests

### `/e2e/`
End-to-end tests:
- Complete user workflows
- Browser automation
- Full system integration

### `/fixtures/`
Test data and fixtures:
- Sample datasets
- Mock configurations
- Test artifacts

## Running Tests

### All tests:
```bash
python -m pytest tests/
```

### Specific category:
```bash
python -m pytest tests/arcade/
python -m pytest tests/mlflow/
python -m pytest tests/training/
```

### Integration tests:
```bash
python -m pytest tests/integration/
```

### In Docker container:
```bash
docker-compose exec django python -m pytest tests/
```

## Test Naming Convention

- `test_*.py` - Standard pytest naming
- Category prefixes help identify test scope:
  - `test_arcade_*` - ARCADE dataset tests
  - `test_mlflow_*` - MLflow functionality tests
  - `test_training_*` - Training pipeline tests
  - `test_ui_*` / `test_dropdown_*` / `test_safari_*` - UI tests

## Dependencies

Make sure to install test dependencies:
```bash
pip install -r requirements/test.txt
```
