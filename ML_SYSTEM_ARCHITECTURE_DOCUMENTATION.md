# ML Manager System - Architektura i Pipeline

## Spis Treści
1. [Przegląd Systemu](#przegląd-systemu)
2. [Architektura Modułowa](#architektura-modułowa)  
3. [Pipeline Trenowania](#pipeline-trenowania)
4. [Pipeline Inferencji](#pipeline-inferencji)
5. [Zarządzanie Danymi](#zarządzanie-danymi)
6. [Monitorowanie i Logowanie](#monitorowanie-i-logowanie)
7. [API i Integracje](#api-i-integracje)
8. [Deployment i Skalowanie](#deployment-i-skalowanie)
9. [Rozwiązywanie Problemów](#rozwiązywanie-problemów)

---

## Przegląd Systemu

### Cel Systemu
ML Manager to system do trenowania, zarządzania i wdrażania modeli uczenia maszynowego do analizy obrazów medycznych (głównie segmentacji naczyń wieńcowych).

### Kluczowe Funkcjonalności
- **Trenowanie modeli**: UNet, ResNet, MONAI models
- **Inferencja**: Batch i real-time processing
- **Zarządzanie danymi**: ARCADE, CADICA datasets
- **Monitoring**: MLflow, real-time metrics
- **API**: REST endpoints dla integracji

### Technologie
- **Backend**: Django 4.x, Celery, Redis
- **ML Framework**: PyTorch, MONAI, scikit-image
- **Database**: PostgreSQL, MLflow tracking
- **Container**: Docker, Docker Compose
- **Frontend**: HTML/CSS/JS, Bootstrap

---

## Architektura Modułowa

### Struktura Katalogów
```
core/apps/ml_manager/
├── models/               # Django models (MLModel, Dataset, etc.)
├── views/               # Django views i API endpoints
├── forms/               # Django forms dla UI
├── tasks/               # Celery tasks (async processing)
├── training/            # Core ML training logic
│   ├── train.py        # Main training functions
│   ├── models/         # Model architectures
│   └── core/           # Utilities, transforms
├── datasets/           # Dataset loaders i preprocessing
├── utils/              # Helper functions
├── templates/          # HTML templates
└── static/            # CSS, JS, images
```

### Klasy Główne

#### MLModel (models.py)
```python
class MLModel(models.Model):
    name = models.CharField(max_length=200)
    model_type = models.CharField(max_length=50)  # 'unet', 'resnet', etc.
    status = models.CharField(max_length=20)      # 'training', 'completed', 'failed'
    
    # Training metrics
    train_loss = models.FloatField(null=True)
    val_loss = models.FloatField(null=True)
    best_val_dice = models.FloatField(null=True)
    best_val_iou = models.FloatField(null=True)
    
    # Paths and metadata
    model_directory = models.CharField(max_length=500)
    mlflow_run_id = models.CharField(max_length=100)
    training_config = models.JSONField(default=dict)
```

#### Dataset Management
```python
class Dataset(models.Model):
    name = models.CharField(max_length=200)
    dataset_type = models.CharField(max_length=50)  # 'arcade', 'cadica'
    data_path = models.CharField(max_length=500)
    total_samples = models.IntegerField(default=0)
    config = models.JSONField(default=dict)
```

---

## Pipeline Trenowania

### 1. Inicjalizacja Treningu
```python
# Entry point: views.py -> StartTrainingView
def form_valid(self, form):
    # 1. Utworzenie MLModel record
    model = MLModel.objects.create(...)
    
    # 2. Przygotowanie parametrów
    training_params = self.prepare_training_params(form)
    
    # 3. Wywołanie Celery task
    train_model_task.delay(model.id, training_params)
```

### 2. Celery Task Orchestration
```python
# tasks/tasks.py
@shared_task(bind=True)
def train_model_task(self, model_id, training_params):
    # 1. Setup environment
    setup_training_environment(model_id)
    
    # 2. Call subprocess for isolation
    result = subprocess.run([
        sys.executable, 'training/train.py',
        '--model-id', str(model_id),
        '--config', json.dumps(training_params)
    ])
    
    # 3. Process results
    update_model_from_results(model_id, result)
```

### 3. Core Training Logic
```python
# training/train.py
def train_model(args):
    # 1. Load dataset
    train_loader, val_loader = create_data_loaders(args)
    
    # 2. Create model
    model = create_model_from_registry(args.model_type)
    
    # 3. Setup training components
    optimizer = torch.optim.Adam(model.parameters())
    criterion = get_loss_function(args.loss_type)
    
    # 4. Training loop
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion)
        
        # 5. MLflow logging
        mlflow.log_metrics({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_dice': val_metrics['dice']
        })
        
        # 6. Save checkpoints
        if val_metrics['dice'] > best_dice:
            save_checkpoint(model, epoch, val_metrics)
```

### 4. Model Registry
```python
# training/models/
def create_model_from_registry(model_type, device='cpu', **config):
    """Factory pattern for model creation"""
    if model_type == 'unet':
        return UNet(**config)
    elif model_type == 'configurable_monai_unet':
        return ConfigurableMonaiUNet(**config)
    elif model_type == 'resnet':
        return ResNetModel(**config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

### 5. Data Pipeline
```python
# datasets/torch_arcade_loader.py
class ARCADEBinarySegmentation(Dataset):
    def __init__(self, data_path, transforms=None):
        self.data_path = data_path
        self.transforms = transforms
        self.samples = self._load_samples()
    
    def __getitem__(self, idx):
        image, mask = self._load_image_mask(idx)
        
        if self.transforms:
            transformed = self.transforms({
                'image': image,
                'label': mask
            })
            return transformed['image'], transformed['label']
        
        return image, mask
```

### 6. Transforms Pipeline
```python
# training/train.py
def get_monai_transforms(params, for_training=True):
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelTransform(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
    ]
    
    # Medical preprocessing
    if params.get('use_medical_preprocessing'):
        transforms.append(Lambdad(
            keys=["image"], 
            func=partial(medical_preprocessing_wrapper, ...)
        ))
    
    # Training augmentations
    if for_training:
        transforms.extend([
            RandFlipd(keys=["image", "label"], prob=0.5),
            RandRotate90d(keys=["image", "label"], prob=0.5),
            RandSpatialCropd(keys=["image", "label"], ...)
        ])
    
    return Compose(transforms)
```

---

## Pipeline Inferencji

### 1. Inferencja Request Flow
```python
# views.py -> GeneralInferenceView
def form_valid(self, form):
    # 1. Create InferenceResult record
    inference_result = InferenceResult.objects.create(...)
    
    # 2. Queue inference task
    run_inference_task.delay(model_id, image_path, config)
    
    # 3. Return immediate response
    return redirect('inference_result', pk=inference_result.id)
```

### 2. Inference Processing
```python
# tasks/tasks.py
@shared_task
def run_inference_task(model_id, image_path, config):
    # 1. Load model
    model = load_model_from_checkpoint(model_id)
    
    # 2. Preprocess image
    image = preprocess_image(image_path, config)
    
    # 3. Run inference
    with torch.no_grad():
        prediction = model(image)
    
    # 4. Post-process results
    processed_mask = post_process_prediction(prediction, config)
    
    # 5. Generate visualizations
    save_inference_results(processed_mask, config)
```

### 3. Model Loading Strategy
```python
# utils/enhanced_inference.py
def load_model_from_checkpoint(model_path):
    # 1. Detect checkpoint format
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 2. Extract metadata
    metadata = checkpoint.get('model_metadata', {})
    
    # 3. Create model with correct architecture
    model = create_model_from_registry(
        model_type=metadata['model_type'],
        **metadata['model_config']
    )
    
    # 4. Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model
```

### 4. Post-processing Pipeline
```python
# utils/enhanced_inference.py
def apply_post_processing(mask, config):
    """Enhanced post-processing with configurable options"""
    
    # Morphological operations
    if config.get('apply_opening'):
        mask = morphology.opening(mask, disk(config['kernel_size']))
    
    if config.get('apply_closing'):
        mask = morphology.closing(mask, disk(config['kernel_size']))
    
    # Component filtering
    if config.get('min_component_size') > 0:
        mask = morphology.remove_small_objects(
            mask.astype(bool), 
            min_size=config['min_component_size']
        )
    
    # Confidence scoring
    confidence_scores = calculate_confidence_scores(mask)
    
    return mask, confidence_scores
```

---

## Zarządzanie Danymi

### 1. Dataset Organization
```
data/
├── datasets/
│   ├── arcade/
│   │   ├── images/
│   │   ├── masks/
│   │   └── metadata.json
│   └── cadica/
│       ├── raw/
│       ├── processed/
│       └── annotations/
├── models/
│   └── organized/
│       └── 2025/07/
└── mlflow/
    └── [run_ids]/
        └── artifacts/
```

### 2. Data Loaders
```python
# datasets/torch_arcade_loader.py
def create_arcade_dataloader(
    data_path,
    batch_size=8,
    num_workers=4,
    transforms=None,
    dataset_type='binary_segmentation'
):
    dataset_class = {
        'binary_segmentation': ARCADEBinarySegmentation,
        'artery_classification': ARCADEArteryClassification,
        'multiclass_segmentation': ARCADEMulticlassSegmentation
    }[dataset_type]
    
    dataset = dataset_class(data_path, transforms=transforms)
    
    # Important: Use num_workers=0 if transforms contain non-picklable objects
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader
```

### 3. Medical Preprocessing
```python
# training/train.py (global scope for pickling)
def medical_preprocessing_wrapper(image_array, preprocessing_type, preprocessing_params):
    """Global wrapper for medical preprocessing - pickle-able"""
    return apply_medical_preprocessing(
        image_array, 
        preprocessing_type, 
        **preprocessing_params
    )

def apply_medical_preprocessing(image, preprocessing_type, **params):
    """Apply medical-specific image preprocessing"""
    
    if preprocessing_type == 'angiography':
        # CLAHE for contrast enhancement
        if params.get('use_clahe', True):
            image = apply_clahe(image, 
                              clip_limit=params.get('clahe_clip_limit', 3.0))
        
        # Vessel enhancement
        if params.get('use_frangi', False):
            image = apply_frangi_filter(image, 
                                      scale_range=params.get('frangi_scale_range'))
        
        # Denoising
        if params.get('use_denoising', False):
            image = denoise_image(image, 
                                variance=params.get('noise_variance', 0.1))
    
    return image
```

---

## Monitorowanie i Logowanie

### 1. MLflow Integration
```python
# training/train.py
def setup_mlflow_tracking(model_id, params):
    """Setup MLflow tracking for experiment"""
    
    # Set experiment
    experiment_name = f"model_{model_id}"
    mlflow.set_experiment(experiment_name)
    
    # Start run
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params(params)
        
        # Log model architecture
        mlflow.log_text(str(model), "model_architecture.txt")
        
        return run.info.run_id

def log_training_metrics(epoch, metrics):
    """Log training metrics to MLflow"""
    mlflow.log_metrics({
        'epoch': epoch,
        'train_loss': metrics['train_loss'],
        'val_loss': metrics['val_loss'],
        'val_dice': metrics['val_dice'],
        'val_iou': metrics['val_iou'],
        'learning_rate': metrics['lr']
    }, step=epoch)

def save_model_artifacts(model, epoch, metrics, model_dir):
    """Save model and artifacts to MLflow"""
    
    # Save model checkpoint
    checkpoint_path = f"{model_dir}/epoch_{epoch}_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'metrics': metrics,
        'model_metadata': {
            'model_type': 'unet',
            'input_channels': 1,
            'num_classes': 1
        }
    }, checkpoint_path)
    
    # Log to MLflow
    mlflow.log_artifact(checkpoint_path, "checkpoints")
    
    # If best model, save as final
    if metrics['val_dice'] == best_dice:
        mlflow.log_artifact(checkpoint_path, "final_model")
```

### 2. Real-time Monitoring
```python
# utils/monitoring.py
class TrainingMonitor:
    def __init__(self, model_id):
        self.model_id = model_id
        self.start_time = time.time()
        
    def log_epoch_metrics(self, epoch, metrics):
        """Log metrics and update database"""
        
        # Update Django model
        model = MLModel.objects.get(id=self.model_id)
        model.current_epoch = epoch
        model.train_loss = metrics['train_loss']
        model.val_loss = metrics['val_loss']
        model.val_dice = metrics['val_dice']
        
        # Update best metrics
        if metrics['val_dice'] > (model.best_val_dice or 0):
            model.best_val_dice = metrics['val_dice']
            model.best_epoch = epoch
        
        model.save()
        
        # Log to MLflow
        mlflow.log_metrics(metrics, step=epoch)
        
    def log_system_metrics(self):
        """Log system resource usage"""
        import psutil
        
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        
        metrics = {
            'cpu_usage': cpu_percent,
            'memory_usage': memory_info.percent,
            'memory_available': memory_info.available / 1024**3  # GB
        }
        
        logger.info(f"Model {self.model_id} - CPU: {cpu_percent}%, "
                   f"Memory: {memory_info.used / 1024**2:.2f} MB")
        
        return metrics
```

### 3. Error Handling i Recovery
```python
# tasks/tasks.py
def handle_training_failure(model_id, error):
    """Handle training failure and cleanup"""
    
    try:
        model = MLModel.objects.get(id=model_id)
        model.status = 'failed'
        model.error_message = str(error)
        model.save()
        
        # Log error to MLflow
        if hasattr(model, 'mlflow_run_id') and model.mlflow_run_id:
            with mlflow.start_run(run_id=model.mlflow_run_id):
                mlflow.log_param("error", str(error))
                mlflow.set_tag("status", "failed")
        
        # Cleanup resources
        cleanup_training_resources(model_id)
        
    except Exception as cleanup_error:
        logger.error(f"Error during failure handling: {cleanup_error}")

def cleanup_training_resources(model_id):
    """Cleanup training resources"""
    
    # Stop monitoring threads
    if model_id in training_monitors:
        training_monitors[model_id].stop()
        del training_monitors[model_id]
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Remove temporary files
    temp_dir = f"/tmp/training_{model_id}"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
```

---

## API i Integracje

### 1. REST API Endpoints
```python
# views.py
class ModelListAPIView(APIView):
    """API endpoint for listing models"""
    
    def get(self, request):
        models = MLModel.objects.filter(status='completed')
        serializer = MLModelSerializer(models, many=True)
        return Response(serializer.data)

class InferenceAPIView(APIView):
    """API endpoint for running inference"""
    
    def post(self, request):
        serializer = InferenceRequestSerializer(data=request.data)
        if serializer.is_valid():
            # Queue inference task
            task = run_inference_task.delay(
                serializer.validated_data['model_id'],
                serializer.validated_data['image'],
                serializer.validated_data['config']
            )
            
            return Response({
                'task_id': task.id,
                'status': 'queued'
            })
        
        return Response(serializer.errors, status=400)

# URLs configuration
urlpatterns = [
    path('api/models/', ModelListAPIView.as_view(), name='api-models'),
    path('api/inference/', InferenceAPIView.as_view(), name='api-inference'),
    path('api/inference/<task_id>/status/', InferenceStatusAPIView.as_view()),
]
```

### 2. Celery Task Status API
```python
# views.py
def get_task_status(request, task_id):
    """Get Celery task status"""
    
    from celery.result import AsyncResult
    result = AsyncResult(task_id)
    
    response_data = {
        'task_id': task_id,
        'status': result.status,
    }
    
    if result.status == 'SUCCESS':
        response_data['result'] = result.result
    elif result.status == 'FAILURE':
        response_data['error'] = str(result.result)
    elif result.status == 'PROGRESS':
        response_data['progress'] = result.result
    
    return JsonResponse(response_data)
```

### 3. HTTP Polling dla Real-time Updates

System wykorzystuje inteligentny HTTP polling zamiast WebSockets dla real-time updates:

```python
# views.py
def get_realtime_logs(request, model_id):
    """Real-time log endpoint with smart polling support"""
    try:
        model = get_object_or_404(MLModel, id=model_id)
        
        # Get parameters for efficient polling
        last_timestamp = request.GET.get('since', '')
        lines_limit = int(request.GET.get('lines', 50))
        
        # Find and read log file
        log_path = find_model_log_path(model)
        if log_path and os.path.exists(log_path):
            logs = read_logs_since_timestamp(log_path, last_timestamp, lines_limit)
            return JsonResponse({
                'logs': logs,
                'status': model.status,
                'last_update': get_current_timestamp()
            })
        
        return JsonResponse({'logs': [], 'status': model.status})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
```

```javascript
// model_progress.js - Frontend polling logic
class ModelProgressUpdater {
    constructor() {
        this.updateInterval = 5000; // 5 sekund dla normalnych modeli
        this.fastUpdateInterval = 2000; // 2 sekundy dla pending modeli
        this.pendingModels = new Set();
    }

    async updateProgress() {
        const trainingRows = document.querySelectorAll('tr[data-model-status="training"]');
        const pendingRows = document.querySelectorAll('tr[data-model-status="pending"]');
        
        // Dynamiczne dostosowanie częstotliwości
        if (pendingRows.length > 0) {
            this.currentInterval = this.fastUpdateInterval;
        } else {
            this.currentInterval = this.updateInterval;
        }
        
        // Aktualizacja każdego modelu
        const allRows = [...trainingRows, ...pendingRows];
        for (const row of allRows) {
            await this.updateSingleModel(row.dataset.modelId);
        }
    }
}
```

---

## Deployment i Skalowanie

### 1. Docker Configuration
```yaml
# docker-compose.cpu.yml
version: '3.8'
services:
  django:
    build: .
    volumes:
      - ./core:/app/core
      - ./data:/app/core/data
    environment:
      - DJANGO_SETTINGS_MODULE=core.config.settings.container
    depends_on:
      - db
      - redis

  training-worker:
    build: .
    command: celery -A core.config.celery worker -Q training --loglevel=info
    volumes:
      - ./core:/app/core
      - ./data:/app/core/data
    environment:
      - DJANGO_SETTINGS_MODULE=core.config.settings.container
    depends_on:
      - db
      - redis

  default-worker:
    build: .
    command: celery -A core.config.celery worker -Q default --loglevel=info
    volumes:
      - ./core:/app/core
      - ./data:/app/core/data

  db:
    image: postgres:13
    environment:
      POSTGRES_DB: coronary_experiments
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres

  redis:
    image: redis:7-alpine
```

### 2. Makefile Commands
```makefile
# Makefile
start-cpu: ## Start CPU version
	docker-compose -f docker-compose.cpu.yml up -d

stop-cpu: ## Stop CPU version
	docker-compose -f docker-compose.cpu.yml down

restart-cpu: ## Restart CPU version
	docker-compose -f docker-compose.cpu.yml restart

logs-django-cpu: ## Show Django logs
	docker-compose -f docker-compose.cpu.yml logs -f django

logs-training-cpu: ## Show Training worker logs
	docker-compose -f docker-compose.cpu.yml logs -f training-worker
```

### 3. Environment Configuration
```python
# settings/container.py
import os

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ.get('POSTGRES_DB', 'coronary_experiments'),
        'USER': os.environ.get('POSTGRES_USER', 'postgres'),
        'PASSWORD': os.environ.get('POSTGRES_PASSWORD', 'postgres'),
        'HOST': os.environ.get('POSTGRES_HOST', 'db'),
        'PORT': os.environ.get('POSTGRES_PORT', '5432'),
    }
}

# Celery
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://redis:6379/0')
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://redis:6379/0')

# MLflow
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'file:///app/core/data/mlflow')
```

---

## Rozwiązywanie Problemów

### 1. Częste Problemy i Rozwiązania

#### Problem: "Can't pickle local object"
```python
# ❌ Błędny kod - lokalna funkcja
def get_transforms():
    def local_transform(x):
        return x * 2
    return Compose([Lambda(local_transform)])

# ✅ Poprawny kod - globalna funkcja
def global_transform(x):
    return x * 2

def get_transforms():
    return Compose([Lambda(global_transform)])
```

#### Problem: CUDA Out of Memory
```python
# Rozwiązania:
# 1. Zmniejsz batch_size
# 2. Użyj gradient checkpointing
# 3. Wyczyść CUDA cache

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# W DataLoader ustaw pin_memory=False dla CPU
dataloader = DataLoader(dataset, pin_memory=False)
```

#### Problem: Model nie ładuje się w inferencji
```python
# Sprawdź format checkpointa
checkpoint = torch.load(model_path, map_location='cpu')
print("Checkpoint keys:", list(checkpoint.keys()))

# Użyj strict=False do debugowania
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

### 2. Debugging Tools
```python
# training/train.py
def debug_data_pipeline(dataloader, num_batches=3):
    """Debug data pipeline"""
    
    logger.info("=== DATA PIPELINE DEBUG ===")
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
            
        images, masks = batch['image'], batch['label']
        
        logger.info(f"Batch {i}:")
        logger.info(f"  Images shape: {images.shape}")
        logger.info(f"  Images dtype: {images.dtype}")
        logger.info(f"  Images range: [{images.min():.3f}, {images.max():.3f}]")
        logger.info(f"  Masks shape: {masks.shape}")
        logger.info(f"  Masks unique values: {torch.unique(masks)}")

def debug_model_forward(model, sample_input):
    """Debug model forward pass"""
    
    logger.info("=== MODEL DEBUG ===")
    logger.info(f"Input shape: {sample_input.shape}")
    
    # Hook to capture intermediate outputs
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.shape
        return hook
    
    # Register hooks
    for name, layer in model.named_modules():
        if len(name.split('.')) == 2:  # Only main layers
            layer.register_forward_hook(get_activation(name))
    
    # Forward pass
    with torch.no_grad():
        output = model(sample_input)
    
    logger.info(f"Output shape: {output.shape}")
    logger.info("Layer shapes:")
    for name, shape in activations.items():
        logger.info(f"  {name}: {shape}")
```

### 3. Performance Optimization
```python
# training/train.py
class OptimizedTrainingLoop:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            images, masks = batch['image'], batch['label']
            
            # Move to device
            images = images.cuda(non_blocking=True)
            masks = masks.cuda(non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                predictions = self.model(images)
                loss = self.criterion(predictions, masks)
            
            # Mixed precision backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            # Memory cleanup every N batches
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        return total_loss / len(dataloader)
```

### 4. Monitoring i Alerty
```python
# utils/monitoring.py
class SystemMonitor:
    def __init__(self, alert_threshold=90):
        self.alert_threshold = alert_threshold
        
    def check_system_health(self):
        """Check system health and send alerts"""
        
        import psutil
        
        # Memory usage
        memory = psutil.virtual_memory()
        if memory.percent > self.alert_threshold:
            self.send_alert(f"High memory usage: {memory.percent}%")
        
        # Disk usage
        disk = psutil.disk_usage('/')
        if disk.percent > self.alert_threshold:
            self.send_alert(f"High disk usage: {disk.percent}%")
        
        # GPU memory (if available)
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_used = torch.cuda.memory_allocated(i)
                memory_total = torch.cuda.max_memory_allocated(i)
                usage_percent = (memory_used / memory_total) * 100
                
                if usage_percent > self.alert_threshold:
                    self.send_alert(f"High GPU {i} memory: {usage_percent:.1f}%")
    
    def send_alert(self, message):
        """Send alert notification"""
        logger.warning(f"ALERT: {message}")
        # Add your notification service here (email, Slack, etc.)
```

---

## Podsumowanie

### Kluczowe Zasady Projektowe
1. **Modularność**: Każdy komponent ma jasno określoną odpowiedzialność
2. **Skalowalnośc**: System obsługuje multiple workers i GPU scaling  
3. **Monitorowanie**: Real-time tracking wszystkich aspektów systemu
4. **Error Handling**: Graceful failure recovery i debugging tools
5. **API-First**: Wszystkie funkcjonalności dostępne przez API

### Best Practices
1. **Zawsze używaj globalnych funkcji** w transforms do uniknięcia pickle errors
2. **Implement proper checkpointing** dla długich procesów treningowych  
3. **Use MLflow consistently** dla tracking experiments
4. **Monitor system resources** podczas treningu
5. **Test multiprocessing** w development environment

### Rozszerzenia Systemowe
- **Auto-scaling**: Automatic worker scaling based on queue length
- **Distributed Training**: Multi-GPU i multi-node support
- **Model Serving**: Production API endpoints with load balancing
- **A/B Testing**: Model comparison framework
- **Data Versioning**: Dataset versioning i lineage tracking

---

*Dokumentacja wygenerowana: 16 lipca 2025*
*Wersja systemu: v2.1.0*
