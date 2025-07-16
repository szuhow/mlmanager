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

Proces treningu składa się z trzech warstw:

1. **Warstwa widoku (Django)** – `StartTrainingView` w `views.py` po walidacji formularza:
   • tworzy rekord `MLModel` w bazie,
   • buduje słownik `training_config` z parametrów formularza,
   • wywołuje `MLTrainingService.start_training()`.

2. **Warstwa usługowa** – `MLTrainingService` (`services/training_service.py`):
   • aktualizuje status modelu na `training`,
   • uruchamia zadanie Celery `train_model_task.delay(model_id, training_config)` w kolejce `training`.

3. **Warstwa asynchroniczna (Celery)** – `train_model_task` w `tasks/tasks.py`:
   • przygotowuje środowisko MLflow, waliduje parametry,
   • buduje listę argumentów CLI dla skryptu `train.py` (każdy parametr jako `--flag value`),
   • uruchamia **izolowany** proces `subprocess.Popen(...)` z tym skryptem (funkcja `run_training_direct` → `managed_process`).

   Dzięki osobnemu procesowi wyciek pamięci lub błąd GPU nie zabija workera Celery. Stop training realizowany jest przez plik `stop_training.flag` oraz `SIGTERM`/`SIGKILL` dla całej grupy procesów.

4. **Skrypt trenowania** – `core/apps/ml_manager/training/train.py`:
   • ładuje dane i model (fabryka w `training/models/...`),
   • wykonuje pętlę treningową, zapisuje checkpointy do `<model_directory>/checkpoints`,
   • loguje metryki do stdout (parsowane przez Celery) oraz MLflow,
   • po zakonczeniu zapisuje najlepsze wagi w `MLModel.model_weights_path`.

### Kluczowe pliki
```
views.py                → StartTrainingView.form_valid
services/training_service.py → MLTrainingService
tasks/tasks.py          → train_model_task, run_training_direct
utils/training_utils.py → TrainingController (buduje komendę, obsługa Popen)
training/train.py       → właściwy kod ML
```

---

## Pipeline Inferencji

1. **Widok** – `GeneralInferenceView` lub `ModelInferenceView` przyjmuje plik wejściowy i tworzy rekord `InferenceResult`.
2. **Serwis** – `MLPredictionService` (`services/prediction_service.py`) zapisuje tymczasowo obraz i wywołuje zadanie Celery `run_inference_task.delay(model_id, image_path, inference_params)`.
3. **Celery** – `run_inference_task` (w tym samym pliku `tasks/tasks.py`):
   • wyszukuje ścieżkę do wag (`model.model_weights_path` lub automatyczne skanowanie katalogu/MLflow),
   • uruchamia funkcję `run_inference_direct`, która w podprocesie wywołuje `train.py` w trybie `--mode predict`,
   • zapisuje wyniki do `MEDIA_ROOT/inference/outputs/<id>/`, aktualizuje rekord `InferenceResult`.

Model ładowany jest zawsze w CPU lub GPU zgodnie z parametrem `device`; pipeline przewiduje obsługę checkpointów (użytkownik może wybrać konkretny plik `.pth`).

### Kluczowe pliki
```
views.py                    → GeneralInferenceView / ModelInferenceView
services/prediction_service.py → MLPredictionService
tasks/tasks.py             → run_inference_task, run_inference_direct
training/train.py          → --mode predict (część wspólna ze skryptem treningu)
```

---

## Rejestr Architektury Modeli

System korzysta z centralnego **Model Architecture Registry** (`core/apps/ml_manager/utils/architecture_registry.py`).

### Jak to działa?
1. **Inicjalizacja** – przy pierwszym imporcie funkcji `get_default_registry()` tworzony jest singleton klasy `ModelArchitectureRegistry`, a następnie wywoływane są helpery `_register_builtin_architectures()` i `_register_resunet_models()`.
2. **Rejestracja wbudowanych modeli** – plik skanuje lokalne źródła (np. `training/models/unet/unet_model.py`) i rejestruje:
   • `unet` (implementacja lokalna),
   • zestaw fallbacków: `monai_unet`, `resunet`, `deep_resunet`, `resunet_attention`, `deep_resunet_attention`.
   Każdy wpis to `ArchitectureInfo` zawierający klasę modelu, domyślne hiperparametry, kategorię, wersję itd.
3. **Rejestracja dynamiczna** – metoda `register_from_module()` pozwala załadować klasę modelu z dowolnego pliku `.py` wskazanego w parametrach.
4. **Pobieranie modelu** – funkcja
```python
model, arch_info = create_model_from_registry(model_type, device, **model_kwargs)
```
   • wyszukuje `arch_info` w rejestrze,
   • łączy `arch_info.default_config` z przekazanymi `model_kwargs`,
   • tworzy instancję klasy `arch_info.model_class`,
   • przenosi ją na wskazane urządzenie (`.to(device)`).

### Gdzie to jest używane?
• `training/train.py` – podczas trenowania (`train_model`) i inferencji (`--mode predict`).
• `utils/model_summary.py` oraz `utils/enhanced_inference.py` do podglądu architektury i wnioskowania offline.

### Dodawanie nowej architektury
```python
from core.apps.ml_manager.utils.architecture_registry import get_default_registry, ArchitectureInfo

registry = get_default_registry()
registry.register(ArchitectureInfo(
    key='super_unet',
    display_name='Super U-Net',
    framework='PyTorch',
    description='Eksperymentalna wersja U-Net',
    model_class=SuperUNet,
    default_config={'n_channels': 1, 'n_classes': 1},
    category='segmentation',
    supports_2d=True,
    version='0.1.0',
))
```

Po rejestracji model można wybrać w formularzu „Start Training” przez pole `model_type`.

### Walidacja architektury
`ModelArchitectureRegistry.validate_architecture(key)` sprawdza, czy model da się poprawnie zainicjalizować – wywoływane w `create_model_from_registry` przed faktycznym użyciem.

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

---

## Architektura Augmentacji Danych

W procesie trenowania kluczową rolę odgrywa warstwa augmentacji, która jest definiowana w funkcji `get_monai_transforms` (plik `training/train.py`).  Pipeline jest budowany jako obiekt `Compose` z biblioteki **MONAI** i może dynamicznie uwzględniać zarówno medyczne preprocessing, jak i klasyczne augmentacje.

### Logika generowania transformacji
1. **Ładowanie i normalizacja**  – stałe transformaty otwierające: `LoadImaged`, `EnsureChannelFirstd`, `ScaleIntensityd`.
2. **Opcjonalny preprocessing medyczny** (jeśli w `training_config` ustawiono `use_medical_preprocessing=True`).  Wykonywany w globalnej funkcji `medical_preprocessing_wrapper`, by zachować pickle-ability dla multiprocessing.
3. **Augmentacje treningowe**  – dodawane tylko dla fazy *train* (pomijane w walidacji / inferencji):
   • losowe obroty, flippowanie, skalowanie,
   • wycinanie losowych patchy (`RandSpatialCropd`),
   • zmiany jasności/kontrastu (`RandAdjustContrastd`, `RandGaussianNoised` – gdy aktywne).
4. **Zamknięcie**  – konwersja do tensora i ewentualne klonowanie kanałów.

### Skrócony przykład kodu
```python
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    RandFlipd, RandRotate90d, RandSpatialCropd, RandAdjustContrastd,
    RandGaussianNoised, Lambdad
)


def get_monai_transforms(params: dict, for_training: bool = True):
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
    ]

    # Medyczny preprocessing (CLAHE, filtr Frangi, denoising…)
    if params.get("use_medical_preprocessing"):
        transforms.append(Lambdad(
            keys=["image"],
            func=lambda d: medical_preprocessing_wrapper(
                d,  # obraz numpy
                preprocessing_type=params.get("preprocessing_type", "angiography"),
                preprocessing_params=params,
            ),
        ))

    if for_training:
        transforms += [
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandRotate90d(keys=["image", "label"], prob=0.5),
            RandSpatialCropd(keys=["image", "label"], roi_size=params.get("crop_size", 256), random_center=True, random_size=False),
        ]

        if params.get("use_random_intensity", False):
            transforms += [
                RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
                RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.01),
            ]

    return Compose(transforms)
```

### Konfiguracja przez formularz UI
Użytkownik może włączyć poszczególne augmentacje w formularzu *Start Training* (pole `use_random_flip`, `use_random_rotate`, `use_random_intensity`, `crop_size` itd.).  Parametry te trafiają do `training_config`, a następnie do funkcji `get_monai_transforms`.

### Najważniejsze cechy
• **Modularność** – każdy krok można łatwo włączyć/wyłączyć poprzez parametr w konfiguracji.  
• **Pickle-friendly** – własne funkcje (np. `medical_preprocessing_wrapper`) zadeklarowane globalnie, aby działały z `num_workers>0`.  
• **Wsparcie 2D/3D** – transformacje działają dla danych volumetrycznych dzięki `spatial_dims` w MONAI.

---

## Uruchamianie Treningu z Terminala

System pozwala uruchomić trening **bez interfejsu webowego** poprzez bezpośrednie wywołanie skryptu `train.py`.  Plik znajduje się w:
```
core/apps/ml_manager/training/train.py
```

### Minimalny przykład
```bash
python core/apps/ml_manager/training/train.py \
    --mode train \
    --data-path data/datasets/basic \
    --model-type unet \
    --epochs 50
```

### Najczęściej używane parametry
| Parametr | Domyślnie | Opis |
|----------|-----------|------|
| `--mode` | *wymagany* | `train` lub `predict` |
| `--data-path` | – | Ścieżka do folderu lub archiwum z danymi |
| `--model-type` | `unet` | Klucz architektury z rejestru |
| `--batch-size` | 32 | Wielkość mini-batcha |
| `--epochs` | 100 | Liczba epok |
| `--learning-rate` | 0.001 | Początkowy LR |
| `--device` | `cuda` | Urządzenie (`cpu`, `cuda`, `cuda:0` …) |
| `--validation-split` | 0.2 | Procent danych na walidację |
| `--threshold` | 0.5 | Próg binarizacji przy segm. |
| `--random-flip` / `--random-rotate` etc. | off | Flagi augmentacji |

### Zaawansowane flagi (wybrane)
* **Scheduler** – `--lr-scheduler plateau|step|cosine` + `--lr-*` parametry.
* **Early-Stopping** – `--use-early-stopping` + `--early-stopping-*`.
* **Checkpointing** – `--checkpoint-strategy best|epoch|interval` , `--max-checkpoints`.
* **Medyczny preprocessing** – `--use-medical-preprocessing` oraz prefiksy `--preprocessing-*`.
* **Mixed Precision** – `--use-mixed-precision` (wymaga GPU i PyTorch AMP).

Kompletną listę parametrów uzyskasz poleceniem:
```bash
python core/apps/ml_manager/training/train.py --help
```

### Uruchomienie inferencji
```bash
python core/apps/ml_manager/training/train.py \
    --mode predict \
    --model-path models/organized/…/checkpoints/best.pth \
    --input-path sample.jpg \
    --output-dir tmp/inference_result \
    --device cpu
```

### Pełna lista parametrów CLI (skrót)

| Kategoria | Flagi | Opis |
|-----------|-------|------|
| **Tryb pracy** | `--mode train|predict`, `--save-training-template` | Wybór trybu lub zapis szablonu configu |
| **Ścieżki i identyfikatory** | `--data-path`, `--model-path`, `--input-path`, `--output-dir`, `--mlflow-run-id`, `--model-id`, `--celery-task-id` | Lokalizacje danych, wag, plików wejściowych i identyfikatory MLflow/Celery |
| **Model** | `--model-type`, `--model-family`, `--model-size`, `--custom-channels`, `--use-attention`, `--use-deep-architecture`, `--use-residual-connections` | Wybór architektury i jej konfiguracji |
| **Uczenie** | `--batch-size`, `--epochs`, `--learning-rate`, `--optimizer`, `--device`, `--validation-split`, `--num-workers` | Podstawowe hiperparametry uczenia |
| **Scheduler LR** | `--lr-scheduler`, `--lr-patience`, `--lr-factor`, `--lr-step-size`, `--lr-gamma`, `--min-lr` | Kontrola harmonogramu LR |
| **Loss/Metryki** | `--loss-type`, `--loss-function`, `--dice-weight`, `--bce-weight`, `--loss-smooth`, `--primary-metric` | Wybór funkcji straty i metryki monitorowania |
| **Checkpointy** | `--checkpoint-strategy`, `--checkpoint-freq`, `--checkpoint-interval`, `--max-checkpoints`, `--checkpoint-metric`, `--checkpoint-mode` | Strategie zapisu wag |
| **Early-Stopping** | `--use-early-stopping`, `--early-stopping-patience`, `--early-stopping-min-epochs`, `--early-stopping-min-delta`, `--early-stopping-metric` | Parametry wczesnego zatrzymania |
| **Augmentacje proste** | `--random-flip`, `--random-rotate`, `--random-scale`, `--random-intensity`, `--crop-size`, `--threshold` | Klasyczne augmentacje i próg segmentacji |
| **Medyczny preprocessing** | `--use-medical-preprocessing`, `--medical-preprocessing-type`, wszystkie flagi `--preprocessing-*` | Zaawansowany preprocessing angiograficzny (CLAHE, Frangi, denoising, gamma, itp.) |
| **Mixed precision / Enhancements** | `--use-mixed-precision`, `--use-enhanced-training`, `--use-loss-scheduling`, `--loss-scheduler-type` | Usprawnienia wydajności i adaptacyjne |
| **Inferencja** | `--weights-path` (opcjonalna inna waga) | Dodatkowe opcje dla trybu `predict` |

Dla kompletnie najnowszej listy wywołaj:
```bash
python core/apps/ml_manager/training/train.py --help | less
```

### Mapowanie parametrów na fragmenty kodu
Poniżej przedstawiono, gdzie w pliku `train.py` (lub plikach pomocniczych) dany argument jest używany.

| Parametr CLI | Główna funkcja / linia | Krótki opis działania |
|--------------|-----------------------|-----------------------|
| `--data-path` | `get_datasets_with_auto_detection` (ok. l. 2325) | Wykrywa typ zbioru, tworzy `train_loader` / `val_loader`. |
| `--model-type` | `create_model_from_registry` (l. 1135) | Pobiera klasę z rejestru architektur i instancjuje model. |
| `--batch-size` | `get_monai_datasets` / DataLoader (l. 2198) | Ustawia `batch_size` przekazywany do `DataLoader`. |
| `--epochs` | pętla w `train_model` (l. 3021) | Steruje liczbą iteracji głównej pętli treningowej. |
| `--learning-rate` | `create_optimizer` (l. 2491) | Tworzy optymalizator z podanym LR. |
| `--optimizer` | `create_optimizer` (l. 2491) | Wybiera klasę optymalizatora (`Adam`, `SGD`, …). |
| `--device` | przekazywany do `create_model_from_registry` i `.to(device)` | Określa CPU / GPU dla modelu i tensora wejściowego. |
| `--validation-split` | `get_datasets_with_auto_detection` | Rozdziela dane na train/val. |
| `--num-workers` | `DataLoader` | Liczba workerów I/O. |
| `--lr-scheduler` + `--lr-*` | `create_optimizer` + helper `get_scheduler` (wewn. funkcji) | Konfiguruje obiekt scheduler i aktualizuje LR po epoce. |
| `--loss-type`, `--loss-function`, `--dice-weight`, `--bce-weight` | `create_advanced_loss_function` (l. 2515) | Buduje funkcję straty – mieszane lub pojedyncze. |
| `--checkpoint-strategy`, `--max-checkpoints` | funkcje zapisu w `train_model` (sekcja *Save checkpoint*) | Kontroluje, kiedy i ile plików `.pth` zachować. |
| `--use-early-stopping`, `--early-stopping-*` | blok Early-Stopping w `train_model` (w pętli epok) | Przerywa trening po braku poprawy. |
| `--random-flip` / `--random-rotate` / … | `get_monai_transforms` (l. 1617) | Dodaje odpowiednie transformacje augmentacyjne. |
| `--crop-size` | `RandSpatialCropd` w `get_monai_transforms` | Określa wielkość patchy. |
| `--use-medical-preprocessing` + `--preprocessing-*` | `medical_preprocessing_wrapper` (l. 1465) | Włącza CLAHE, unsharp, Frangi, denoising itp. |
| `--use-mixed-precision` | w `train_model` otoczone `torch.cuda.amp.autocast()` | Aktywuje AMP dla GPU. |
| `--mode predict` + `--input-path` + `--model-path` | funkcja `run_inference` (l. 5994) | Ładuje model, obrazy i zapisuje maski w `output_dir`. |
| `--threshold` | `run_inference` → post-processing | Proguje wynik segmentacji. |

> **Wskazówka**: numery linii mogą się nieznacznie różnić przy kolejnych commitach – szukaj powyższych funkcji nazwą w IDE.

---

## Dostępne Architektury Modeli

Poniższa tabela przedstawia zestaw kluczy (`model-type`) akceptowanych obecnie przez parametr `--model-type` oraz sposób pozyskania klasy modelu w rejestrze:

| Klucz (`model-type`) | Klasa / moduł źródłowy | Kategoria | Uwagi |
|----------------------|------------------------|-----------|-------|
| `unet` | `training/models/unet/unet_model.py::UNet` | Segmentation | Lokalna implementacja podstawowa; domyślnie 5 poziomów encoder-decoder. |
| `monai_unet` | Fallback → `UNet` | Segmentation | Jeśli MONAI nie jest zainstalowane, rejestr zwraca `UNet` jako zamiennik. |
| `resunet` | `training/models/resunet_model.py::ResUNet` | Segmentation | Residual U-Net z blokami res. |
| `deep_resunet` | `training/models/resunet_model.py::DeepResUNet` | Segmentation | Głębsza wersja ResUNet. |
| `resunet_attention` | `ResUNet` + flag `use_attention=True` | Segmentation | Ten sam kod co ResUNet, ale z attention gates. |
| `deep_resunet_attention` | `DeepResUNet` + `use_attention=True` | Segmentation | Głębsza ResUNet z attention. |
| `attention_unet` | Fallback → `UNet` | Segmentation | Wariant z attention; fallback gdy brak dedykowanej klasy. |
| `unet_plus_plus` | Fallback → `UNet` | Segmentation | Nested U-Net (fallback). |
| `deeplab` | Fallback → `UNet` | Segmentation | Placeholder do przyszłej integracji DeepLab. |
| `segnet` | Fallback → `UNet` | Segmentation | Placeholder SegNet. |
| `unet_classifier` | `training/models/classification_models.py::UNetClassifier` | Classification | Konwolucyjna klasyfikacja (rzadziej używana). |
| `resunet_classifier` | `classification_models.py::ResUNetClassifier` | Classification | Klasyfikator z blokami res. |
| `deep_resunet_classifier` | `classification_models.py::DeepResUNetClassifier` | Classification | J.w. dla Deep ver. |
| `resunet_attention_classifier` | `classification_models.py::ResUNetClassifier` (`use_attention=True`) | Classification | Attention + classif. |

> **Uwaga**: w kodzie mogą istnieć kolejne architektury eksperymentalne rejestrowane dynamicznie przez `register_from_module()` – lista powyżej obejmuje te dostępne „out-of-the-box”. Uzyskasz ją w runtime poleceniem:
> ```python
> from core.apps.ml_manager.utils.architecture_registry import get_default_registry
> print(list(get_default_registry().get_all_architectures().keys()))
> ```

### Mechanizm ładowania – krok po kroku
1. Skrypt treningowy lub inferencyjny wywołuje `create_model_from_registry(model_type, device, **cfg)`.
2. Funkcja:
   1. pobiera singleton rejestru (`get_default_registry()`),
   2. wywołuje `validate_architecture(model_type)` – sprawdzając, czy architektura istnieje i czy jej klasa da się zaimportować,
   3. łączy `arch_info.default_config` z parametrami przekazanymi w CLI / przez UI; parametry `in_channels` / `out_channels` są mapowane na `n_channels` / `n_classes` dla kompatybilności,
   4. tworzy instancję `arch_info.model_class(**final_cfg)` i przenosi ją na `device`.
3. W przypadku modeli klasyfikacyjnych (task_type=`artery_classification`) model typu `unet` zostanie automatycznie zamieniony na odpowiadający `*_classifier` (mapowanie w kodzie l. 1125-1150 `train.py`).
4. Po inicjalizacji funkcja zwraca `(model, arch_info)`, a dalej idzie konfiguracja optymalizatora, loss itd.

---
