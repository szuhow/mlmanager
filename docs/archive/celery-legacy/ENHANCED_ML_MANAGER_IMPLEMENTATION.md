# Enhanced ML Manager Implementation Guide

## Wprowadzone Ulepszenia

Zgodnie z uwagami użytkownika, wprowadzono następujące kluczowe ulepszenia:

### 1. 🔧 Ulepszone Post-Processing dla Segmentacji

**Problem**: Modele segmentacyjne (zwłaszcza w wczesnych epokach) wykrywają dużo małych, nieistotnych elementów (fałszywe pozytywy/szum).

**Rozwiązanie**: Implementacja zaawansowanego post-processingu w `ml/utils/post_processing.py`:

- **Progowanie adaptacyjne**: Wyższe progi pewności (0.6-0.8) zamiast standardowych 0.5
- **Operacje morfologiczne**: 
  - Opening - usuwa małe obiekty i cienkie połączenia
  - Closing - wypełnia małe dziury i łączy bliskie fragmenty
- **Filtrowanie komponentów**: Usuwanie połączonych komponentów mniejszych niż próg (50-100 pikseli)
- **Filtrowanie kształtów**: Sprawdzanie czy komponenty mają charakterystykę naczyń krwionośnych

```python
# Przykład użycia
from ml.utils.post_processing import enhanced_post_processing

result = enhanced_post_processing(
    model_output,
    model_type='segmentation',
    threshold=0.6,  # Wyższy próg
    min_component_size=50,  # Usuń małe komponenty
    confidence_threshold=0.6  # Próg pewności
)
```

### 2. 🚀 System Zarządzania Treningiem z Celery

**Problem**: Brak właściwego systemu zatrzymywania treningu i zarządzania procesami.

**Rozwiązanie**: Implementacja systemu Celery w `core/apps/ml_manager/tasks.py`:

- **Asynchroniczne zadania**: Training działa w tle jako Celery task
- **Graceful stopping**: Prawidłowe zatrzymywanie treningu przez stop_training_task
- **Monitoring procesów**: Śledzenie aktywnych zadań treningowych
- **Cleanup**: Automatyczne czyszczenie przestarzałych procesów

```python
# Uruchamianie treningu
task = train_model_task.delay(model_id, enhanced_config)

# Zatrzymywanie treningu
stop_task = stop_training_task.delay(model_id)
```

### 3. 📊 Ulepszone Funkcje Straty

**Problem**: Standardowa BCE nie radzi sobie z niezrównoważonymi klasami i może generować szum.

**Rozwiązanie**: Implementacja zaawansowanych funkcji straty w `utils/training_utils.py`:

- **Combined Dice + Focal Loss**: Łączy zalety obu podejść
- **Tversky Loss**: Lepsze dla niezbalansowanych danych
- **Focal Loss**: Koncentruje się na trudnych przykładach
- **Parametry regulacji**: Weight decay, dropout dla zmniejszenia przeuczenia

```python
# Konfiguracja ulepszonych funkcji straty
enhanced_config = {
    'loss_function': 'combined_dice_focal',
    'dice_weight': 0.7,  # Większy nacisk na segmentację
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    'weight_decay': 1e-4,  # L2 regularization
    'dropout_rate': 0.1,   # Dropout dla regularyzacji
}
```

### 4. 🎯 System Checkpointów

**Problem**: W inferencji dla konkretnego modelu powinny być tylko jego checkpointy, a w głównej inferencji - wszystkie.

**Rozwiązanie**: Rozdzielenie logiki checkpointów w views:

#### Inferencja dla konkretnego modelu (`/models/<id>/inference/`):
```python
def _get_model_checkpoints(self, model):
    """Checkpointy tylko dla tego modelu"""
    model_dir = Path(settings.MEDIA_ROOT) / 'models' / str(model.id)
    # Zwraca tylko checkpointy z katalogu tego modelu
```

#### Inferencja główna (`/inference/`):
```python
def _get_all_checkpoints(self):
    """Checkpointy ze wszystkich modeli"""
    for model in MLModel.objects.filter(status='completed'):
        all_checkpoints.extend(self._get_model_checkpoints(model))
```

### 5. 📈 Early Stopping i Regularyzacja

**Rozwiązanie**: Implementacja inteligentnego early stopping:

- **Monitoring metryk**: Śledzenie val_dice_score, val_loss
- **Patience**: Zatrzymanie po N epokach bez poprawy
- **Learning rate scheduling**: Automatyczne zmniejszanie LR
- **Checkpoint management**: Zapisywanie tylko najlepszych modeli

## 🛠 Instalacja i Konfiguracja

### 1. Zainstaluj wymagane pakiety:

```bash
pip install -r requirements/base.txt
```

### 2. Skonfiguruj Redis dla Celery:

```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis

# Uruchom Redis
redis-server
```

### 3. Dodaj konfigurację Celery do Django settings:

```python
# settings.py
from config.settings_celery_example import *
```

### 4. Uruchom Celery worker:

```bash
# W katalogu projektu
celery -A core worker --loglevel=info

# Dla różnych kolejek
celery -A core worker --loglevel=info -Q training,inference,control
```

### 5. Uruchom Celery beat (dla zadań okresowych):

```bash
celery -A core beat --loglevel=info
```

## 📊 Używanie Nowych Funkcji

### Enhanced Inference

1. **Dla konkretnego modelu**:
   ```
   /ml_manager/models/<model_id>/inference/
   ```

2. **Inferencja główna** (wszystkie modele):
   ```
   /ml_manager/inference/
   ```

### Konfiguracja Post-Processing

W formularzu inferencji można skonfigurować:
- **Threshold**: 0.6-0.8 dla bardziej konserwatywnej segmentacji
- **Min Component Size**: 50-100 pikseli dla usunięcia małych elementów
- **Morphological Operations**: Opening/Closing dla redukcji szumu
- **Confidence Threshold**: Próg pewności dla finalnej maski

### Enhanced Training

```python
# API endpoint
POST /ml_manager/training/start/
{
    "model_id": 123,
    "training_config": {
        "loss_function": "combined_dice_focal",
        "dice_weight": 0.7,
        "early_stopping_patience": 15,
        "weight_decay": 1e-4
    }
}

# Zatrzymanie treningu
POST /ml_manager/training/stop/
{
    "model_id": 123
}
```

## 📈 Monitoring

### Status treningu:
```
GET /ml_manager/training/status/<model_id>/
```

### Status inferencji:
```
GET /ml_manager/inference/status/<task_id>/
```

## 🔄 Migracja z Poprzedniej Wersji

1. **Backup istniejących modeli**
2. **Dodaj nowe pola do modelu MLModel** (jeśli potrzebne)
3. **Uruchom migracje Django**
4. **Skonfiguruj Redis i Celery**
5. **Przetestuj nowe funkcje na development**

## 📝 Zalecenia Użytkowania

### Dla redukcji szumu w segmentacji:

1. **Zacznij od wyższych progów** (0.6-0.7)
2. **Ustaw min_component_size na 50-100** w zależności od rozdzielczości
3. **Włącz operacje morfologiczne**
4. **Użyj combined_dice_focal loss** dla nowych treningów
5. **Monitoruj early stopping** na val_dice_score

### Dla zarządzania treningiem:

1. **Zawsze używaj Celery** dla długich treningów
2. **Monitoruj metryki** przez /training/status/
3. **Ustaw odpowiedni patience** dla early stopping (10-20 epok)
4. **Regularnie czyść stare checkpointy**

## 🚨 Troubleshooting

### Celery nie działa:
- Sprawdź czy Redis jest uruchomiony
- Sprawdź logi Celery worker
- Upewnij się że CELERY_BROKER_URL jest poprawnie skonfigurowany

### Post-processing nie redukuje szumu:
- Zwiększ threshold do 0.7-0.8
- Zwiększ min_component_size
- Sprawdź czy apply_opening=True

### Training nie zatrzymuje się:
- Sprawdź czy stop_training_task został wywołany
- Sprawdź logi Celery worker
- W ostateczności użyj SIGTERM na proces

Ten system znacząco poprawia jakość inferencji segmentacyjnej poprzez redukcję fałszywych pozytywów i lepsze zarządzanie procesami treningowymi.
