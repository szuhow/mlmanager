# Coronary Experiments - ML Manager

Aplikacja do zarządzania modelami ML z integracją Celery i Docker.

## Struktura aplikacji

```
coronary-experiments/
├── core/                           # Główny kod Django
│   ├── apps/
│   │   └── ml_manager/            # Aplikacja ML Manager
│   │       ├── models.py          # Modele Django
│   │       ├── views.py           # Widoki
│   │       ├── services/          # Serwisy biznesowe
│   │       ├── tasks/             # Taski Celery
│   │       ├── training/          # Skrypty treningowe
│   │       └── utils/             # Narzędzia pomocnicze
│   └── config/                    # Konfiguracja Django
│       ├── settings/              # Ustawienia dla różnych środowisk
│       └── celery.py              # Konfiguracja Celery
├── infrastructure/                # Infrastruktura
│   ├── docker/                    # Pliki Docker
│   └── requirements/              # Wymagania Python
├── docker-compose.cpu.yml         # CPU version
├── docker-compose.gpu.yml         # GPU version
└── Makefile                       # Komendy zarządzające
```

## Instalacja i uruchomienie

### Wersja CPU

1. **Zbuduj kontenery:**
   ```bash
   make build-cpu
   ```

2. **Uruchom serwisy:**
   ```bash
   make start-cpu
   ```

3. **Sprawdź status:**
   ```bash
   make status-cpu
   ```

4. **Wykonaj migracje:**
   ```bash
   make migrate-cpu
   ```

### Wersja GPU

1. **Wymagania:**
   - NVIDIA Docker runtime
   - NVIDIA GPU drivers

2. **Zbuduj kontenery:**
   ```bash
   make build-gpu
   ```

3. **Uruchom serwisy:**
   ```bash
   make start-gpu
   ```

4. **Sprawdź status:**
   ```bash
   make status-gpu
   ```

5. **Wykonaj migracje:**
   ```bash
   make migrate-gpu
   ```

## Serwisy

### Django Application
- **URL:** http://localhost:8000
- **Opis:** Główna aplikacja webowa
- **Funkcje:** 
  - Zarządzanie modelami ML
  - Interface do treningu
  - Interfejs inferencji

### Celery Workers
- **Training Worker:**
  - Dedykowany trenowaniu modeli ML
  - Jeden task na raz (GPU memory management)
  - Kolejka: `training`

- **Default Worker:**
  - Obsługuje inferencję, cleanup i ogólne zadania
  - Wielozadaniowy (concurrency: 2)
  - Kolejki: `default`, `cleanup`, `inference`

### Redis
- **Funkcje:**
  - Broker dla Celery
  - Cache dla Django
  - Przechowywanie sesji

### PostgreSQL
- **Funkcje:**
  - Główna baza danych
  - Przechowywanie modeli
  - Logi treningowe

## Taski Celery

### Trening modeli
```python
from core.apps.ml_manager.tasks.tasks import train_model_task

# Uruchom trening
result = train_model_task.delay(model_id, training_params)
```

### Inferencja
```python
from core.apps.ml_manager.tasks.tasks import run_inference_task

# Uruchom inferencję
result = run_inference_task.delay(model_id, image_path, inference_params)
```

### Zatrzymanie treningu
```python
from core.apps.ml_manager.tasks.tasks import stop_training_task

# Zatrzymaj trening
result = stop_training_task.delay(model_id)
```

## Konfiguracja

### Zmienne środowiskowe
Skopiuj `.env.example` do `.env` i dostosuj:

```bash
cp .env.example .env
```

### Kluczowe ustawienia:
- `SECRET_KEY`: Klucz Django
- `DEBUG`: Tryb debugowania
- `DATABASE_URL`: URL bazy danych
- `CELERY_BROKER_URL`: URL Redis dla Celery
- `MLFLOW_TRACKING_URI`: URL MLflow (opcjonalnie)

## Komendy Makefile

### Budowanie
```bash
make build-cpu      # Zbuduj wersję CPU
make build-gpu      # Zbuduj wersję GPU
```

### Uruchamianie
```bash
make start-cpu      # Uruchom wersję CPU
make start-gpu      # Uruchom wersję GPU
```

### Zarządzanie
```bash
make stop-cpu       # Zatrzymaj wersję CPU
make stop-gpu       # Zatrzymaj wersję GPU
make restart-cpu    # Restartuj wersję CPU
make restart-gpu    # Restartuj wersję GPU
```

### Logi
```bash
make logs-cpu       # Logi wersji CPU
make logs-gpu       # Logi wersji GPU
```

### Baza danych
```bash
make migrate-cpu    # Migracje dla CPU
make migrate-gpu    # Migracje dla GPU
```

### Czyszczenie
```bash
make clean-cpu      # Wyczyść wersję CPU
make clean-gpu      # Wyczyść wersję GPU
```

## Rozwój

### Dodawanie nowych tasków
1. Utwórz task w `core/apps/ml_manager/tasks/tasks.py`
2. Zaimportuj w `core/apps/ml_manager/tasks/__init__.py`
3. Użyj w serwisach

### Dodawanie nowych modeli
1. Zarejestruj w `core/apps/ml_manager/utils/architecture_registry.py`
2. Dodaj konfigurację domyślną
3. Przetestuj z training script

### Struktura tasków
```python
@shared_task(bind=True, name='ml_manager.my_task')
def my_task(self, param1, param2):
    """Opis tasku"""
    try:
        # Logika tasku
        return {'success': True, 'result': result}
    except Exception as e:
        return {'success': False, 'error': str(e)}
```

## Monitorowanie

### Flower Dashboard
- URL: http://localhost:5555
- Funkcje: Monitoring tasków, statystyki, zarządzanie

### Django Admin
- URL: http://localhost:8000/admin/
- Funkcje: Zarządzanie modelami, użytkownikami, konfiguracją

### Logi
```bash
# Wszystkie logi
make logs-cpu

# Konkretny serwis
docker-compose -f docker-compose.cpu.yml logs -f django
docker-compose -f docker-compose.cpu.yml logs -f celery-worker
```

## Troubleshooting

### Problemy z GPU
```bash
# Sprawdź NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Sprawdź kontenery GPU
make status-gpu
```

### Problemy z Celery
```bash
# Sprawdź worker
docker-compose -f docker-compose.cpu.yml exec celery-worker celery inspect active

# Sprawdź Redis
docker-compose -f docker-compose.cpu.yml exec redis redis-cli ping
```

### Problemy z Django
```bash
# Sprawdź Django shell
make shell-cpu

# Sprawdź migracje
make migrate-cpu
```

## Wymagania systemowe

### CPU Version
- Docker >= 20.10
- Docker Compose >= 1.29
- RAM >= 8GB
- Dysk >= 20GB

### GPU Version
- NVIDIA Docker runtime
- NVIDIA drivers >= 470.57.02
- CUDA >= 11.0
- RAM >= 16GB
- Dysk >= 50GB

## Bezpieczeństwo

### Produkcja
1. Zmień `SECRET_KEY`
2. Ustaw `DEBUG=false`
3. Skonfiguruj HTTPS
4. Ograniczenia `ALLOWED_HOSTS`
5. Secure Redis/PostgreSQL

### Dane
- Modele w `data/models/`
- Logi w `data/logs/`
- Media w `data/media/`
- Baza danych w volume `postgres_data`
