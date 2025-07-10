# Scripts Directory

Ten folder zawiera wszystkie skrypty pomocnicze do zarządzania środowiskiem deweloperskim MLManager.

## Skrypty

### `setup.sh`
Główny skrypt konfiguracji infrastruktury:
- Logowanie do GitHub Container Registry
- Tworzenie katalogów
- Pobieranie obrazów Docker
- Uruchamianie kontenerów

### `dev.sh`
Narzędzie do zarządzania środowiskiem deweloperskim:
```bash
./scripts/dev.sh start      # Uruchom kontenery
./scripts/dev.sh stop       # Zatrzymaj kontenery
./scripts/dev.sh restart    # Restart kontenerów
./scripts/dev.sh logs       # Pokaż logi
./scripts/dev.sh shell      # Wejdź do kontenera Django
./scripts/dev.sh rebuild    # Przebuduj kontenery
./scripts/dev.sh clean      # Wyczyść system Docker
```

### `django-setup.sh`
Zarządzanie Django:
```bash
./scripts/django-setup.sh setup      # Pełny setup (migracje + superuser)
./scripts/django-setup.sh migrate    # Tylko migracje
./scripts/django-setup.sh user       # Stwórz superusera
./scripts/django-setup.sh shell      # Django shell
./scripts/django-setup.sh reset      # Reset bazy danych
./scripts/django-setup.sh test       # Uruchom testy
```

### `quick-setup.sh`
Szybki setup całego środowiska w jednej komendzie:
- Uruchamia infrastrukturę
- Konfiguruje Django
- Tworzy superusera

### `setup_environment.sh`
Dodatkowe ustawienia środowiska (jeśli istnieje).

## Użycie przez Makefile

Wszystkie skrypty są dostępne przez Makefile w głównym katalogu:

```bash
make quick          # Szybki setup
make setup          # Główny setup
make start          # Uruchom
make stop           # Zatrzymaj
make logs           # Logi
make django-setup   # Setup Django
make django-migrate # Migracje
make django-user    # Nowy użytkownik
make django-shell   # Django shell
```

## Struktura

```
scripts/
├── README.md           # Ta dokumentacja
├── setup.sh           # Główny setup infrastruktury
├── dev.sh             # Zarządzanie kontenerami
├── django-setup.sh    # Zarządzanie Django
├── quick-setup.sh     # Szybki setup wszystkiego
└── setup_environment.sh # Dodatkowe ustawienia
```

## Wymagania

- Docker i Docker Compose
- Bash
- Token GitHub (CR_PAT) w pliku `.env`
- Uprawnienia do wykonywania skryptów (`chmod +x`)

## Rozwiązywanie problemów

1. **Błąd uprawnień**: `chmod +x scripts/*.sh`
2. **Port zajęty**: Sprawdź `lsof -i :5000` i `lsof -i :8000`
3. **Problemy z Docker**: `make clean` i spróbuj ponownie
4. **Brak tokenów**: Sprawdź plik `.env`
