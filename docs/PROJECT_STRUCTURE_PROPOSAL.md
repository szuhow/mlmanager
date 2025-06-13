# Propozycja uporządkowania struktury projektu MLManager

## 🔍 Analiza obecnej struktury

### Problemy obecnej struktury:
1. **Rozdrobienie plików konfiguracyjnych** w głównym katalogu
2. **Testy rozproszone** - folder `tests/` zawiera 50+ plików testowych
3. **Brak separacji środowisk** (dev/prod/test)
4. **Dokumentacja w kilku miejscach** (`docs/`, główny katalog)
5. **Mieszanie logiki ML z Django** w jednym miejscu
6. **Brak wyraźnego podziału na warstwy**

## 🎯 Proponowana nowa struktura

```
mlmanager/
├── 📁 config/                          # Konfiguracja środowisk
│   ├── docker-compose.yml              # Production
│   ├── docker-compose.dev.yml          # Development  
│   ├── docker-compose.test.yml         # Testing
│   ├── Dockerfile                      # Django app
│   ├── Dockerfile.mlflow               # MLflow service
│   └── nginx.conf                      # Proxy config
│
├── 📁 core/                            # Główna aplikacja Django
│   ├── coronary_experiments/           # Django settings
│   ├── ml_manager/                     # Główna app Django
│   ├── manage.py
│   ├── requirements/                   # Wymagania per środowisko
│   │   ├── base.txt
│   │   ├── dev.txt
│   │   ├── prod.txt
│   │   └── test.txt
│   └── static/                         # Statyki Django
│
├── 📁 ml/                              # Warstwa Machine Learning
│   ├── shared/                         # Wspólne komponenty ML
│   ├── models/                         # Modele i artefakty
│   ├── training/                       # Skrypty treningowe
│   │   ├── pipelines/
│   │   ├── configs/
│   │   └── scripts/
│   └── inference/                      # Predykcja
│
├── 📁 data/                            # Dane i storage
│   ├── datasets/                       # Datasety
│   ├── artifacts/                      # MLflow artifacts
│   ├── models/                         # Zapisane modele
│   └── temp/                           # Tymczasowe pliki
│
├── 📁 tests/                           # Testy uporządkowane
│   ├── unit/                           # Testy jednostkowe
│   │   ├── test_models.py
│   │   ├── test_views.py
│   │   └── test_ml_utils.py
│   ├── integration/                    # Testy integracyjne
│   │   ├── test_training_workflow.py
│   │   └── test_mlflow_integration.py
│   ├── e2e/                           # Testy end-to-end
│   │   └── test_complete_workflow.py
│   ├── fixtures/                       # Dane testowe
│   └── conftest.py                     # Pytest config
│
├── 📁 scripts/                         # Skrypty automatyzacji
│   ├── deployment/                     # Deployment
│   │   ├── deploy.sh
│   │   └── backup.sh
│   ├── development/                    # Development helpers
│   │   ├── dev.sh
│   │   ├── setup.sh
│   │   └── django-setup.sh
│   └── maintenance/                    # Maintenance
│       ├── cleanup.sh
│       └── migrate.sh
│
├── 📁 docs/                            # Dokumentacja
│   ├── api/                            # API docs
│   ├── deployment/                     # Deployment guides
│   ├── development/                    # Dev setup
│   ├── architecture/                   # Architecture diagrams
│   └── troubleshooting/                # Problem solving
│
├── 📁 infrastructure/                  # DevOps i infrastruktura
│   ├── kubernetes/                     # K8s manifests
│   ├── terraform/                      # Infrastructure as Code
│   ├── monitoring/                     # Prometheus/Grafana configs
│   └── ci-cd/                         # GitHub Actions/Jenkins
│
├── 📄 .env.example                     # Template dla zmiennych środowiskowych
├── 📄 .gitignore
├── 📄 README.md                        # Główna dokumentacja
├── 📄 Makefile                         # Główne komendy
├── 📄 pyproject.toml                   # Python project config
└── 📄 docker-compose.yml               # Symlink do config/docker-compose.yml
```

## 🔄 Plan migracji

### Faza 1: Reorganizacja konfiguracji
```bash
mkdir -p config
mv docker-compose.yml config/
mv Dockerfile config/
# Stworzenie wariantów dla różnych środowisk
```

### Faza 2: Podział aplikacji Django
```bash
mkdir -p core
mv coronary_experiments/ core/
mv ml_manager/ core/
mv manage.py core/
mv requirements.txt core/requirements/base.txt
```

### Faza 3: Separacja warstwy ML
```bash
mkdir -p ml/{training,inference}
mv shared/ ml/
mv models/ data/models/
# Reorganizacja logiki ML
```

### Faza 4: Uporządkowanie testów
```bash
mkdir -p tests/{unit,integration,e2e,fixtures}
# Kategoryzacja i przeniesienie testów
```

### Faza 5: Skrypty i dokumentacja
```bash
mkdir -p scripts/{deployment,development,maintenance}
mkdir -p docs/{api,deployment,development,architecture}
# Przeniesienie i kategoryzacja
```

## 🎯 Korzyści nowej struktury

### 1. **Separation of Concerns**
- Django app ↔ ML logic ↔ DevOps ↔ Tests
- Każda warstwa ma swoje miejsce

### 2. **Skalowalnośći**
- Łatwe dodawanie nowych modeli ML
- Niezależne środowiska (dev/test/prod)
- Modularna architektura

### 3. **Łatwiejsze utrzymanie**
- Czytelne miejsca dla różnych typów plików
- Łatwiejsze onboardowanie nowych developerów
- Lepsze praktyki DevOps

### 4. **Zgodność ze standardami**
- Django best practices
- MLOps patterns
- Python project standards

## 🛠️ Implementacja

### Skrypt automatyzacji migracji:
```bash
#!/bin/bash
# scripts/development/restructure.sh

echo "🔄 Rozpoczynam restruktukturyzację projektu..."

# Backup obecnej struktury
cp -r . ../mlmanager_backup_$(date +%Y%m%d_%H%M%S)

# Tworzenie nowej struktury
mkdir -p {config,core,ml/{training,inference},data/{datasets,artifacts,models,temp}}
mkdir -p {tests/{unit,integration,e2e,fixtures},docs/{api,deployment,development,architecture}}
mkdir -p {scripts/{deployment,development,maintenance},infrastructure/{kubernetes,terraform,monitoring,ci-cd}}

# Migracja plików...
# (szczegółowe kroki)

echo "✅ Restrukturyzacja zakończona!"
```

### Aktualizacja Makefile:
```makefile
# Nowe komendy dla nowej struktury
setup-dev:
	@./scripts/development/setup.sh

deploy-prod:
	@./scripts/deployment/deploy.sh

test-all:
	@cd core && python -m pytest ../tests/

build-ml:
	@cd ml && python -m training.pipelines.main
```

## 📋 TODO dla implementacji

- [ ] Stworzenie skryptu migracji
- [ ] Aktualizacja docker-compose dla nowej struktury  
- [ ] Przeniesienie i kategoryzacja testów
- [ ] Aktualizacja dokumentacji
- [ ] Konfiguracja środowisk dev/test/prod
- [ ] Aktualizacja CI/CD pipeline
- [ ] Sprawdzenie wszystkich importów Python
- [ ] Aktualizacja Makefile i skryptów

## ⚠️ Uwagi

1. **Stopniowa migracja** - nie wszystko na raz
2. **Testy po każdej fazie** - sprawdzenie czy wszystko działa
3. **Backup** - kopia zapasowa przed rozpoczęciem
4. **Team alignment** - uzgodnienie z zespołem
5. **Documentation update** - aktualizacja README i docs

---

*Czy chcesz rozpocząć implementację tej struktury? Mogę stworzyć szczegółowy skrypt migracji.*
