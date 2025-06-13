# MLManager - Skrypty przeniesione do scripts/

## Zmiany

Wszystkie skrypty `.sh` zostały przeniesione do folderu `scripts/` dla lepszej organizacji:

```
Przed:
├── dev.sh
├── django-setup.sh  
├── quick-setup.sh
├── setup.sh
└── setup_environment.sh

Po:
└── scripts/
    ├── README.md
    ├── dev.sh
    ├── django-setup.sh
    ├── quick-setup.sh
    ├── setup.sh
    └── setup_environment.sh
```

## Zaktualizowane odwołania

### Makefile
Wszystkie odwołania w `Makefile` zostały zaktualizowane:
```makefile
# Przed
@./dev.sh start

# Po  
@./scripts/dev.sh start
```

### Skrypty
Wszystkie wewnętrzne odwołania między skryptami zostały zaktualizowane:
```bash
# Przed
./dev.sh start

# Po
./scripts/dev.sh start
```

## Użycie

**Makefile (zalecane):**
```bash
make quick          # Szybki setup
make start          # Uruchom
make logs           # Logi
make django-setup   # Setup Django
```

**Bezpośrednio:**
```bash
./scripts/quick-setup.sh
./scripts/dev.sh start
./scripts/django-setup.sh setup
```

## Korzyści

1. **Organizacja**: Wszystkie skrypty w jednym miejscu
2. **Czytelność**: Główny katalog jest bardziej uporządkowany
3. **Łatwiejsze zarządzanie**: Jeden folder do kontroli wersji
4. **Dokumentacja**: README.md w folderze scripts/

Wszystkie komendy Makefile działają tak samo jak wcześniej!
