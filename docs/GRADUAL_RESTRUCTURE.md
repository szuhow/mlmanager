# Stopniowa reorganizacja projektu MLManager

## 🎯 Plan mniejszych, bezpiecznych kroków

Zamiast jednorazowej wielkiej reorganizacji, można wykonać małe kroki:

### **Krok 1: Porządkowanie testów** (najważniejsze)
```bash
mkdir -p tests/organized/{unit,integration,e2e}
# Przeniesienie testów po kategorii
```

### **Krok 2: Separacja konfiguracji Docker**
```bash
mkdir -p docker/
mv docker-compose.yml docker/
mv Dockerfile docker/
# Symlinka dla kompatybilności
```

### **Krok 3: Uporządkowanie requirements**
```bash
mkdir -p requirements/
split requirements.txt na dev/prod/test
```

### **Krok 4: Dokumentacja**
```bash
# Przeniesienie dokumentów z głównego katalogu do docs/
```

## 🚀 Skrypty dla stopniowej reorganizacji

### Krok 1: Tylko testy
```bash
./scripts/reorganize-tests.sh
```

### Krok 2: Tylko Docker
```bash
./scripts/reorganize-docker.sh  
```

### Krok 3: Tylko requirements
```bash
./scripts/reorganize-requirements.sh
```

## ✅ Korzyści stepowego podejścia:

1. **Bezpieczeństwo** - każdy krok można przetestować
2. **Cofnięcie** - łatwe rollback pojedynczego kroku
3. **Minimalne ryzyko** - mniejsze zmiany = mniejsze problemy
4. **Stopniowe przyzwyczajenie** - zespół może się dostosować

---

**Chcesz rozpocząć od reorganizacji testów?** To najbardziej potrzebne i bezpieczne.
