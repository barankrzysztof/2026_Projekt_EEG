# Klasyfikacja ADHD na podstawie sygnałów EEG

Projekt rekrutacyjny polegający na budowie binarnego klasyfikatora diagnozującego ADHD na podstawie sygnałów EEG. Rozwiązanie opiera się na ekstrakcji cech w dziedzinie częstotliwości (fale Delta, Theta, Alpha, Beta), a następnie klasyfikacji za pomocą modeli uczenia maszynowego

## Architektura projektu
* `src/` - moduły wielokrotnego użytku (ładowanie danych, preprocess, feature engineering, wizualizacje, trenowanie modeli).
* `notebooks/` - notatniki zawierające analizę danych, trenowanie modeli oraz ewaluację wyników.

## Jak uruchomić projekt?

1. Klonowanie repozytorium i instalacja zależności:
```bash
git clone <adres_repo>
cd <folder_projektu>
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Pobranie danych:
Należy pobrać [zbiór danych](https://www.kaggle.com/datasets/danizo/eeg-dataset-for-adhd/data), nazwać plik `adhdata.csv` i dać do folderu `data`.

3. Uruchomienie analizy i modeli
Należy najpierw uruchomić komórki po kolei w notatniku z analizą danych (`eda.ipynb`), następne notatnik z modelami `models.ipynb`.