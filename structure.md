# Structure du Projet KNN Image Classifier 🖼️

## Vue d'ensemble

```
knn-image-classifier/
├── src/                      # Code source principal
│   ├── data/                 # Nouveau module de gestion des données
│   │   ├── __init__.py
│   │   ├── mnist_loader.py   # Téléchargement et préparation MNIST
│   │   └── dataset.py        # Classes abstraites pour les datasets
│   │
│   ├── strategies/           # Implémentation du pattern Strategy
│   │   ├── __init__.py
│   │   ├── distance.py       # Stratégies de calcul de distance
│   │   └── preprocessing.py  # Stratégies de prétraitement
│   │
│   └── knn.py               # Classificateur principal
│
├── data/                     # Données stockées
│   └── mnist/               # Dataset MNIST
│       ├── raw/             # Fichiers bruts téléchargés
│       └── processed/       # Données prétraitées
│
├── tests/                    # Tests
│   ├── unit/
│   │   └── data/           # Tests pour le nouveau module
│   ├── integration/
│   └── performance/
│
├── docs/                     # Documentation
│   ├── mnist.md             # Guide d'utilisation MNIST
│   └── performance.md       # Résultats des benchmarks
│
└── scripts/                  # Scripts utilitaires
    └── download_mnist.py    # Script CLI pour MNIST
```

## Composants Principaux

### 1. Module de Gestion des Données (`src/data/`)

#### `dataset.py`

- Classes abstraites définissant l'interface pour les datasets
- Gestion commune des données (chargement, validation)
- Patterns Iterator et Factory pour les datasets

#### `mnist_loader.py`

- Téléchargement du dataset MNIST
- Décompression et validation des fichiers
- Conversion en format numpy
- Cache et gestion des données prétraitées

### 2. Stratégies (`src/strategies/`)

#### `distance.py` (Existant)

- Calcul des distances entre images
- Implémentations : Euclidienne, Manhattan, etc.

#### `preprocessing.py` (Existant)

- Prétraitement des images
- Normalisation et redimensionnement

### 3. Classificateur (`src/knn.py`)

- Implémentation du k-NN
- Interface avec les stratégies
- Parallélisation des calculs

## Flux de Données

1. Téléchargement → `mnist_loader.py`
2. Stockage brut → `data/mnist/raw/`
3. Prétraitement → `preprocessing.py`
4. Stockage traité → `data/mnist/processed/`
5. Classification → `knn.py`

## Formats de Données

### Stockage Raw

- Format original MNIST (fichiers IDX)
- Compression gzip

### Stockage Processed

- Arrays NumPy (.npy)
- Images 28x28 normalisées (0-1)
- Labels encodés

## Tests

### Unit Tests

- `test_mnist_loader.py` : Téléchargement et validation
- `test_dataset.py` : Interface dataset
- Tests existants pour KNN et stratégies

### Integration Tests

- Workflow complet de données
- Performance avec le dataset complet

## Scripts

### `download_mnist.py`

```python
# Exemple d'utilisation:
# python scripts/download_mnist.py --dest data/mnist
```

## Dépendances Principales

- NumPy : Manipulation des données
- Requests : Téléchargement
- Pillow : Traitement d'images
- pytest : Tests

## Conventions

### Nommage

- Classes : PascalCase
- Fonctions/variables : snake_case
- Constantes : MAJUSCULES

### Documentation

- Docstrings : Format Google
- Type hints systématiques
- README par module
