# KNN Image Classifier 🖼️

Un classificateur d'images évolutif utilisant l'algorithme k-Nearest Neighbors. Conçu comme un projet éducatif qui évolue progressivement de tâches simples vers des défis plus complexes de reconnaissance d'images.

## Features

- Implémentation progressive de la reconnaissance d'images :

  - Phase 1 : Reconnaissance de chiffres manuscrits (MNIST)
  - Phase 2 : Extension aux lettres manuscrites
  - Phase 3 : Formes et symboles simples
  - Phases futures : Visages, dessins simples, mots manuscrits

- Architecture modulaire et extensible :

  - Pattern Strategy pour les méthodes de calcul de distance
  - Approche orientée objet avec principes SOLID
  - Facilement adaptable pour de nouvelles catégories d'images
  - Visualisation des résultats et des k plus proches voisins

- Interface intuitive :
  - Zone de dessin interactive
  - Affichage en temps réel des prédictions
  - Visualisation des images similaires trouvées

## Installation

1. Cloner le repository :

```bash
git clone https://github.com/ThibaultG94/knn-image-classifier.git
cd knn-image-classifier
```

2. Créer un environnement virtuel :

```bash
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
```

3. Installer les dépendances :

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.knn import ImageClassifier

# Initialiser le classificateur
classifier = ImageClassifier(k=3)

# Entraîner le modèle
classifier.fit(training_images, labels)

# Faire des prédictions
prediction = classifier.predict(new_image)

# Évaluer la performance
accuracy = classifier.evaluate(test_images, test_labels)
```

## Structure du Projet

```
knn-image-classifier/
├── data/               # Données d'entraînement par catégorie
│   ├── digits/        # Dataset MNIST
│   ├── letters/       # Future extension
│   └── symbols/       # Future extension
├── src/
│   ├── knn.py         # Classificateur principal
│   ├── strategies/    # Implémentations des stratégies
│   │   ├── distance.py
│   │   └── preprocessing.py
│   ├── image.py       # Traitement d'images
│   └── utils.py       # Fonctions utilitaires
├── tests/             # Tests unitaires
├── notebooks/         # Jupyter notebooks de démonstration
└── requirements.txt   # Dépendances
```

## Développement

### Exécuter les Tests

```bash
pytest tests/
```

### Contribuer

1. Forker le repository
2. Créer une branche pour votre fonctionnalité
3. Implémenter et tester vos changements
4. Créer une pull request

## Technologies Utilisées

- Python 3.x
- NumPy pour les calculs matriciels
- Pillow pour le traitement d'images
- Matplotlib pour la visualisation
- pytest pour les tests unitaires

## Feuille de Route

- [x] Architecture de base et pattern Strategy
- [x] Implémentation du calcul de distance
- [ ] Interface de dessin web
- [ ] Support des chiffres manuscrits (MNIST)
- [ ] Visualisation des k plus proches voisins
- [ ] Extension aux lettres manuscrites
- [ ] Support des formes et symboles

## License

Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.

## Remerciements

- La communauté open source Python
- Les contributeurs du dataset MNIST
- Tous les camarades qui participent au développement
