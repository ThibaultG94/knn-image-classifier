# Import des modules principaux pour un accès simplifié
from .knn import ImageClassifier
from .data.dataset import Dataset
from .strategies.distance import EuclideanDistance
from .strategies.preprocessing import MNISTPreprocessing

# Définir un __all__ pour limiter les imports publics
__all__ = [
    "ImageClassifier",
    "Dataset",
    "EuclideanDistance",
    "MNISTPreprocessing",
]
