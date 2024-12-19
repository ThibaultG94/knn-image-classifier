from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
from typing import Union

class PreprocessingStrategy(ABC):
    @abstractmethod
    def preprocess(self, image: Union[Image.Image, str, np.ndarray]):
        """Prétraite une image pour la classification"""
        pass

class MNISTPreprocessing(PreprocessingStrategy):
    """
    Stratégie de prétraitement adaptée au format MNIST :
    - Images en niveaux de gris
    - Taille 28x28 pixels
    - Valeurs normalisées entre 0 et 1
    """
    def __init__(self, target_size=(28, 28)):
        self.target_size = target_size
    
    def preprocess(self, image: Union[Image.Image, str, np.ndarray]):
        """
        Prétraite une image pour la rendre compatible avec MNIST.

        Args:
            image: PIL Image, numpy array, ou chemin vers une image.

        Returns:
            numpy array normalisé de taille 28x28.

        Raises:
            TypeError: Si le type de l'entrée est invalide.
            ValueError: Si la conversion échoue ou si l'image n'est pas utilisable.
        """
        # Vérifier le type de l'entrée
        if not isinstance(image, (Image.Image, str, np.ndarray)):
            raise TypeError(
                f"L'entrée doit être de type PIL.Image.Image, str ou np.ndarray. "
                f"Type reçu : {type(image)}"
            )

        try:
            # Si l’entrée est un chemin (str), charger l’image avec Pillow
            if isinstance(image, str):
                image = Image.open(image)

            # Si l’entrée est un tableau NumPy, convertir en image Pillow
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            # Convertir en niveaux de gris
            image = image.convert('L')

            # Redimensionner à la taille cible
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)

            # Convertir en tableau NumPy et normaliser entre 0 et 1
            image_array = np.array(image, dtype=np.float32) / 255.0

            return image_array

        except FileNotFoundError:
            raise ValueError("Le chemin spécifié est invalide ou le fichier est introuvable.")
        except OSError:
            raise ValueError("L'image fournie est corrompue ou n'est pas une image valide.")
        except Exception as e:
            raise ValueError(f"Une erreur inattendue est survenue : {str(e)}")
