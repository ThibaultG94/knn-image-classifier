from abc import ABC, abstractmethod
import numpy as np

class DistanceStrategy(ABC):
    @abstractmethod
    def calculate_distance(self, image1, image2):
        pass

class EuclideanDistance(DistanceStrategy):
    def calculate_distance(self, image1, image2):
        """
        Calcule la distance euclidienne entre deux images.
        Les images doivent être des arrays NumPy de même dimension.
        
        Par exemple, si chaque image est 28x28 pixels, elle sera représentée
        comme un vecteur de 784 valeurs (28*28).
        """
        # Vérification des dimensions
        if image1.shape != image2.shape:
            raise ValueError("Les images doivent avoir les mêmes dimensions")
        
        # La distance euclidienne est la racine carrée de la somme 
        # des différences au carré
        return np.sqrt(np.sum((image1 - image2) ** 2))
