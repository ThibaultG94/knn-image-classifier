from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Iterator, Optional
import numpy as np

class Dataset(ABC):
    """
    Classe abstraite définissant l'interface pour tous les datasets.
    Suit le pattern Template Method pour standardiser le chargement des données
    tout en permettant des implémentations spécifiques.
    """
    def __init__(self, data_dir: str, download: bool = True):
        self.data_dir = Path(data_dir)
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None

        # Crée le répertoire si nécessaire
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if not self._check_exists():
            if download:
                self._download()
            else:
                raise FileNotFoundError(
                    "Les fichiers de données n'existent pas. "
                    "Utilisez download=True pour les télécharger."
                )
        
        self._load_data()

    @abstractmethod
    def _check_exists(self) -> bool:
        """Vérifie si les données sont déjà téléchargées"""
        pass

    @abstractmethod
    def _download(self) -> None:
        """Télécharge les données si nécessaire"""
        pass

    @abstractmethod
    def _load_data(self) -> None:
        """Charge les données en mémoire"""
        pass

    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retourne les données d'entraînement et leurs labels"""
        if self.train_data is None:
            raise RuntimeError("Les données d'entraînement n'ont pas été chargées")
        return self.train_data, self.train_labels

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retourne les données de test et leurs labels"""
        if self.test_data is None:
            raise RuntimeError("Les données de test n'ont pas été chargées")
        return self.test_data, self.test_labels