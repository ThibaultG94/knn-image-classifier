import gzip
import numpy as np
import requests
from pathlib import Path
from typing import Dict, Tuple
from .dataset import Dataset

class MNISTDataset(Dataset):
    """
    Gère le dataset MNIST pour la reconnaissance de chiffres manuscrits.
    
    Le dataset contient :
    - 60 000 images d'entraînement avec leurs labels
    - 10 000 images de test avec leurs labels
    - Images en 28x28 pixels, niveaux de gris (0-255)
    """
    
    # URLs officielles des fichiers MNIST
    URLS: Dict[str, str] = {
        'train_images': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'train_labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'test_images': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'test_labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    }

    def _check_exists(self) -> bool:
        """Vérifie si tous les fichiers MNIST sont déjà téléchargés"""
        # Extrait les noms de fichiers des URLs (après le dernier /)
        expected_files = [url.split('/')[-1] for url in self.URLS.values()]
        
        files_exist = all(
            (self.data_dir / filename).exists()
            for filename in expected_files
        )
        return files_exist and (
            self.train_data is not None or 
            (self.data_dir / "processed").exists()
        )

    def _download(self) -> None:
        """Télécharge les fichiers MNIST compressés"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        for name, url in self.URLS.items():
            print(f"Téléchargement de {name}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            file_path = self.data_dir / f"{name}.gz"
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"Sauvegardé dans {file_path}")

    def _load_data(self) -> None:
        """Charge et prétraite les données MNIST"""
        processed_dir = self.data_dir / "processed"
        if processed_dir.exists():
            # Charger les données prétraitées si elles existent
            self.train_data = np.load(processed_dir / "train_images.npy")
            self.train_labels = np.load(processed_dir / "train_labels.npy")
            self.test_data = np.load(processed_dir / "test_images.npy")
            self.test_labels = np.load(processed_dir / "test_labels.npy")
            return

        # Charger et prétraiter les données brutes
        self.train_data, self.train_labels = self._read_mnist_files("train")
        self.test_data, self.test_labels = self._read_mnist_files("test")

        # Sauvegarder les données prétraitées
        processed_dir.mkdir(exist_ok=True)
        np.save(processed_dir / "train_images.npy", self.train_data)
        np.save(processed_dir / "train_labels.npy", self.train_labels)
        np.save(processed_dir / "test_images.npy", self.test_data)
        np.save(processed_dir / "test_labels.npy", self.test_labels)

    def _read_mnist_files(self, dataset: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lit les fichiers MNIST compressés et retourne les images et labels.
        
        Args:
            dataset: "train" ou "test"
            
        Returns:
            Tuple contenant (images, labels)
            - images: numpy array de forme (N, 28, 28) normalisé entre 0 et 1
            - labels: numpy array de forme (N,) contenant les chiffres (0-9)
        """
        # Charger les images
        with gzip.open(self.data_dir / f"{dataset}_images.gz", 'rb') as f:
            # Les premiers octets sont l'en-tête
            _ = int.from_bytes(f.read(4), 'big')  # Magic number
            n_images = int.from_bytes(f.read(4), 'big')
            n_rows = int.from_bytes(f.read(4), 'big')
            n_cols = int.from_bytes(f.read(4), 'big')
            
            # Lire toutes les images
            buf = f.read()
            images = np.frombuffer(buf, dtype=np.uint8)
            images = images.reshape(n_images, n_rows, n_cols)
            images = images.astype(np.float32) / 255.0  # Normalisation

        # Charger les labels
        with gzip.open(self.data_dir / f"{dataset}_labels.gz", 'rb') as f:
            _ = int.from_bytes(f.read(4), 'big')  # Magic number
            n_labels = int.from_bytes(f.read(4), 'big')
            
            # Lire tous les labels
            buf = f.read()
            labels = np.frombuffer(buf, dtype=np.uint8)

        return images, labels