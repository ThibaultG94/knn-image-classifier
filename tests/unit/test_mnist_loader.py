import pytest
import numpy as np
import gzip
from pathlib import Path
from src.data.mnist_loader import MNISTDataset

class TestMNISTDataset:
    @pytest.fixture
    def mnist_dir(self, tmp_path):
        """Crée un répertoire temporaire pour les tests"""
        test_dir = tmp_path / "mnist_test"
        test_dir.mkdir(exist_ok=True)
        return test_dir

    @pytest.fixture
    def mock_mnist_files(self, mnist_dir):
        """
        Crée des fichiers MNIST factices pour tester le chargement.
        Cette fixture simule la structure des vrais fichiers MNIST.
        """
        # En-tête du format IDX pour les images (magic number + dimensions)
        image_header = (
            2051).to_bytes(4, 'big')  # Magic number pour les images
        n_images = (10).to_bytes(4, 'big')  # 10 images de test
        n_rows = (28).to_bytes(4, 'big')
        n_cols = (28).to_bytes(4, 'big')

        # Créer des images factices
        fake_images = np.random.randint(0, 255, (10, 28, 28), 
                                      dtype=np.uint8).tobytes()
        
        # En-tête pour les labels
        label_header = (2049).to_bytes(4, 'big')  # Magic number pour les labels
        n_labels = (10).to_bytes(4, 'big')
        fake_labels = np.random.randint(0, 10, 10, dtype=np.uint8).tobytes()

        # Sauvegarder les fichiers de test
        for prefix in ['train', 'test']:
            # Images
            with gzip.open(mnist_dir / f"{prefix}_images.gz", 'wb') as f:
                f.write(image_header + n_images + n_rows + n_cols + fake_images)
            
            # Labels
            with gzip.open(mnist_dir / f"{prefix}_labels.gz", 'wb') as f:
                f.write(label_header + n_labels + fake_labels)

        return mnist_dir

    def test_init_creates_directory(self, mnist_dir):
        """Vérifie que le constructeur crée le répertoire si nécessaire"""
        dataset = MNISTDataset(data_dir=mnist_dir, download=False)
        assert mnist_dir.exists()

    def test_download_mnist(self, mnist_dir):
        """
        Vérifie que le téléchargement fonctionne.
        Note : Ce test nécessite une connexion Internet
        """
        dataset = MNISTDataset(data_dir=mnist_dir, download=True)
        # Vérifie que tous les fichiers existent
        for name in dataset.URLS.keys():
            assert (mnist_dir / f"{name}.gz").exists()

    def test_load_data_shape(self, mnist_dir, mock_mnist_files):
        """Vérifie que les données chargées ont la bonne forme"""
        dataset = MNISTDataset(data_dir=mnist_dir, download=False)
        train_images, train_labels = dataset.get_train_data()
        
        assert train_images.shape == (10, 28, 28)  # 10 images de 28x28
        assert train_labels.shape == (10,)  # 10 labels
        assert train_images.dtype == np.float32
        assert 0 <= train_images.min() <= train_images.max() <= 1  # Normalisé

    def test_data_consistency(self, mnist_dir, mock_mnist_files):
        """Vérifie que les données restent cohérentes après sauvegarde/chargement"""
        # Premier chargement
        dataset1 = MNISTDataset(data_dir=mnist_dir, download=False)
        images1, labels1 = dataset1.get_train_data()

        # Deuxième chargement (devrait charger depuis le cache)
        dataset2 = MNISTDataset(data_dir=mnist_dir, download=False)
        images2, labels2 = dataset2.get_train_data()

        np.testing.assert_array_equal(images1, images2)
        np.testing.assert_array_equal(labels1, labels2)

    def test_invalid_directory(self):
        """Vérifie la gestion des erreurs avec un répertoire invalide"""
        with pytest.raises(Exception):
            dataset = MNISTDataset(data_dir="/chemin/invalide/impossible", 
                                 download=False)
            dataset.get_train_data()