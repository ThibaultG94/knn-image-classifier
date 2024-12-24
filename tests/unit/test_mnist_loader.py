import pytest
import numpy as np
import gzip
from pathlib import Path
from unittest.mock import patch
from src.data.mnist_loader import MNISTDataset

class TestMNISTDataset:
    @pytest.fixture(autouse=True)
    def setup_mnist_files(self, tmp_path):
        """
        Prépare un environnement de test avec des fichiers MNIST valides.
        Utilise tmp_path pour créer un répertoire de test unique pour chaque test.
        """
        mnist_dir = tmp_path / "mnist_test"
        mnist_dir.mkdir(parents=True, exist_ok=True)

        # Définition des noms de fichiers exacts comme dans MNISTDataset
        file_config = {
            'train-images-idx3-ubyte.gz': (2051, 10, 28, 28),  # magic, nb_images, height, width
            'train-labels-idx1-ubyte.gz': (2049, 10),          # magic, nb_labels
            't10k-images-idx3-ubyte.gz': (2051, 10, 28, 28),   # Mêmes dimensions pour le test
            't10k-labels-idx1-ubyte.gz': (2049, 10)            # Mêmes dimensions pour le test
        }

        def create_mnist_files():
            """Crée les fichiers MNIST de test avec la structure exacte attendue"""
            for filename, header_info in file_config.items():
                filepath = mnist_dir / filename
                with gzip.open(filepath, 'wb', compresslevel=6) as f:
                    # Écriture de l'en-tête
                    for value in header_info:
                        f.write(value.to_bytes(4, 'big'))
                    
                    # Écriture des données
                    if 'images' in filename:
                        # Création d'images de test (10 images de 28x28)
                        images = np.zeros((10, 28, 28), dtype=np.uint8)
                        for i in range(10):
                            images[i].fill(i * 25)
                        f.write(images.tobytes())
                    else:
                        # Création de labels (0-9)
                        labels = np.arange(10, dtype=np.uint8)
                        f.write(labels.tobytes())

        # Création des fichiers
        create_mnist_files()

        # Vérification des fichiers créés
        for filename in file_config:
            filepath = mnist_dir / filename
            assert filepath.exists(), f"Fichier manquant : {filename}"
            try:
                with gzip.open(filepath, 'rb') as f:
                    # Lecture de l'en-tête pour vérifier le format
                    magic = int.from_bytes(f.read(4), 'big')
                    expected_magic = file_config[filename][0]
                    assert magic == expected_magic, f"Magic number incorrect pour {filename}"
            except Exception as e:
                raise RuntimeError(f"Fichier invalide {filename}: {e}")

        return mnist_dir

    @pytest.fixture
    def mock_response(self):
        """
        Simule une réponse HTTP avec des données MNIST valides.
        Cette fixture est utilisée pour simuler le téléchargement des fichiers.
        """
        class MockResponse:
            def __init__(self):
                # En-tête MNIST valide
                header = bytearray()
                header.extend((2051).to_bytes(4, 'big'))  # Magic number
                header.extend((10).to_bytes(4, 'big'))    # Nombre d'images
                header.extend((28).to_bytes(4, 'big'))    # Hauteur
                header.extend((28).to_bytes(4, 'big'))    # Largeur
                
                # Données d'images
                images = np.zeros((10, 28, 28), dtype=np.uint8)
                
                # Création du contenu compressé
                with gzip.open(Path("temp.gz"), 'wb', compresslevel=6) as f:
                    f.write(header)
                    f.write(images.tobytes())
                
                # Lecture du contenu compressé
                with open("temp.gz", 'rb') as f:
                    self.content = f.read()
                
                # Nettoyage
                Path("temp.gz").unlink()
            
            def raise_for_status(self):
                pass

        return MockResponse()

    def test_init_creates_directory(self, setup_mnist_files):
        """
        Vérifie que le constructeur crée le répertoire si nécessaire 
        et charge correctement les données.
        """
        dataset = MNISTDataset(data_dir=setup_mnist_files, download=False)
        assert dataset.train_data is not None
        assert dataset.train_labels is not None
        assert isinstance(dataset.train_data, np.ndarray)
        assert isinstance(dataset.train_labels, np.ndarray)

    @patch('requests.get')
    def test_download_mnist(self, mock_get, setup_mnist_files, mock_response):
        """
        Vérifie que le téléchargement et le chargement des données fonctionnent 
        correctement avec les requêtes HTTP mockées.
        """
        mock_get.return_value = mock_response
        dataset = MNISTDataset(data_dir=setup_mnist_files, download=True)
        assert dataset.train_data is not None
        assert dataset.train_labels is not None
        assert dataset.test_data is not None
        assert dataset.test_labels is not None
        assert isinstance(dataset.train_data, np.ndarray)
        assert isinstance(dataset.train_labels, np.ndarray)

    def test_load_data_shape(self, setup_mnist_files):
        """
        Vérifie que les dimensions des données chargées sont correctes.
        Les images doivent être de dimension (10, 28, 28) et les labels (10,).
        """
        dataset = MNISTDataset(data_dir=setup_mnist_files, download=False)
        assert dataset.train_data.shape == (10, 28, 28)
        assert dataset.train_labels.shape == (10,)
        assert dataset.test_data.shape == (10, 28, 28)
        assert dataset.test_labels.shape == (10,)
        # Vérification de la plage des valeurs
        assert np.all((dataset.train_data >= 0) & (dataset.train_data <= 255))
        assert np.all((dataset.train_labels >= 0) & (dataset.train_labels <= 9))

    def test_data_consistency(self, setup_mnist_files):
        """
        Vérifie la cohérence des données entre deux chargements successifs.
        Les données doivent être identiques pour garantir la reproductibilité.
        """
        dataset1 = MNISTDataset(data_dir=setup_mnist_files, download=False)
        dataset2 = MNISTDataset(data_dir=setup_mnist_files, download=False)
        
        # Vérification des données d'entraînement
        np.testing.assert_array_equal(dataset1.train_data, dataset2.train_data)
        np.testing.assert_array_equal(dataset1.train_labels, dataset2.train_labels)
        
        # Vérification des données de test
        np.testing.assert_array_equal(dataset1.test_data, dataset2.test_data)
        np.testing.assert_array_equal(dataset1.test_labels, dataset2.test_labels)

    def test_invalid_directory(self):
        """
        Vérifie la gestion des erreurs avec un chemin invalide.
        Doit lever une FileNotFoundError si le répertoire n'existe pas.
        """
        with pytest.raises(FileNotFoundError):
            dataset = MNISTDataset(data_dir="./invalid_path", download=False)