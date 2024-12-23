import pytest
import numpy as np
from src.knn import ImageClassifier
from src.strategies.preprocessing import MNISTPreprocessing
from src.strategies.distance import EuclideanDistance

class TestWorkflow:
    @pytest.fixture
    def mnist_sample(self):
        # Simuler un petit échantillon MNIST
        train_images = [np.random.randint(0, 255, (28, 28)) for _ in range(100)]
        train_labels = [i % 10 for i in range(100)]  # 0-9 répétés
        test_images = [np.random.randint(0, 255, (28, 28)) for _ in range(10)]
        test_labels = [i for i in range(10)]
        return train_images, train_labels, test_images, test_labels

    def test_end_to_end(self, mnist_sample):
        train_images, train_labels, test_images, test_labels = mnist_sample
        
        # Créer et entraîner le classificateur
        classifier = ImageClassifier(
            k=3,
            preprocessing_strategy=MNISTPreprocessing(),
            distance_strategy=EuclideanDistance()
        )
        
        classifier.fit(train_images, train_labels)
        
        # Évaluer sur l'ensemble de test
        accuracy = classifier.evaluate(test_images, test_labels)
        
        # Avec des données aléatoires, on s'attend à une précision > 10%
        # (meilleure que le hasard pur sur 10 classes)
        assert accuracy > 0.1