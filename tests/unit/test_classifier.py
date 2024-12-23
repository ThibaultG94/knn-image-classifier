import pytest
import numpy as np
from src import ImageClassifier

class TestImageClassifier:
    @pytest.fixture
    def sample_dataset(self):
        """Crée un petit dataset de test"""
        images = [
            np.zeros((28, 28)),  # Classe 0
            np.ones((28, 28)),   # Classe 1
            np.zeros((28, 28))   # Classe 0
        ]
        labels = [0, 1, 0]
        return images, labels
    
    def test_voting_system(self, sample_dataset):
        """Test du système de vote avec k=3"""
        classifier = ImageClassifier(k=3)
        images, labels = sample_dataset
        classifier.fit(images, labels)
        
        # L'image test est plus proche des zeros
        test_image = np.zeros((28, 28)) + 0.1
        assert classifier.predict(test_image) == 0