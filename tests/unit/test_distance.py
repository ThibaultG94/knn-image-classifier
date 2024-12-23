from src import EuclideanDistance
import numpy as np

class TestEuclideanDistance:
    def test_same_image_distance(self):
        """Deux images identiques doivent avoir une distance de 0"""
        strategy = EuclideanDistance()
        img = np.ones((28, 28))
        assert strategy.calculate_distance(img, img) == 0
    
    def test_known_distance(self):
        """Test avec une distance connue"""
        strategy = EuclideanDistance()
        img1 = np.zeros((2, 2))
        img2 = np.ones((2, 2))
        # La distance devrait être sqrt(4) = 2 car on a 4 pixels différant de 1
        assert strategy.calculate_distance(img1, img2) == 2.0