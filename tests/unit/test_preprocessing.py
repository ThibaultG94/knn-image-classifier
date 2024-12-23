import pytest
import numpy as np
from PIL import Image
from src import MNISTPreprocessing

class TestMNISTPreprocessing:
    @pytest.fixture
    def preprocessor(self):
        return MNISTPreprocessing()
    
    def test_correct_output_shape(self, preprocessor):
        # Créer une image test de taille arbitraire
        test_image = np.random.randint(0, 255, (100, 150), dtype=np.uint8)
        test_image = Image.fromarray(test_image)
        
        # Prétraiter l'image
        result = preprocessor.preprocess(test_image)
        
        # Vérifier la forme et le type
        assert result.shape == (28, 28)
        assert result.dtype == np.float32
        assert 0 <= result.min() <= result.max() <= 1.0

    def test_invalid_input(self, preprocessor):
        with pytest.raises(TypeError):
            preprocessor.preprocess(42)  # Un entier n'est pas une image valide