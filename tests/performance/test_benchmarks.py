import pytest
import numpy as np
from src import ImageClassifier
import time

class TestPerformance:
    @pytest.fixture
    def large_dataset(self):
        return [np.random.rand(28, 28) for _ in range(1000)], [0] * 1000

    def test_parallel_processing(self, large_dataset):
        images, labels = large_dataset
        start = time.time()
        classifier = ImageClassifier(k=3)
        classifier.fit(images, labels)
        parallel_time = time.time() - start
        
        assert parallel_time < 5.0  # Un seuil raisonnable à ajuster