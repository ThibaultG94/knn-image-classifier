from .strategies.distance import EuclideanDistance

class ImageClassifier:
    def __init__(self, k=3, distance_strategy=None):
        self.k = k
        self.distance_strategy = distance_strategy or EuclideanDistance()
        self.training_images = None
        self.labels = None
    
    def fit(self, images, labels):
        """Entraîne le classificateur avec des exemples"""
        # Vérifions d'abord que les données sont valides
        if len(images) != len(labels):
            raise ValueError("Le nombre d'images et d'étiquettes doit être identique")
        
        self.training_images = images
        self.labels = labels
    
    def predict(self, image):
        """Prédit la catégorie d'une nouvelle image"""
        if self.training_images is None:
            raise ValueError("Le classificateur doit d'abord être entraîné avec fit()")
        
        # La logique de prédiction viendra ici
        pass