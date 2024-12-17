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
        """
        Prédit la catégorie d'une nouvelle image en utilisant les k plus proches voisins.
        
        Args:
            image: L'image à classifier (numpy array)
            
        Returns:
            La catégorie prédite
            
        Raises:
            ValueError: Si le classificateur n'a pas été entraîné
        """
        if self.training_images is None:
            raise ValueError("Le classificateur doit d'abord être entraîné avec fit()")

        # Calculer les distances avec toutes les images d'entraînement
        distances = []
        for i, training_image in enumerate(self.training_images):
            distance = self.distance_strategy.calculate_distance(image, training_image)
            distances.append((distance, self.labels[i]))
        
        # Trier les distances et prendre les k plus proches
        distances.sort(key=lambda x: x[0])  # Trier par distance
        k_nearest = distances[:self.k]
        
        # Voter pour déterminer la classe
        votes = {}
        for _, label in k_nearest:
            votes[label] = votes.get(label, 0) + 1
        
        # Retourner la classe qui a reçu le plus de votes
        return max(votes.items(), key=lambda x: x[1])[0]