from .strategies.distance import EuclideanDistance
from .strategies.preprocessing import MNISTPreprocessing
from concurrent.futures import ThreadPoolExecutor
import heapq

class ImageClassifier:
    def __init__(self, k=3, distance_strategy=None, preprocessing_strategy=None):
        self.k = k
        self.distance_strategy = distance_strategy or EuclideanDistance()
        self.preprocessing_strategy = preprocessing_strategy or MNISTPreprocessing()
        self.training_images = None
        self.labels = None
    
    def fit(self, images, labels):
        """Entraîne le classificateur avec des images prétraitées"""
        # Vérifions d'abord que les données sont valides
        if len(images) != len(labels):
            raise ValueError("Le nombre d'images et d'étiquettes doit être identique")
        
        # Prétraiter toutes les images d'entraînement
        with ThreadPoolExecutor() as executor:
            self.training_images = list(executor.map(self.preprocessing_strategy.preprocess, images))        
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
        
        # Vérifier que k est valide
        if self.k < 1:
            raise ValueError("k doit être supérieur ou égal à 1")
        if self.k > len(self.training_images):
            raise ValueError("k ne peut pas être supérieur au nombre d'images d'entraînement")
        
        # Prétraiter l'image d'entrée
        processed_image = self.preprocessing_strategy.preprocess(image)

        # Calculer les distances avec toutes les images d'entraînement
        distances = []
        for i, training_image in enumerate(self.training_images):
            distance = self.distance_strategy.calculate_distance(processed_image, training_image)
            distances.append((distance, self.labels[i]))
        
        # Trier les distances et prendre les k plus proches
        k_nearest = heapq.nsmallest(self.k, distances, key=lambda x: x[0])  # Prendre les k plus proches
        
        # Voter pour déterminer la classe
        votes = {}
        for _, label in k_nearest:
            votes[label] = votes.get(label, 0) + 1
        
        # Retourner la classe qui a reçu le plus de votes
        return max(votes.items(), key=lambda x: x[1])[0]