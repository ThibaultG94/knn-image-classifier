# KNN Image Classifier 🖼️

A Python implementation of k-Nearest Neighbors algorithm for image classification. Built for educational purposes with step-by-step implementation of core machine learning concepts.

## Features

- Simple yet powerful k-NN implementation for image classification
- Pure Python/NumPy implementation for educational purposes
- Easy-to-use interface similar to scikit-learn
- Comprehensive visualization tools
- Detailed performance metrics

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ThibaultG94/knn-image-classifier.git
cd knn-image-classifier
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.knn import ImageClassifier

# Initialize classifier
classifier = ImageClassifier(k=3)

# Train the model
classifier.fit(training_images, labels)

# Make predictions
prediction = classifier.predict(new_image)

# Evaluate performance
accuracy = classifier.evaluate(test_images, test_labels)
```

## Project Structure

```
knn-image-classifier/
├── data/               # Training and test data
├── src/               # Source code
│   ├── knn.py         # Core algorithm
│   ├── image.py       # Image processing
│   └── utils.py       # Utility functions
├── tests/             # Unit tests
├── notebooks/         # Jupyter notebooks
└── requirements.txt   # Dependencies
```

## Development

### Running Tests

```bash
pytest tests/
```

### Contributing

1. Fork the repository
2. Create your feature branch
3. Make your changes
4. Run the tests
5. Submit a pull request

## Technologies Used

- Python 3.x
- NumPy for computations
- Pillow for image processing
- Matplotlib for visualization
- scikit-learn for evaluation metrics

## License

This project is licensed under the MIT License - see the LICENSE file for details.
