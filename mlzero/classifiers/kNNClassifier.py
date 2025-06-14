import numpy as np
from mlzero.classifiers.baseClassifier import BaseClassifier
from collections import Counter

class kNNClassifier(BaseClassifier):
    """
    kNNClassifier implements the k-Nearest Neighbors classification algorithm.
    """
    def __init__(self, k=5):
        """
        Initialize the kNNClassifier.

        Parameters:
            k (int): Number of neighbors to use.
        """
        self.n_neighbors = k
        self.X_train = []
        self.y_train = []

    def calc_distance(self, pointA, pointB):
        """
        Calculate the Euclidean distance between two points.

        Parameters:
            pointA (np.ndarray): First point.
            pointB (np.ndarray): Second point.

        Returns:
            float: Euclidean distance.
        """
        return np.linalg.norm(pointA - pointB)
    
    def majority_count(self, neighbor_indices):
        """
        Determine the most common class label among the neighbors.

        Parameters:
            neighbor_indices (list): Indices of the nearest neighbors.

        Returns:
            int or float: Most common class label among neighbors.
        """
        labels = [self.y_train[i] for i in neighbor_indices]
        label_counts = Counter(labels)
        return label_counts.most_common(1)[0][0]

    def fit(self, X_train, y_train):
        """
        Store the training data for kNN classification.

        Parameters:
            X_train (np.ndarray): Training input data.
            y_train (np.ndarray): Training target labels.

        Returns:
            None
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        Predict class labels for the test data.

        Parameters:
            X_test (np.ndarray): Test input data.

        Returns:
            np.ndarray: Predicted class labels.
        """
        predictions = []

        for test_point in X_test:
            distances = []
            for train_point in self.X_train:
                distances.append(self.calc_distance(test_point, train_point))

            dist_neigh = sorted(list(enumerate(distances)), key=lambda x: x[1])[:self.n_neighbors]
            neighbor_indices = [index for index, _ in dist_neigh]

            prediction = self.majority_count(neighbor_indices)
            predictions.append(prediction)

        return np.array(predictions)
