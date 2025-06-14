from abc import ABC, abstractmethod
import numpy as np

class BaseClassifier(ABC):
    @abstractmethod
    def fit(self, X, y):
        """Fit the classifier to the training data."""
        pass

    @abstractmethod
    def predict(self, X):
        """Predict class labels for samples in X."""
        pass

    def score(self, X, y):
        """Default implementation of accuracy score."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
