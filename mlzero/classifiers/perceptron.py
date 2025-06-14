import numpy as np
from mlzero.classifiers.baseClassifier import BaseClassifier

class Perceptron(BaseClassifier):
    """
    Perceptron implements the perceptron algorithm for binary classification.
    """

    def __init__(self, lr=0.01, epochs=50, random_state=42):
        """
        Initialize the Perceptron model.

        Parameters:
            lr (float): Learning rate for weight updates.
            epochs (int): Number of training epochs.
            random_state (int): Seed for reproducibility.
        """
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit the Perceptron model to the training data.

        Parameters:
            X (np.ndarray): Training input data.
            y (np.ndarray): Training target labels.

        Returns:
            self: Fitted estimator.
        """
        randomGen = np.random.RandomState(self.random_state)
        self.w_ = randomGen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.0)
        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0
            for xi, yi in zip(X, y):
                yhat = self.predict(xi)
                update = self.lr * (yi - yhat)
                self.w_ += update * xi
                self.b_ += update
                if update != 0:
                    errors += 1
            self.errors_.append(errors)

        return self

    def predict(self, X):
        """
        Predict binary class labels for input data X.

        Parameters:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted class labels (0 or 1).
        """
        yhat = np.dot(X, self.w_) + self.b_
        ypred = yhat >= 0
        return ypred.astype(int)
