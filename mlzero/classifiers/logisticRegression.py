import numpy as np
from mlzero.classifiers.baseClassifier import BaseClassifier


class LogisticRegression(BaseClassifier):
    """
    LogisticRegression implements logistic regression for binary classification using gradient descent.
    """

    def __init__(self, lr=0.01, random_state=42):
        """
        Initialize the LogisticRegression model.

        Parameters:
            lr (float): Learning rate for gradient descent.
            random_state (int): Seed for reproducibility.
        """
        self.weights = []
        self.lr = lr
        self.random_state = random_state

    @staticmethod
    def sigmoid(X):
        """
        Compute the sigmoid function.

        Parameters:
            X (np.ndarray): Input array.

        Returns:
            np.ndarray: Output after applying sigmoid function.
        """
        return 1 / (1 + np.exp(-X))

    def fit(self, X, y, epochs=50):
        """
        Fit the LogisticRegression model using gradient descent.

        Parameters:
            X (np.ndarray): Input data.
            y (np.ndarray): Target values.
            epochs (int): Number of training epochs.

        Returns:
            None
        """
        X = np.c_[np.ones(shape=(X.shape[0], 1)), X]

        randomGen = np.random.RandomState(self.random_state)
        self.weights = randomGen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        m = X.shape[0]

        for _ in range(epochs):
            ypred = self.predict_proba(X)

            dl_dw = (y - ypred).dot(X)
            dl_dw = -dl_dw / m
            self.weights = self.weights - self.lr * (dl_dw)

    def predict_proba(self, X):
        """
        Predict probability estimates for input data X.

        Parameters:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Probability estimates for the positive class.
        """
        return self.sigmoid(X.dot(self.weights))

    def predict(self, X, threshold=0.5):
        """
        Predict binary class labels for input data X.

        Parameters:
            X (np.ndarray): Input data.
            threshold (float): Threshold for classifying as positive class.

        Returns:
            np.ndarray: Predicted class labels (0 or 1).
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)





