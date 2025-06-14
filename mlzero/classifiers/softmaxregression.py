import numpy as np
from mlzero.classifiers.baseClassifier import BaseClassifier

class SoftmaxRegression(BaseClassifier):
    """
    SoftmaxRegression implements multinomial logistic regression (softmax regression) for multiclass classification.
    """
    def __init__(self, n_classes, random_state=42, lr=0.01):
        """
        Initialize the SoftmaxRegression model.

        Parameters:
            n_classes (int): Number of classes.
            random_state (int): Seed for reproducibility.
            lr (float): Learning rate for gradient descent.
        """
        self.weights = []
        self.n_classes = n_classes
        self.random_state = random_state
        self.lr = lr

    def fit(self, X, y, epochs=100):
        """
        Fit the SoftmaxRegression model using gradient descent.

        Parameters:
            X (np.ndarray): Input data.
            y (np.ndarray): Target values (integer class labels).
            epochs (int): Number of training epochs.

        Returns:
            None
        """
        X = np.c_[np.ones((X.shape[0], 1)), X]
        n_samples, n_features = X.shape

        # One-hot encode y
        Y_onehot = np.eye(self.n_classes)[y]


        # Initialize weights: shape (n_features, n_classes)
        randomGen = np.random.RandomState(self.random_state)
        self.weights = randomGen.normal(loc=0.0, scale=0.01, size=X.shape[1])

        
        for _ in range(epochs):
            logits = X.dot(self.weights)  # (N, K)
            probs = self.softmax(logits)  # (N, K)

            gradient = (1/n_samples) * X.T.dot(probs - Y_onehot)  # (D+1, K)
            self.weights -= self.lr * gradient

    def softmax(self, Z):
        """
        Compute the softmax function for the input array Z.

        Parameters:
            Z (np.ndarray): Input array of logits.

        Returns:
            np.ndarray: Softmax probabilities.
        """
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # for numerical stability
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def predict_proba(self, X):
        """
        Predict class probabilities for input data X.

        Parameters:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        X = np.c_[np.ones((X.shape[0], 1)), X]
        logits = X.dot(self.weights)
        return self.softmax(logits)

    def predict(self, X):
        """
        Predict class labels for input data X.

        Parameters:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted class labels (integers).
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


