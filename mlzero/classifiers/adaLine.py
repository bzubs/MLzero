import numpy as np
from mlzero.classifiers.baseClassifier import BaseClassifier
class AdaLineGD(BaseClassifier):
    """
    AdaLineGD implements the Adaptive Linear Neuron (Adaline) algorithm using gradient descent.
    """
    def __init__(self, lr=0.01, n_iter=100, random_state=42):
        """
        Initialize the AdaLineGD model.

        Parameters:
            lr (float): Learning rate for gradient descent.
            n_iter (int): Number of training iterations.
            random_state (int): Seed for reproducibility.
        """
        self.lr = lr
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit the AdaLineGD model using gradient descent.

        Parameters:
            X (np.ndarray): Input data.
            y (np.ndarray): Target values.

        Returns:
            self: Fitted estimator.
        """
        random_gen = np.random.RandomState(self.random_state)
        self.w_ = random_gen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = 0.0
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output

            self.w_ += self.lr * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.lr * 2.0 * errors.sum() / X.shape[0]

            loss = (errors ** 2).mean()
            self.losses_.append(loss)

        return self

    def net_input(self, X):
        """
        Calculate the net input (weighted sum plus bias).

        Parameters:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Net input values.
        """
        return X.dot(self.w_) + self.b_

    def activation(self, X):
        """
        Compute the activation (identity function for AdaLine).

        Parameters:
            X (np.ndarray): Net input values.

        Returns:
            np.ndarray: Activated values (same as input).
        """
        return X

    def predict(self, X, threshold=0.5):
        """
        Predict binary class labels for input data X.

        Parameters:
            X (np.ndarray): Input data.
            threshold (float): Threshold for classifying as positive class.

        Returns:
            np.ndarray: Predicted class labels (0 or 1).
        """
        return np.where(self.activation(self.net_input(X)) >= threshold, 1, 0)

    def get_params(self):
        """
        Get parameters of the AdaLineGD model.

        Returns:
            dict: Model parameters.
        """
        return {'lr': self.lr, 'n_iter': self.n_iter, 'random_state': self.random_state}

    def set_params(self, **params):
        """
        Set parameters of the AdaLineGD model.

        Parameters:
            **params: Model parameters to set.

        Returns:
            self: Updated estimator.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
