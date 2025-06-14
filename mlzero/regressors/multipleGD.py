import numpy as np
from mlzero.regressors.baseRegressor import BaseRegressor
class MultipleRegressionGD(BaseRegressor):
    """
    MultipleRegressionGD implements multivariate linear regression using gradient descent.
    """
    def __init__(self, random_state=42):
        """
        Initialize the MultipleRegressionGD model.
        
        Parameters:
            random_state (int): Seed for reproducibility.
        """
        self.weights = np.array([])  # Initialize weights as a numpy array
        self.bias = 0.0
        self.random_state= random_state
        self.loss_history = []
        self.weights_history = []
        self.bias_history = []

    def calc_loss(self, X, y):
        """
        Calculate the mean squared error loss.
        
        Parameters:
            X (np.ndarray): Input data.
            y (np.ndarray): Target values.
        
        Returns:
            float: Mean squared error loss.
        """
        yhat = X.dot(self.weights) + self.bias
        loss = np.sum((yhat - y) ** 2) / (2 * len(y))
        return loss

    def fit(self, X, y, epochs=100, lr=0.01):
        """
        Fit the MultipleRegressionGD model using gradient descent.
        
        Parameters:
            X (np.ndarray): Input data.
            y (np.ndarray): Target values.
            epochs (int): Number of training epochs.
            lr (float): Learning rate.
        
        Returns:
            None
        """
        n = len(y)
        randomGen = np.random.RandomState(self.random_state)
        self.weights = randomGen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        for i in range(epochs):
            yhat = X.dot(self.weights) + self.bias
            dldw = X.T.dot(yhat - y) / n
            dldb = np.sum(yhat - y) / n
            self.weights -= lr * dldw
            self.bias -= lr * dldb
            loss = self.calc_loss(X, y)
            self.loss_history.append(loss)
            self.weights_history.append(self.weights.copy())
            self.bias_history.append(self.bias)

    def predict(self, X):
        """
        Predict target values for input data X.
        
        Parameters:
            X (np.ndarray): Input data.
        
        Returns:
            np.ndarray: Predicted values.
        """
        return X.dot(self.weights) + self.bias

