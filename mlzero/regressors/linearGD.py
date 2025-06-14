import numpy as np
from mlzero.regressors.baseRegressor import BaseRegressor
class LinearRegressionGD(BaseRegressor):
    """
    LinearRegressionGD implements univariate linear regression using gradient descent.
    """
    def __init__(self):
        """
        Initialize the LinearRegressionGD model.
        """
        self.weights = np.array([])  # Initialize weights as a numpy array
        self.bias = 0
        self.loss_history = []
        self.weight_history = []
        self.bias_history = []

    def calc_loss(self, x, y):
        """
        Calculate the mean squared error loss.
        
        Parameters:
            x (np.ndarray): Input data.
            y (np.ndarray): Target values.
        
        Returns:
            float: Mean squared error loss.
        """
        yhat = x * self.weights + self.bias
        loss = np.sum((yhat - y) ** 2) / (2 * len(y))
        return loss

    def fit(self, x, y, epochs=100, lr=0.01):
        """
        Fit the LinearRegressionGD model using gradient descent.
        
        Parameters:
            x (np.ndarray): Input data.
            y (np.ndarray): Target values.
            epochs (int): Number of training epochs.
            lr (float): Learning rate.
        
        Returns:
            None
        """
        n = len(x)
        for i in range(epochs):
            yhat = x * self.weights + self.bias
            dldw = np.sum((yhat - y) * x) / n
            dldb = np.sum((yhat - y)) / n

            self.weights -= lr * dldw
            self.bias -= lr * dldb

            loss = self.calc_loss(x, y)
            self.loss_history.append(loss) 
            self.weight_history.append(self.weights)
            self.bias_history.append(self.bias)

    def predict(self, x):
        """
        Predict target values for input data x.
        
        Parameters:
            x (np.ndarray): Input data.
        
        Returns:
            np.ndarray: Predicted values.
        """
        return x * self.weights + self.bias

