import numpy as np
from mlzero.regressors.baseRegressor import BaseRegressor

class L2Regressor(BaseRegressor):
    """
    L2Regressor implements Ridge regression using gradient descent.
    """
    def __init__(self, lam=1.0, random_state=42) -> None:
        """
        Initialize the L2Regressor.
        
        Parameters:
            lam (float): Regularization strength.
            random_state (int): Seed for reproducibility.
        """
        self.weights = np.array([])
        self.bias = 0
        self.lam = lam
        self.random_state = random_state
        self.loss_history = []
        self.weights_history = []
        self.bias_history = []

    def calc_loss(self, X, y):
        """
        Calculate the L2-regularized mean squared error loss.
        
        Parameters:
            X (np.ndarray): Input data.
            y (np.ndarray): Target values.
        
        Returns:
            float: The L2-regularized loss value.
        """
        yhat = X.dot(self.weights) + self.bias
        loss = np.sum((yhat- y)**2) + self.lam*np.sum(self.weights**2)
        return loss
    
    def fit(self, X, y, epochs=100, lr=0.01):
        """
        Fit the L2Regressor model using gradient descent.
        
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
        for _ in range(epochs):
            yhat = X.dot(self.weights) + self.bias
            dl_dw = X.T.dot(yhat - y) / n + 2*self.lam*self.weights/n
            dl_db = np.sum(yhat - y) / n
            self.weights -= lr * dl_dw
            self.bias -= lr * dl_db
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

