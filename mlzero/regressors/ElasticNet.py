import numpy as np
from mlzero.regressors.baseRegressor import BaseRegressor

class ElasticNet(BaseRegressor):
    """
    ElasticNet implements Elastic Net regression using gradient descent.
    """
    def __init__(self, random_state=42, lam=1.0, alpha=0.5) -> None:
        """
        Initialize the ElasticNet regressor.
        
        Parameters:
            random_state (int): Seed for reproducibility.
            lam (float): Regularization strength.
            alpha (float): Mixing parameter between L1 and L2 regularization.
        """
        self.weights=np.array([])
        self.bias=0
        self.lam = lam
        self.random_state= random_state
        self.alpha = alpha
        self.loss_history = []
        self.weights_history = []
        self.bias_history = []

    def calc_loss(self, X, y):
        """
        Calculate the Elastic Net-regularized mean squared error loss.
        
        Parameters:
            X (np.ndarray): Input data.
            y (np.ndarray): Target values.
        
        Returns:
            float: The Elastic Net-regularized loss value.
        """
        n = y.shape[0]
        yhat = X.dot(self.weights) + self.bias
        mse = np.sum((yhat - y) ** 2) / n
        reg_term = self.lam * (self.alpha * np.sum(np.abs(self.weights)) + (1 - self.alpha) * np.sum(self.weights ** 2))
        loss = mse + reg_term
        return loss

    
    def fit(self, X, y, epochs=100, lr=0.01):
        """
        Fit the ElasticNet model using gradient descent.
        
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
            dl_dw = X.T.dot(yhat - y) / n + self.lam * (self.alpha*np.sign(self.weights) + (1-self.alpha)*self.weights)
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