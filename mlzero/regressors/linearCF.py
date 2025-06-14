import numpy as np
from mlzero.regressors.baseRegressor import BaseRegressor
class LinearRegressionCF(BaseRegressor):
    """
    LinearRegressionCF implements linear regression using the closed-form solution (normal equation).
    """
    def __init__(self):
        """
        Initialize the LinearRegressionCF model.
        """
        self.theta = None  # will contain weights and bias

    def fit(self, X, y):
        """
        Fit the LinearRegressionCF model using the normal equation.
        
        Parameters:
            X (np.ndarray): Input data.
            y (np.ndarray): Target values.
        
        Returns:
            None
        """
        # Add a column of ones to X for bias term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # shape (n_samples, n_features + 1)

        # Compute theta using normal equation
        self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def predict(self, X):
        """
        Predict target values for input data X.
        
        Parameters:
            X (np.ndarray): Input data.
        
        Returns:
            np.ndarray: Predicted values.
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)



