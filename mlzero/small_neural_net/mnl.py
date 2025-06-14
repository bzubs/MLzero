import numpy as np

class MultiNeuronLinearGD:
    """
    MultiNeuronLinearGD implements a simple multi-neuron linear model with gradient descent.
    """
    def __init__(self, samples, features, lr=0.0001, epochs=100, random_state=42):
        """
        Initialize the model parameters.
        
        Parameters:
            samples (int): Number of samples in the dataset.
            features (int): Number of features per sample.
            lr (float): Learning rate for gradient descent.
            epochs (int): Number of training epochs.
            random_state (int): Seed for reproducibility.
        """
        np.random.seed(random_state)
        self.samples = samples
        self.features = features
        self.lr = lr
        self.epochs = epochs
        self.weights = np.random.randint(0, 10, size=(features, features)).astype(float)
        self.bias = 0.0
        self.loss_history = []

    def forward(self, X):
        """
        Perform the forward pass for the multi-neuron linear model.
        
        Parameters:
            X (np.ndarray): Input data of shape (samples, features).
        
        Returns:
            np.ndarray: Predicted outputs of shape (samples, 1).
        """
        y_pred = []
        for i in range(self.samples):
            weighted_sum = np.dot(X[i], self.weights[i % self.features]) + self.bias
            y_pred.append(weighted_sum)
        return np.array(y_pred).reshape(-1, 1)

    def compute_loss(self, y_pred, y_true):
        """
        Compute the mean squared error loss.
        
        Parameters:
            y_pred (np.ndarray): Predicted outputs.
            y_true (np.ndarray): True target values.
        
        Returns:
            float: Mean squared error loss.
        """
        return np.mean((y_pred - y_true) ** 2) / 2

    def fit(self, X, y):
        """
        Train the model using gradient descent.
        
        Parameters:
            X (np.ndarray): Input data of shape (samples, features).
            y (np.ndarray): Target values of shape (samples, 1).
        
        Returns:
            None
        """
        for epoch in range(self.epochs):
            y_pred = self.forward(X)
            error = y_pred - y
            dW = np.zeros_like(self.weights)
            dB = np.sum(error)
            for i in range(self.samples):
                dW[i % self.features] += error[i] * X[i]
            self.weights -= self.lr * dW / self.samples
            self.bias -= self.lr * dB / self.samples
            loss = self.compute_loss(y_pred, y)
            self.loss_history.append(loss)

    def predict(self, X):
        """
        Predict outputs for the given input data.
        
        Parameters:
            X (np.ndarray): Input data of shape (samples, features).
        
        Returns:
            np.ndarray: Predicted outputs of shape (samples, 1).
        """
        return self.forward(X)
