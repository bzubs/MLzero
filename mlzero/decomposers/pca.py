import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA) for dimensionality reduction.
    """
    def __init__(self, n_components):
        """
        Initialize the PCA model.

        Parameters:
            n_components (int): Number of principal components to keep.
        """
        self.n_components = n_components
        self.components = []
        self.mean = None

    def fit(self, X):
        """
        Fit the PCA model to the data X.

        Parameters:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            None
        """
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        cov = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[:, idxs]  # reorder columns
        self.components = eigenvectors[:, :self.n_components]  # keep as columns

    def transform(self, X):
        """
        Project the data X onto the principal components.

        Parameters:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Transformed data of shape (n_samples, n_components).
        """
        X = X - self.mean
        return np.dot(X, self.components)  # no need to transpose
