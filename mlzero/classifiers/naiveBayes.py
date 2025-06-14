import numpy as np
from mlzero.classifiers.baseClassifier import BaseClassifier


class NaiveBayesClassifier(BaseClassifier):
    """
    NaiveBayesClassifier implements Gaussian Naive Bayes classification.
    """
    def fit(self, X, y):
        """
        Fit the Naive Bayes model to the training data.

        Parameters:
            X (np.ndarray): Training input data.
            y (np.ndarray): Training target labels.

        Returns:
            None
        """
        # Get the number of samples and features
        n_samples, n_features = X.shape
        # Find all unique class labels
        self.classes = np.unique(y)
        n_classes = len(self.classes)


        # Initialize mean, variance, and prior arrays for each class and feature
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)


        # Calculate mean, variance, and prior for each class
        for idx, c in enumerate(self.classes):
            X_current = X[c == y]  # Select samples belonging to class c
            self.mean[idx, :] = X_current.mean(axis=0)  # Mean of each feature for class c
            self.var[idx, :] = X_current.var(axis=0)    # Variance of each feature for class c
            self.priors[idx] = X_current.shape[0] / float(n_samples)  # Prior probability P(y=c)

    def predict(self, X):
        """
        Predict class labels for the input data X.

        Parameters:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted class labels.
        """
        # Predict the class label for each sample in X
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)


    def _predict(self, x):
        """
        Compute the posterior probability for each class for a single sample.

        Parameters:
            x (np.ndarray): Input sample.

        Returns:
            int or float: Predicted class label for the sample.
        """
        # Compute the posterior probability for each class
        posteriors = []
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])  # Log prior probability
            class_condn = np.sum(np.log(self.class_conditional(idx, x)))  # Log likelihood
            posterior = prior + class_condn  # Posterior (up to a constant)
            posteriors.append(posterior)
        # Return the class with the highest posterior probability
        return self.classes[np.argmax(posteriors)]



    def class_conditional(self, class_idx, x):
        """
        Calculate the probability of x given class_idx using Gaussian likelihood.

        Parameters:
            class_idx (int): Index of the class.
            x (np.ndarray): Input sample.

        Returns:
            np.ndarray: Class-conditional probabilities for each feature.
        """
        # Calculate the probability of x given class_idx using Gaussian likelihood
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        # Gaussian probability density function for each feature
        numerator = np.exp(-(x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator