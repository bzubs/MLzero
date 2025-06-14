import numpy as np
from classifiers.kNNClassifier import kNNClassifier
import matplotlib.pyplot as plt

def main():
    # Generate linearly separable synthetic data
    np.random.seed(42)
    n_samples = 50
    X_pos = np.random.randn(n_samples // 2, 2) + np.array([2, 2])  # Positive class centered at (2,2)
    X_neg = np.random.randn(n_samples // 2, 2) + np.array([-2, -2]) # Negative class centered at (-2,-2)
    X = np.vstack((X_pos, X_neg))
    y = np.hstack((np.ones(n_samples // 2), np.zeros(n_samples // 2)))

    # Shuffle the data
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Split into train and test sets
    split = int(0.7 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Initialize and train the kNN classifier
    model = kNNClassifier(k=5)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)
    print("Predictions:", predictions)
    print("Actual labels:", y_test)
    print("Accuracy:", np.mean(predictions == y_test))

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='red', label='Train Class 0', marker='o')
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='blue', label='Train Class 1', marker='o')
    plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color='red', label='Test Class 0', marker='x')
    plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='blue', label='Test Class 1', marker='x')
    plt.title('kNN Classifier: Train/Test Split')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
