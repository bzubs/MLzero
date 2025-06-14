import numpy as np
from classifiers.logisticRegression import LogisticRegression
import matplotlib.pyplot as plt

def main():
    # Generate linearly separable synthetic data
    np.random.seed(42)
    n_samples = 50
    X_pos = np.random.randn(n_samples // 2, 2) + np.array([2, 2])  # Positive class centered at (2,2)
    X_neg = np.random.randn(n_samples // 2, 2) + np.array([-2, -2]) # Negative class centered at (-2,-2)
    X = np.vstack((X_pos, X_neg))
    y = np.hstack((np.ones(n_samples // 2), np.zeros(n_samples // 2)))

    # Initialize the Logistic Regression model
    model = LogisticRegression(lr=0.1, random_state=42)

    # Train the model
    model.fit(X, y, epochs=200)

    # Print the learned weights
    print("Learned weights:", model.weights)

    # Make predictions
    predictions = model.predict(np.c_[np.ones((X.shape[0], 1)), X])
    print("Predictions:", predictions)
    print("Actual labels:", y)

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')

    # Plot decision boundary (for 2D features)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x_values = np.linspace(x_min, x_max, 100)
    # Decision boundary: w0 + w1*x1 + w2*x2 = 0 => x2 = -(w0 + w1*x1)/w2
    w = model.weights
    if w[2] != 0:
        y_values = -(w[0] + w[1] * x_values) / w[2]
        plt.plot(x_values, y_values, color='green', label='Decision Boundary')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('Logistic Regression: Data and Decision Boundary')
    plt.show()

if __name__ == "__main__":
    main()
