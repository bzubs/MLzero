import numpy as np
from ..linearCF import LinearRegressionCF

def main():
    # Generate a simple dataset
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([1.5, 3.0, 4.5, 6.0, 7.5])

    # Initialize LinearRegressionClosedForm model
    model = LinearRegressionCF()

    # Train the model
    model.fit(X, y)

    # Print the learned parameters (weights and bias)
    print("Theta (weights and bias):", model.theta)

    # Make predictions
    predictions = model.predict(X)
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()
