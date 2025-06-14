import numpy as np
from mlzero.regressors.polylinear import PolynomialRegression

def main():
    # Generate sample data
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([1.0, 4.0, 9.0, 16.0, 25.0])  # Quadratic relationship

    # Initialize the Polynomial Regression model
    model = PolynomialRegression(n_degree=2, random_state=42)

    # Train the model
    model.fit(X, y, epochs=1000, lr=0.01)

    # Print the learned weights
    print("Learned weights:", model.weights)

    # Print the loss history
    print("Final loss:", model.loss_history[-1])

    # Make predictions
    predictions = model.predict(X)
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()
