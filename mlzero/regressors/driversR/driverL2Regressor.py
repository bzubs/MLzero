import numpy as np
from mlzero.regressors.L2Regressor import L2Regressor

def main():
    # Generate sample data
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([1.5, 3.0, 4.5, 6.0, 7.5])  # Linear relationship

    # Initialize the L2 Regressor model
    model = L2Regressor(lam=0.1, random_state=42)

    # Train the model
    model.fit(X, y, epochs=1000, lr=0.01)

    # Print the learned weights and bias
    print("Learned weights:", model.weights)
    print("Learned bias:", model.bias)

    # Print the final loss
    print("Final loss:", model.loss_history[-1])

    # Make predictions
    predictions = model.predict(X)
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()
