import numpy as np
from mlzero.regressors.ElasticNet import ElasticNet

def main():
    # Generate sample data
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([1.5, 3.0, 4.5, 6.0, 7.5])  # Linear relationship

    # Initialize the ElasticNet model
    model = ElasticNet(random_state=42, lam=0.1, alpha=0.5)

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
