import numpy as np
from ..adaLine import AdaLineGD

def main():
    # Generate a simple dataset
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([1, 0, 1, 0, 1])

    # Initialize AdaLineGD model
    model = AdaLineGD(lr=0.01, n_iter=100, random_state=42)

    # Train the model
    model.fit(X, y)

    # Print the learned weights and bias
    print("Weights:", model.w_)
    print("Bias:", model.b_)

    # Make predictions
    predictions = model.predict(X)
    print("Predictions:", predictions)

    # Print the loss over iterations
    print("Losses:", model.losses_)

if __name__ == "__main__":
    main()
