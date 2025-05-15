import numpy as np
from regressors.multipleGD import MultipleRegressionGD


# Sample data
X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]])
y = np.array([5.0, 7.0, 9.0, 11.0, 13.0])

model = MultipleRegressionGD()
model.fit(X, y)

# Loss history
for i, loss in enumerate(model.loss_history):
    print(f"Loss at iteration {i+1}: {loss:.4f}")

print(f"Final weights: {model.weights}")
print(f"Final bias: {model.bias}")
