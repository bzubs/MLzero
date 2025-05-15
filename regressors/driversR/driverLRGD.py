import numpy as np
from ..linearGD import LinearRegressionGD

# Sample data
x = np.array([5.0, 4.0, 3.0, 11.0, 7.0, 8.0, 6.0, 14.0, 2.0, 5.0])
y = np.array([1.0, 2.0, 5.0, 7.0, 2.0, 9.0, 12.0, 15.0, 3.0, 8.0])

model = LinearRegressionGD()
model.fit(x, y)

# Loss history
for i, loss in enumerate(model.loss_history):
    print(f"Loss at iteration {i+1}: {loss:.4f}")

print(f"Final weight: {model.weight}")
print(f"Final bias: {model.bias}")
