import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Parameters
SAMPLES = 10
FEATURES = 6
EPOCHS = 100
LR = 0.0001

# Generate random input and output data
X = np.random.randint(1, 101, size=(SAMPLES, FEATURES))
y = np.random.randint(1, 101, size=(SAMPLES, 1))

# Initialize weights and bias
weights = np.random.randint(0, 10, size=(FEATURES, FEATURES)).astype(float)
bias = 0.0

# Loss history for visualization
loss_history = []

def forward(X, weights, bias):
    """
    Custom forward pass for multi-neuron linear model.
    Each row in X (input sample) is multiplied with weights[i % features]
    """
    y_pred = []
    for i in range(SAMPLES):
        weighted_sum = np.dot(X[i], weights[i % FEATURES]) + bias
        y_pred.append(weighted_sum)
    return np.array(y_pred).reshape(-1, 1)

def compute_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2) / 2

# Gradient Descent Training
for epoch in range(EPOCHS):
    y_pred = forward(X, weights, bias)

    error = y_pred - y
    dW = np.zeros_like(weights)
    dB = np.sum(error)

    for i in range(SAMPLES):
        dW[i % FEATURES] += error[i] * X[i]

    weights -= LR * dW / SAMPLES
    bias -= LR * dB / SAMPLES

    loss = compute_loss(y_pred, y)
    loss_history.append(loss)

y_final = forward(X, weights, bias)

print("Input (X):\n", X)
print("\nWeights:\n", weights)
print("\nTarget (y):\n", y.ravel())
print("Predictions:\n", y_final.ravel())
print("\nFinal Loss:", loss_history[-1])

plt.plot(range(EPOCHS), loss_history, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs - MultiNeuronLinearGD")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
