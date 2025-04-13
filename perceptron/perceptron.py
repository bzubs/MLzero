import numpy as np

class SingleNeuronGD:
    def __init__(self):
        self.weight = 0.0
        self.bias = 0.0
        self.loss_history = []
        self.weight_history = []
        self.bias_history = []

    def calc_loss(self, x, y):
        yhat = x * self.weight + self.bias
        loss = np.sum((yhat - y) ** 2) / (2 * len(y))
        return loss

    def gradient_descent(self, x, y, epochs=100, lr=0.01):
        n = len(x)
        for i in range(epochs):
            yhat = x * self.weight + self.bias
            dldw = np.sum((yhat - y) * x) / n
            dldb = np.sum((yhat - y)) / n

            self.weight -= lr * dldw
            self.bias -= lr * dldb

            loss = self.calc_loss(x, y)
            self.loss_history.append(loss)
            self.weight_history.append(self.weight)
            self.bias_history.append(self.bias)

    def predict(self, x):
        return x * self.weight + self.bias

# Sample data
x = np.array([5.0, 4.0, 3.0, 11.0, 7.0, 8.0, 6.0, 14.0, 2.0, 5.0])
y = np.array([1.0, 2.0, 5.0, 7.0, 2.0, 9.0, 12.0, 15.0, 3.0, 8.0])

model = SingleNeuronGD()
model.gradient_descent(x, y)

# Loss history
for i, loss in enumerate(model.loss_history):
    print(f"Loss at iteration {i+1}: {loss:.4f}")

# Uncomment to view final weights and bias
# print(f"Final weight: {model.weight}")
# print(f"Final bias: {model.bias}")
