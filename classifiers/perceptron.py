import numpy as np

class Perceptron:

    def __init__(self, lr=0.01, epochs=50, random_state=42):
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y):
        randomGen = np.random.RandomState(self.random_state)
        self.w_ = randomGen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.0)
        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0
            for xi, yi in zip(X, y):
                yhat = self.predict(xi)
                update = self.lr * (yi - yhat)
                self.w_ += update * xi
                self.b_ += update
                if update != 0:
                    errors += 1
            self.errors_.append(errors)

        return self

    def predict(self, X):
        yhat = np.dot(X, self.w_) + self.b_
        ypred = yhat >= 0
        return ypred.astype(int)
