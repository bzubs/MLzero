import numpy as np
from ..regressors.baseRegressor import BaseRegressor
class AdaLineGD(BaseRegressor):
    def __init__(self, lr=0.01, n_iter=100, random_state=1):
        self.lr = lr
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        random_gen = np.random.RandomState(self.random_state)
        self.w_ = random_gen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = 0.0
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output

            self.w_ += self.lr * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.lr * 2.0 * errors.sum() / X.shape[0]

            loss = (errors ** 2).mean()
            self.losses_.append(loss)

        return self

    def net_input(self, X):
        return X.dot(self.w_) + self.b_

    def activation(self, X):
        # Identity function for AdaLine
        return X

    def predict(self, X, threshold=0.5):
        return np.where(self.activation(self.net_input(X)) >= threshold, 1, 0)

    def get_params(self):
        return {'lr': self.lr, 'n_iter': self.n_iter, 'random_state': self.random_state}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
