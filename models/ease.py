import numpy as np
from scipy.linalg import cho_factor, cho_solve
from models.common import interactions


class EASE:
    """Closed form diagonal-constrained linear autoencoder."""
    def __init__(self, regularization=100):
        self.regularization = regularization

    def fit(self, frame, n_users, n_items, metadata=None):
        if self.regularization <= 0:
            raise ValueError('EASE requires positive regularization')
        self.history = interactions(frame, n_users, n_items, positive=True)
        gram = (self.history.T @ self.history).toarray()
        gram.flat[::n_items+1] += self.regularization
        inverse = cho_solve(cho_factor(gram), np.eye(n_items))
        self.weights = -inverse / inverse.diagonal()[None, :]
        np.fill_diagonal(self.weights, 0)
        return self

    def score(self, users):
        return self.history[users] @ self.weights
