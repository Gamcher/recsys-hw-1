"""Explicit ALS: squared error on observed ratings and constant L2."""

import numpy as np
from scipy.sparse import csr_matrix


class ExplicitALS:
    def __init__(self, rank=16, regularization=1.0, iterations=8, seed=42):
        self.rank = rank
        self.regularization = regularization
        self.iterations = iterations
        self.seed = seed

    def _update_factors(self, observed, fixed_factors):
        factors = np.zeros((observed.shape[0], self.rank))
        penalty = self.regularization * np.eye(self.rank)
        for row in range(observed.shape[0]):
            start, end = observed.indptr[row : row + 2]
            if start == end:
                continue
            features = fixed_factors[observed.indices[start:end]]
            target = observed.data[start:end]
            factors[row] = np.linalg.solve(
                features.T @ features + penalty,
                features.T @ target,
            )
        return factors

    def _objective(self, frame, target):
        predicted = np.einsum(
            "ij,ij->i",
            self.user_factors[frame.user_idx],
            self.item_factors[frame.item_idx],
        )
        squared_error = np.sum((predicted - target) ** 2)
        penalty = self.regularization * (
            np.sum(self.user_factors**2) + np.sum(self.item_factors**2)
        )
        return float(squared_error + penalty)

    def fit(self, frame, n_users, n_items, metadata=None):
        target = frame.overall.to_numpy(dtype=float)
        observed = csr_matrix(
            (target, (frame.user_idx, frame.item_idx)),
            shape=(n_users, n_items),
        )
        transposed = observed.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, 0.1, (n_users, self.rank))
        self.item_factors = rng.normal(0, 0.1, (n_items, self.rank))
        self.loss_history = []
        for _ in range(self.iterations):
            self.user_factors = self._update_factors(observed, self.item_factors)
            self.item_factors = self._update_factors(transposed, self.user_factors)
            self.loss_history.append(self._objective(frame, target))
        return self

    def score(self, users):
        return self.user_factors[np.asarray(users)] @ self.item_factors.T
