import numpy as np
from scipy.sparse import csr_matrix


class ExplicitALS:
    """Observed-entry squared error plus unweighted L2 on both factor matrices."""
    def __init__(self, rank=16, regularization=1.0, iterations=8, seed=42):
        self.rank, self.regularization = rank, regularization
        self.iterations, self.seed = iterations, seed

    def target(self, frame):
        return frame.overall.to_numpy(dtype=float)

    def fit(self, frame, n_users, n_items, metadata=None):
        target = self.target(frame)
        matrix = csr_matrix((target, (frame.user_idx, frame.item_idx)), shape=(n_users, n_items))
        # Zero targets remain observed entries: do not eliminate sparse zeros.
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, .1, (n_users, self.rank))
        self.item_factors = rng.normal(0, .1, (n_items, self.rank))
        eye = self.regularization * np.eye(self.rank)
        def update(observed, other):
            factors = np.zeros((observed.shape[0], self.rank))
            for row in range(observed.shape[0]):
                start, end = observed.indptr[row:row+2]
                features = other[observed.indices[start:end]]
                if end > start:
                    factors[row] = np.linalg.solve(features.T @ features + eye,
                                                   features.T @ observed.data[start:end])
            return factors
        self.loss_history = []
        for _ in range(self.iterations):
            self.user_factors = update(matrix, self.item_factors)
            self.item_factors = update(matrix.T.tocsr(), self.user_factors)
            pred = np.einsum('ij,ij->i', self.user_factors[frame.user_idx], self.item_factors[frame.item_idx])
            loss = np.sum((pred-target)**2) + self.regularization * (
                np.sum(self.user_factors**2)+np.sum(self.item_factors**2))
            self.loss_history.append(float(loss))
        return self

    def score(self, users):
        return self.user_factors[np.asarray(users)] @ self.item_factors.T
