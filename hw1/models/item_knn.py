"""Item-based Pearson CF with user-centered rating prediction."""

import numpy as np
from scipy.sparse import csr_matrix


class ItemKNN:
    def __init__(self, neighbors=40, min_common=3):
        self.neighbors = neighbors
        self.min_common = min_common

    def fit(self, frame, n_users, n_items, metadata=None):
        if self.neighbors < 1 or self.min_common < 2:
            raise ValueError("Use positive neighbors and at least two common ratings")

        ratings = csr_matrix(
            (frame.overall.to_numpy(dtype=float), (frame.user_idx, frame.item_idx)),
            shape=(n_users, n_items),
        )
        observed = ratings.copy()
        observed.data[:] = 1

        counts = np.diff(observed.indptr)
        self.user_means = np.divide(
            np.asarray(ratings.sum(axis=1)).ravel(),
            counts,
            out=np.full(n_users, frame.overall.mean()),
            where=counts > 0,
        )
        self.centered_ratings = ratings.copy()
        self.centered_ratings.data -= np.repeat(self.user_means, counts)
        self.observed = observed

        # Pairwise sufficient statistics use only users who rated both items.
        common = (observed.T @ observed).tocsr()
        products = (ratings.T @ ratings).tocsr()
        sums = (ratings.T @ observed).tocsr()
        square_sums = (ratings.multiply(ratings).T @ observed).tocsr()
        other_sums = sums.T.tocsr()
        other_square_sums = square_sums.T.tocsr()
        statistics = [
            common,
            products,
            sums,
            square_sums,
            other_sums,
            other_square_sums,
        ]
        for matrix in statistics:
            matrix.sort_indices()
            if not (
                np.array_equal(matrix.indptr, common.indptr)
                and np.array_equal(matrix.indices, common.indices)
            ):
                raise RuntimeError("Pairwise statistic supports do not match")

        source_items = []
        target_items = []
        similarities = []
        for target in range(n_items):
            start, end = common.indptr[target : target + 2]
            candidates = common.indices[start:end]
            count = common.data[start:end]
            left_sum = sums.data[start:end]
            right_sum = other_sums.data[start:end]

            covariance = products.data[start:end] - left_sum * right_sum / count
            left_variance = square_sums.data[start:end] - left_sum**2 / count
            right_variance = other_square_sums.data[start:end] - right_sum**2 / count
            denominator = np.sqrt(
                np.maximum(left_variance, 0) * np.maximum(right_variance, 0)
            )
            valid = (
                (count >= self.min_common)
                & (candidates != target)
                & (denominator > 1e-12)
            )
            candidates = candidates[valid]
            correlation = np.clip(covariance[valid] / denominator[valid], -1, 1)
            nonzero = np.abs(correlation) > 1e-12
            candidates, correlation = candidates[nonzero], correlation[nonzero]

            # Keep signed correlations, selecting neighbors by absolute strength.
            order = np.lexsort((candidates, -np.abs(correlation)))[: self.neighbors]
            source_items.extend(candidates[order])
            target_items.extend([target] * len(order))
            similarities.extend(correlation[order])

        self.weights = csr_matrix(
            (similarities, (source_items, target_items)),
            shape=(n_items, n_items),
        )
        self.absolute_weights = abs(self.weights)
        self.diagnostics = {
            "similarity": "Pearson on co-rated users",
            "nnz": self.weights.nnz,
            "negative_weights": int((self.weights.data < 0).sum()),
        }
        return self

    def score(self, users):
        users = np.asarray(users, dtype=int)
        numerator = (self.centered_ratings[users] @ self.weights).toarray()
        denominator = (self.observed[users] @ self.absolute_weights).toarray()
        correction = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
        return self.user_means[users, None] + correction
