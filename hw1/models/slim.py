import numpy as np
from scipy.sparse import csc_matrix, csr_matrix


class SLIM:
    """Own nonnegative elastic-net coordinate descent; no library estimator.

    Exact safe screening: G_ij <= l1 implies b_ij=0 because G and b are
    nonnegative. It is not a top-neighbor cap or a popularity filter.
    """

    def __init__(self, l1=1.0, l2=20.0, tolerance=1e-5, max_epochs=500):
        self.l1 = l1
        self.l2 = l2
        self.tolerance = tolerance
        self.max_epochs = max_epochs

    def fit(self, frame, n_users, n_items, metadata=None):
        if self.l1 < 0 or self.l2 <= 0:
            raise ValueError("SLIM needs l1>=0 and l2>0")
        positive = frame.loc[frame.overall >= 4]
        self.history = csr_matrix(
            (np.ones(len(positive)), (positive.user_idx, positive.item_idx)),
            shape=(n_users, n_items),
            dtype=np.float64,
        )
        gram = (self.history.T @ self.history).tocsc()
        row_idx, values, indptr = [], [], [0]
        epochs_used = []
        residuals = []
        nonconverged = 0
        objective = 0.5 * gram.diagonal().sum()
        for j in range(n_items):
            start, end = gram.indptr[j : j + 2]
            idx = gram.indices[start:end]
            target = gram.data[start:end]
            eligible = (idx != j) & (target > self.l1)
            idx, g = idx[eligible], target[eligible]
            if not len(idx):
                indptr.append(len(values))
                continue
            block = np.array(gram[idx][:, idx].toarray(), order="F")
            diagonal = block.diagonal().copy()
            b = np.zeros(len(idx))
            residual = g.copy()
            threshold = self.tolerance * (1 + g.max())
            violation = np.inf
            for epoch in range(self.max_epochs):
                for k in range(len(idx)):
                    new = max(
                        0.0,
                        (residual[k] + diagonal[k] * b[k] - self.l1)
                        / (diagonal[k] + self.l2),
                    )
                    delta = new - b[k]
                    if delta:
                        b[k] = new
                        residual -= block[:, k] * delta
                gradient = -residual + self.l2 * b + self.l1
                violation = max(
                    np.max(np.abs(gradient[b > 0]), initial=0),
                    np.max(np.maximum(-gradient[b == 0], 0), initial=0),
                )
                if violation <= threshold:
                    break
            if violation > threshold:
                nonconverged += 1
            epochs_used.append(epoch + 1)
            residuals.append(float(violation / threshold))
            objective += (
                -g @ b
                + 0.5 * b @ block @ b
                + self.l1 * b.sum()
                + 0.5 * self.l2 * (b @ b)
            )
            active = b > 0
            row_idx.extend(idx[active])
            values.extend(b[active])
            indptr.append(len(values))
        self.weights = csc_matrix(
            (values, row_idx, indptr), shape=(n_items, n_items)
        ).tocsr()
        self.diagnostics = {
            "nonconverged_columns": nonconverged,
            "max_scaled_kkt": max(residuals, default=0),
            "mean_epochs": float(np.mean(epochs_used)) if epochs_used else 0.0,
            "nnz": self.weights.nnz,
            "objective": float(objective),
        }
        if nonconverged:
            raise RuntimeError(f"SLIM did not converge for {nonconverged} columns")
        return self

    def score(self, users):
        return (self.history[users] @ self.weights).toarray()
