import numpy as np
from scipy.sparse import csr_matrix
from models.common import interactions


class ItemKNN:
    """Cosine of positive histories, co-occurrence shrinkage, top neighbors."""
    def __init__(self, neighbors=50, shrinkage=10):
        self.neighbors, self.shrinkage = neighbors, shrinkage

    def fit(self, frame, n_users, n_items, metadata=None):
        self.history = interactions(frame, n_users, n_items, positive=True)
        gram = (self.history.T @ self.history).toarray()
        norms = np.sqrt(gram.diagonal())
        denom = norms[:, None] * norms[None, :]
        similarity = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 0)
        if self.shrinkage:
            similarity *= gram / (gram + self.shrinkage)
        np.fill_diagonal(similarity, 0)
        # For each source item retain its nearest neighbors; stable ASIN-index ties.
        order = np.argsort(-similarity, axis=1, kind='stable')[:, :min(self.neighbors, n_items-1)]
        values = np.take_along_axis(similarity, order, axis=1)
        self.weights = csr_matrix((values.ravel(),
                                  (np.repeat(np.arange(n_items), order.shape[1]), order.ravel())),
                                 shape=(n_items, n_items))
        self.weights.eliminate_zeros()
        return self

    def score(self, users):
        return (self.history[users] @ self.weights).toarray()
