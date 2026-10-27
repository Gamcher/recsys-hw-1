"""Shared data representations, no trained third-party recommender code."""
import numpy as np
from scipy.sparse import csr_matrix


def interactions(frame, n_users, n_items, positive=False):
    rows = frame[frame.overall >= 4] if positive else frame
    return csr_matrix((np.ones(len(rows)), (rows.user_idx, rows.item_idx)),
                      shape=(n_users, n_items), dtype=np.float64)
