"""Full-catalog macro evaluation with deterministic ties and known-item masks."""
import numpy as np
from models.common import interactions


def cohort(holdout, known, n_items):
    if n_items < 1 or not holdout.item_idx.between(0, n_items-1).all():
        raise ValueError('Holdout item outside candidate catalogue')
    relevant = holdout[holdout.overall >= 4]
    seen = set(zip(known.user_idx, known.item_idx))
    truth = {}
    for row in relevant.itertuples():
        if (row.user_idx, row.item_idx) not in seen:
            truth.setdefault(int(row.user_idx), set()).add(int(row.item_idx))
    if not truth:
        raise ValueError('Empty evaluation cohort')
    return dict(sorted(truth.items()))


def topk(scores, known, k=10):
    scores = np.array(scores, dtype=float, copy=True)
    if scores.ndim != 2 or scores.shape != known.shape:
        raise ValueError('Scores and known-item mask must have the same 2D shape')
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError('K must be a positive integer')
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite predictions')
    rows, cols = known.nonzero()
    scores[rows, cols] = -np.inf
    if (np.isfinite(scores).sum(axis=1) < k).any():
        raise ValueError('Fewer than K eligible candidates')
    return np.argsort(-scores, axis=1, kind='stable')[:, :k]


def ranking_metrics(recommendations, relevant, k=10):
    recommendations = np.asarray(recommendations)
    relevant = list(relevant)
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError('K must be a positive integer')
    if recommendations.ndim != 2 or recommendations.shape != (len(relevant), k) or not relevant:
        raise ValueError('Expected one K-item recommendation list per user')
    if any(not truth for truth in relevant):
        raise ValueError('Each evaluated user needs relevant items')
    if any(len(set(row)) != k for row in recommendations):
        raise ValueError('Recommendations must contain unique items')
    recalls, ndcgs = [], []
    discount = 1 / np.log2(np.arange(2, k+2))
    for recs, truth in zip(recommendations, relevant):
        hit = np.array([item in truth for item in recs], dtype=float)
        recalls.append(hit.sum()/len(truth))
        ndcgs.append(float(hit @ discount) / discount[:min(k,len(truth))].sum())
    return {f'NDCG@{k}': float(np.mean(ndcgs)), f'Recall@{k}': float(np.mean(recalls))}


def evaluate(model, truth, known, n_users, n_items, batch_size=64):
    if not truth or batch_size < 1:
        raise ValueError('Evaluation needs users and positive batch size')
    if any(u < 0 or u >= n_users or not items or any(i < 0 or i >= n_items for i in items)
           for u, items in truth.items()):
        raise ValueError('Invalid evaluation user or relevant item')
    known_matrix = interactions(known, n_users, n_items)
    users = np.array(sorted(truth))
    sums = {'NDCG@10': 0., 'Recall@10': 0.}
    for start in range(0,len(users),batch_size):
        batch = users[start:start+batch_size]
        recs = topk(model.score(batch), known_matrix[batch])
        metrics = ranking_metrics(recs,[truth[u] for u in batch])
        for key in sums:
            sums[key] += metrics[key]*len(batch)
    return {**{k: v/len(users) for k,v in sums.items()}, 'users': len(users)}
