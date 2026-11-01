"""TimeSVD++ with temporal biases/factors and a learned implicit-history sum."""

import numpy as np
import torch
from scipy.sparse import csr_matrix


class TimeNetwork(torch.nn.Module):
    def __init__(self, n_users, n_items, n_days, n_bins, rank, mean):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.tensor(float(mean)))
        self.p = torch.nn.Embedding(n_users, rank)
        self.q = torch.nn.Embedding(n_items, rank)
        self.y = torch.nn.EmbeddingBag(n_items, rank, mode="sum")
        self.user_drift = torch.nn.Embedding(n_users, rank)
        self.day_factors = torch.nn.Embedding(n_days + 1, rank, padding_idx=0)
        self.bu = torch.nn.Embedding(n_users, 1)
        self.bi = torch.nn.Embedding(n_items, 1)
        self.bias_drift = torch.nn.Embedding(n_users, 1)
        self.day_bias = torch.nn.Embedding(n_days + 1, 1, padding_idx=0)
        self.item_bin = torch.nn.Embedding(n_items * n_bins, 1)
        self.n_bins = n_bins
        for table in [self.p, self.q, self.y]:
            torch.nn.init.normal_(table.weight, std=0.03)
        for table in [
            self.user_drift,
            self.day_factors,
            self.bu,
            self.bi,
            self.bias_drift,
            self.day_bias,
            self.item_bin,
        ]:
            torch.nn.init.zeros_(table.weight)

    def forward(
        self,
        users,
        items,
        dev,
        day_ids,
        bins,
        history_items,
        history_offsets,
        history_normalization,
        history_inverse,
    ):
        implicit = (
            self.y(history_items, history_offsets) * history_normalization[:, None]
        )
        profile = (
            self.p(users)
            + self.user_drift(users) * dev[:, None]
            + self.day_factors(day_ids)
            + implicit[history_inverse]
        )
        bias = (
            self.bu(users)
            + self.bias_drift(users) * dev[:, None]
            + self.day_bias(day_ids)
            + self.bi(items)
            + self.item_bin(items * self.n_bins + bins)
        ).squeeze(-1)
        return self.mu + bias + (profile * self.q(items)).sum(dim=1)


class TimeSVDpp:
    def __init__(
        self,
        rank=16,
        regularization=10.0,
        learning_rate=0.005,
        epochs=8,
        batch_size=2048,
        n_bins=30,
        beta=0.4,
        seed=42,
    ):
        self.rank = rank
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_bins = n_bins
        self.beta = beta
        self.seed = seed

    def deviation(self, users, days):
        delta = (np.asarray(days) - self.mean_day[np.asarray(users)]) / 365.0
        return (np.sign(delta) * np.abs(delta) ** self.beta).astype(np.float32)

    def bins(self, days):
        values = (np.asarray(days) - self.min_day) * self.n_bins // self.span
        return np.clip(values, 0, self.n_bins - 1).astype(np.int64)

    def day_ids(self, users, days):
        users = np.asarray(users)
        days = np.broadcast_to(days, users.shape)
        valid = (days >= self.min_day) & (days <= self.max_day)
        keys = users * self.span + (days - self.min_day)
        position = np.searchsorted(self.day_keys, keys)
        safe = np.minimum(position, len(self.day_keys) - 1)
        matched = (
            valid & (position < len(self.day_keys)) & (self.day_keys[safe] == keys)
        )
        return np.where(matched, safe + 1, 0).astype(np.int64)

    def history_pack(self, users):
        parts = [self.histories[int(u)] for u in users]
        lengths = np.array([len(p) for p in parts], dtype=np.int64)
        indices = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
        offsets = np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64)
        norm = np.divide(
            1.0, np.sqrt(lengths), out=np.zeros(len(lengths)), where=lengths > 0
        ).astype(np.float32)
        return torch.tensor(indices), torch.tensor(offsets), torch.tensor(norm)

    def fit(self, frame, n_users, n_items, metadata=None):
        torch.manual_seed(self.seed)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        users = frame.user_idx.to_numpy(dtype=np.int64)
        items = frame.item_idx.to_numpy(dtype=np.int64)
        days = (frame.unixReviewTime.to_numpy() // 86400).astype(np.int64)
        self.min_day = int(days.min())
        self.max_day = int(days.max())
        self.span = self.max_day - self.min_day + 1
        self.reference_time = int(frame.unixReviewTime.max())
        counts = np.bincount(users, minlength=n_users)
        self.mean_day = np.divide(
            np.bincount(users, weights=days, minlength=n_users),
            counts,
            out=np.full(n_users, float(days.mean())),
            where=counts > 0,
        )
        self.day_keys = np.unique(users * self.span + (days - self.min_day))
        history = csr_matrix(
            (np.ones(len(frame)), (frame.user_idx, frame.item_idx)),
            shape=(n_users, n_items),
            dtype=np.float64,
        )
        self.histories = [
            history.indices[history.indptr[u] : history.indptr[u + 1]]
            for u in range(n_users)
        ]
        self.network = TimeNetwork(
            n_users,
            n_items,
            len(self.day_keys),
            self.n_bins,
            self.rank,
            frame.overall.mean(),
        )
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        targets = torch.tensor(frame.overall.to_numpy(), dtype=torch.float32)
        user_tensor, item_tensor = torch.tensor(users), torch.tensor(items)
        dev_tensor = torch.tensor(self.deviation(users, days))
        day_tensor = torch.tensor(self.day_ids(users, days))
        bin_tensor = torch.tensor(self.bins(days))
        generator = torch.Generator().manual_seed(self.seed)
        self.loss_history = []
        for _ in range(self.epochs):
            permutation = torch.randperm(len(frame), generator=generator)
            total = 0.0
            for start in range(0, len(frame), self.batch_size):
                batch_indices = permutation[start : start + self.batch_size]
                batch_users = user_tensor[batch_indices]
                unique_users, history_inverse = torch.unique(
                    batch_users, sorted=True, return_inverse=True
                )
                history_items, history_offsets, history_normalization = (
                    self.history_pack(unique_users.numpy())
                )
                prediction = self.network(
                    batch_users,
                    item_tensor[batch_indices],
                    dev_tensor[batch_indices],
                    day_tensor[batch_indices],
                    bin_tensor[batch_indices],
                    history_items,
                    history_offsets,
                    history_normalization,
                    history_inverse,
                )
                mse = (prediction - targets[batch_indices]).square().mean()
                penalty = sum(
                    parameter.square().sum()
                    for name, parameter in self.network.named_parameters()
                    if name != "mu"
                )
                objective = mse + self.regularization * penalty / len(frame)
                optimizer.zero_grad()
                objective.backward()
                optimizer.step()
                total += float(objective.detach()) * len(batch_indices)
            self.loss_history.append(total / len(frame))
        self._export()
        self.diagnostics = {
            "training_user_days": len(self.day_keys),
            "n_bins": self.n_bins,
            "beta": self.beta,
            "history_entries": int(history.nnz),
            "reference_time": self.reference_time,
        }
        return self

    def _export(self):
        with torch.no_grad():
            for name in [
                "p",
                "q",
                "user_drift",
                "day_factors",
                "bu",
                "bi",
                "bias_drift",
                "day_bias",
            ]:
                setattr(self, name, getattr(self.network, name).weight.numpy().copy())
            self.mu = float(self.network.mu)
            self.item_bin = (
                self.network.item_bin.weight.numpy().reshape(-1, self.n_bins).copy()
            )
            y = self.network.y.weight.numpy()
            self.implicit = np.zeros_like(self.p)
            for u, items in enumerate(self.histories):
                if len(items):
                    self.implicit[u] = y[items].sum(axis=0) / np.sqrt(len(items))

    def score(self, users, when=None):
        users = np.asarray(users, dtype=np.int64)
        day = int((self.reference_time + 86400 if when is None else when) // 86400)
        dev = self.deviation(users, day)
        daily = self.day_ids(users, day)
        profile = (
            self.p[users]
            + self.user_drift[users] * dev[:, None]
            + self.day_factors[daily]
            + self.implicit[users]
        )
        user_bias = (
            self.bu[users]
            + self.bias_drift[users] * dev[:, None]
            + self.day_bias[daily]
        ).ravel()
        return (
            self.mu
            + user_bias[:, None]
            + self.bi.ravel()[None, :]
            + self.item_bin[:, int(self.bins(day))][None, :]
            + profile @ self.q.T
        )
