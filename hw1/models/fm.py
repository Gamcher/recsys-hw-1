import numpy as np
import torch


class FMNetwork(torch.nn.Module):
    def __init__(self, n_features, rank, mean):
        super().__init__()
        self.linear = torch.nn.Embedding(n_features, 1, padding_idx=0)
        self.factors = torch.nn.Embedding(n_features, rank, padding_idx=0)
        self.bias = torch.nn.Parameter(torch.tensor(float(mean)))
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.normal_(self.factors.weight, std=0.03)
        with torch.no_grad():
            self.factors.weight[0].zero_()

    def forward(self, fields, weights):
        vectors = self.factors(fields) * weights[:, :, None]
        pairwise = 0.5 * (
            vectors.sum(dim=1).square().sum(dim=1) - vectors.square().sum(dim=(1, 2))
        )
        linear = (self.linear(fields).squeeze(-1) * weights).sum(dim=1)
        return self.bias + linear + pairwise


class FM:
    """Second-order FM; the fitted feature encoder is supplied by the notebook."""

    def __init__(
        self,
        rank=16,
        regularization=1e-4,
        learning_rate=0.01,
        epochs=8,
        batch_size=2048,
        seed=42,
        include_metadata=True,
        include_text=True,
    ):
        self.rank = rank
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.include_metadata = include_metadata
        self.include_text = include_text

    def fit(self, frame, n_users, n_items, metadata=None, *, features=None):
        torch.manual_seed(self.seed)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        if features is None:
            raise ValueError("Pass fitted features from the notebook")
        self.encoder = features
        fields, weights = self.encoder.transform(frame.user_idx, frame.item_idx)
        fields = torch.tensor(fields)
        weights = torch.tensor(weights)
        targets = torch.tensor(frame.overall.to_numpy(), dtype=torch.float32)
        self.network = FMNetwork(self.encoder.n_features, self.rank, targets.mean())
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        generator = torch.Generator().manual_seed(self.seed)
        self.loss_history = []
        for _ in range(self.epochs):
            permutation = torch.randperm(len(frame), generator=generator)
            total = 0.0
            for start in range(0, len(frame), self.batch_size):
                batch_indices = permutation[start : start + self.batch_size]
                batch_fields = fields[batch_indices]
                batch_weights = weights[batch_indices]
                loss = (
                    (self.network(batch_fields, batch_weights) - targets[batch_indices])
                    .square()
                    .mean()
                )
                penalty = (
                    (self.network.factors(batch_fields) * batch_weights[:, :, None])
                    .square()
                    .sum(dim=(1, 2))
                    + (self.network.linear(batch_fields).squeeze(-1) * batch_weights)
                    .square()
                    .sum(dim=1)
                ).mean()
                objective = loss + self.regularization * penalty
                optimizer.zero_grad()
                objective.backward()
                optimizer.step()
                total += float(objective.detach()) * len(batch_indices)
            self.loss_history.append(total / len(frame))
        with torch.no_grad():
            self.feature_factors = self.network.factors.weight.numpy().copy()
            self.linear_weights = self.network.linear.weight.numpy().ravel().copy()
            self.bias = float(self.network.bias)
        item_vectors = (
            self.feature_factors[self.encoder.item_fields]
            * self.encoder.item_weights[:, :, None]
        )
        self.item_factor_sum = item_vectors.sum(axis=1)
        self.item_bias = (
            self.linear_weights[self.encoder.item_fields] * self.encoder.item_weights
        ).sum(axis=1) + 0.5 * (
            (self.item_factor_sum**2).sum(axis=1) - (item_vectors**2).sum(axis=(1, 2))
        )
        self.diagnostics = {
            "metadata": self.include_metadata,
            "description": self.include_text,
            "text_vocab_size": len(self.encoder.text_vocabulary),
            "n_features": self.encoder.n_features,
        }
        return self

    def score(self, users):
        user_features = 1 + np.asarray(users)
        return (
            self.bias
            + self.linear_weights[user_features, None]
            + self.item_bias[None, :]
            + self.feature_factors[user_features] @ self.item_factor_sum.T
        )
