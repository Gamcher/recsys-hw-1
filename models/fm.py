import numpy as np
import torch
from utils.features import ItemFeatures


class _FM(torch.nn.Module):
    def __init__(self,n_features,rank,mean):
        super().__init__()
        self.linear = torch.nn.Embedding(n_features,1)
        self.factors = torch.nn.Embedding(n_features,rank)
        self.bias = torch.nn.Parameter(torch.tensor(float(mean)))
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.normal_(self.factors.weight,std=.03)

    def forward(self, features):
        v = self.factors(features)
        pairs = .5*((v.sum(dim=1)**2).sum(dim=1)-(v**2).sum(dim=(1,2)))
        return self.bias + self.linear(features).sum(dim=(1,2)) + pairs


class FM:
    """Second-order categorical FM; own forward/loss/training and exact scorer."""
    def __init__(self,rank=16,regularization=1e-4,learning_rate=.01,epochs=8,
                 batch_size=4096,seed=42,include_metadata=True):
        self.rank,self.regularization = rank,regularization
        self.learning_rate,self.epochs = learning_rate,epochs
        self.batch_size,self.seed = batch_size,seed
        self.include_metadata = include_metadata

    def fit(self,frame,n_users,n_items,metadata=None):
        torch.manual_seed(self.seed)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        self.encoder = ItemFeatures(self.include_metadata).fit(frame,metadata,n_users,n_items)
        features = torch.tensor(self.encoder.transform(frame.user_idx,frame.item_idx),dtype=torch.long)
        targets = torch.tensor(frame.overall.to_numpy(),dtype=torch.float32)
        self.network = _FM(self.encoder.n_features,self.rank,targets.mean())
        optimizer = torch.optim.Adam(self.network.parameters(),lr=self.learning_rate)
        generator = torch.Generator().manual_seed(self.seed)
        self.loss_history = []
        for _ in range(self.epochs):
            permutation = torch.randperm(len(frame),generator=generator)
            total = 0.
            for start in range(0,len(frame),self.batch_size):
                idx = permutation[start:start+self.batch_size]
                x,y = features[idx],targets[idx]
                prediction = self.network(x)
                mse = ((prediction-y)**2).mean()
                # Explicit per-example feature L2, included in stated objective.
                penalty = (self.network.factors(x).square().sum(dim=(1,2)) +
                           self.network.linear(x).square().sum(dim=(1,2))).mean()
                loss = mse+self.regularization*penalty
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss.detach())*len(idx)
            self.loss_history.append(total/len(frame))
        with torch.no_grad():
            self.v = self.network.factors.weight.numpy().copy()
            self.w = self.network.linear.weight.numpy().ravel().copy()
            self.bias = float(self.network.bias)
        iv = self.v[self.encoder.item_fields]
        self.item_sum = iv.sum(axis=1)
        self.item_constant = self.w[self.encoder.item_fields].sum(axis=1) + .5*(
            (self.item_sum**2).sum(axis=1)-(iv**2).sum(axis=(1,2)))
        return self

    def score(self,users):
        users = np.asarray(users)
        return self.bias+self.w[users,None]+self.item_constant[None,:]+self.v[users] @ self.item_sum.T
