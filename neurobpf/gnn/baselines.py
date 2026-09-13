import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import OneClassSVM


class _AutoEncoder(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.enc = nn.Linear(in_dim, hidden)
        self.dec = nn.Linear(hidden, in_dim)

    def forward(self, x):
        h = F.relu(self.enc(x))
        return self.dec(h)


def ocsvm_scores(features):
    clf = OneClassSVM(nu=0.1)
    clf.fit(features)
    return -clf.decision_function(features)


def mlp_ae_scores(features, hidden=32, epochs=100, seed=0):
    torch.manual_seed(seed)
    x = torch.as_tensor(features, dtype=torch.float32)
    model = _AutoEncoder(x.shape[1], hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = F.mse_loss(model(x), x)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        err = F.mse_loss(model(x), x, reduction="none").mean(dim=1).numpy()
    return err