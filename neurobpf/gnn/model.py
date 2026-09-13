import torch
import torch.nn as nn
import torch.nn.functional as F


def _sym_normalize(adj: torch.Tensor) -> torch.Tensor:
    a = adj + torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
    deg = a.sum(dim=1)
    inv = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    return inv[:, None] * a * inv[None, :]


class GCNConv(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x, adj):
        return adj @ self.lin(x)


class GAE(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, z_dim=64, dropout=0.3, pos_weight=8.0, layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.inter = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(max(layers - 2, 0))]
        )
        self.conv2 = GCNConv(hidden_dim, z_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.pos_weight = pos_weight

    def encode(self, x, adj):
        adj_n = _sym_normalize(adj)
        h = self.act(self.conv1(x, adj_n))
        h = self.drop(h)
        for conv in self.inter:
            h = self.act(conv(h, adj_n))
            h = self.drop(h)
        return self.conv2(h, adj_n)

    def forward(self, x, adj):
        return self.encode(x, adj)

    def decode(self, z):
        return z @ z.T

    def reconstruct_logits(self, x, adj):
        return self.decode(self.encode(x, adj))

    def train_loss(self, x, adj):
        logits = self.reconstruct_logits(x, adj)
        target = (adj > 0).float()
        weight = torch.where(target > 0, self.pos_weight, 1.0)
        return F.binary_cross_entropy_with_logits(
            logits, target, weight=weight, reduction="mean"
        )

    def node_scores(self, x, adj):
        with torch.no_grad():
            logits = self.reconstruct_logits(x, adj)
            sym = 0.5 * logits + 0.5 * logits.T
            target = ((adj > 0) | (adj.T > 0)).float()
            loss = F.binary_cross_entropy_with_logits(
                sym, target, reduction="none"
            )
            n = x.shape[0]
            mask = ~torch.eye(n, dtype=torch.bool)
            return (loss * mask).sum(dim=1) / mask.sum(dim=1)