import torch

from neurobpf.gnn.features import FEAT_DIM, GraphTensor
from neurobpf.gnn.model import GAE, GCNConv


def triangle():
    n = 3
    x = torch.tensor(
        [[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]] * n, dtype=torch.float32
    )
    adj = torch.ones((n, n)) - torch.eye(n)
    return x, adj


def star_with_pair():
    hub, leaves = 0, [1, 2, 3, 4, 5]
    adj = torch.zeros((8, 8))
    for leaf in leaves:
        adj[hub, leaf] = 1.0
        adj[leaf, hub] = 1.0
    adj[6, 7] = 1.0
    adj[7, 6] = 1.0
    max_deg = 5.0
    feats = torch.zeros((8, FEAT_DIM))
    for i in range(8):
        feats[i, 0] = 1.0
        deg = 5.0 if i == 0 else 1.0
        feats[i, 4] = deg / max_deg
        feats[i, 5] = deg / max_deg
        feats[i, 6] = (10.0 if i == 0 else 2.0) / 10.0
    return feats, adj


def test_gcnconv_output_shape():
    conv = GCNConv(in_dim=5, out_dim=3)
    x = torch.randn(4, 5)
    adj = torch.eye(4)
    out = conv(x, adj)
    assert out.shape == (4, 3)


def test_gae_reconstructs_triangle():
    torch.manual_seed(0)
    x, adj = triangle()
    model = GAE(in_dim=FEAT_DIM, hidden_dim=16, z_dim=16)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    before = model.train_loss(x, adj).item()
    for _ in range(300):
        opt.zero_grad()
        loss = model.train_loss(x, adj)
        loss.backward()
        opt.step()
    after = model.train_loss(x, adj).item()
    assert after < before * 0.75
    assert after < 1.8


def test_dangling_pair_nodes_score_higher():
    torch.manual_seed(0)
    star_adj = torch.zeros((6, 6))
    for leaf in range(1, 6):
        star_adj[0, leaf] = 1.0
        star_adj[leaf, 0] = 1.0
    star_x = torch.zeros((6, FEAT_DIM))
    for i in range(6):
        star_x[i, 0] = 1.0
        deg = 5.0 if i == 0 else 1.0
        star_x[i, 4] = deg / 5.0
        star_x[i, 5] = deg / 5.0
        star_x[i, 6] = (10.0 if i == 0 else 2.0) / 10.0
    model = GAE(in_dim=FEAT_DIM, hidden_dim=16, z_dim=16)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(250):
        opt.zero_grad()
        loss = model.train_loss(star_x, star_adj)
        loss.backward()
        opt.step()
    model.eval()
    test_x, test_adj = star_with_pair()
    with torch.no_grad():
        scores = model.node_scores(test_x, test_adj)
    scores = scores.flatten().tolist()
    norm_nodes = [0, 1, 2, 3, 4, 5]
    pair_nodes = [6, 7]
    max_norm = max(scores[i] for i in norm_nodes)
    min_pair = min(scores[i] for i in pair_nodes)
    assert min_pair > max_norm
    assert all(scores[i] >= 0.0 for i in range(8))


def test_forward_shape_and_decode_symmetric():
    model = GAE(in_dim=FEAT_DIM, hidden_dim=8, z_dim=8)
    x = torch.randn(5, FEAT_DIM)
    adj = torch.rand(5, 5)
    z = model.encode(x, adj)
    assert z.shape == (5, 8)
    logits = model.reconstruct_logits(x, adj)
    assert logits.shape == (5, 5)
    assert torch.allclose(logits, logits.T, atol=1e-5)
    assert model.forward(x, adj).shape == (5, 8)


def test_gcn_forward_sparse_matches_dense():
    from neurobpf.gnn.dataset import make_adj, make_sparse_adj
    from neurobpf.gnn.features import snapshot_to_graph
    from neurobpf.graphbuilder import EdgeData, NodeData, Snapshot

    nodes = {"hub": NodeData(node_id="hub", kind="process", label="hub", uid=1000, pid=1)}
    edges = []
    for k in range(4):
        nid = f"leaf{k}"
        nodes[nid] = NodeData(node_id=nid, kind="process", label=nid, uid=1000, pid=k + 2)
        edges.append(EdgeData(src="hub", dst=nid, label="EXEC", ts=1.0))
    g = snapshot_to_graph(Snapshot(0.0, nodes, edges, set()), 0)
    model = GAE(in_dim=FEAT_DIM, hidden_dim=8, z_dim=8)
    model.eval()
    with torch.no_grad():
        z_dense = model.encode(g.x, make_adj(g))
        z_sparse = model.encode(g.x, make_sparse_adj(g))
    assert torch.allclose(z_dense, z_sparse, atol=1e-5)