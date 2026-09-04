from typing import Tuple

import torch
from torch_geometric.utils import get_laplacian, scatter, to_dense_adj

from cycpepflow.commons.utils import extend_graph_order_radius


def center_of_mass(x, dim=0, batch=None):
    num_nodes = x.size(0)

    if batch is None:
        batch = torch.zeros(num_nodes, dtype=torch.long, device=x.device)

    x_com = scatter(x, batch, dim=dim, reduce="mean")[batch]
    return x - x_com


def _topological_hop_edges(
    bond_index: torch.Tensor,
    batch: torch.Tensor,
    num_nodes: int,
    device: torch.device,
    max_hop: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a static topological graph with edge types 1/2/3.

    Edge-type contract for this ablation:
      1 = covalent bond / one graph hop
      2 = shortest-path two-hop relation in the covalent graph
      3 = shortest-path three-hop relation in the covalent graph

    Dynamic radius/nonbonded edges are deliberately not added, so type 0 is absent.
    RDKit bond-order edge_attr, if present upstream, is ignored here so "covalent"
    is a single category independent of single/double/aromatic bond order.
    """
    bond_index = bond_index.to(device=device, dtype=torch.long)
    if batch is None:
        batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    else:
        batch = batch.to(device=device, dtype=torch.long)

    if num_nodes == 0 or bond_index.numel() == 0:
        empty_idx = torch.empty((2, 0), dtype=torch.long, device=device)
        empty_type = torch.empty((0,), dtype=torch.long, device=device)
        return empty_idx, empty_type

    edge_indices = []
    edge_types = []
    if bond_index.numel() > 0:
        edge_indices.append(bond_index)
        edge_types.append(torch.ones(bond_index.shape[1], dtype=torch.long, device=device))

    if max_hop < 2:
        return torch.cat(edge_indices, dim=1), torch.cat(edge_types, dim=0)

    num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
    counts = torch.bincount(batch, minlength=num_graphs)
    max_nodes = int(counts.max().item()) if counts.numel() else num_nodes
    ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])

    # Dense per-molecule adjacency is small for CREMP peptides and avoids radius edges.
    adj1 = to_dense_adj(bond_index, batch=batch, max_num_nodes=max_nodes).bool()
    local = torch.arange(max_nodes, device=device)
    valid = local.unsqueeze(0) < counts.unsqueeze(1)
    valid_pair = valid.unsqueeze(1) & valid.unsqueeze(2)
    eye = torch.eye(max_nodes, dtype=torch.bool, device=device).unsqueeze(0)
    adj1 = adj1 & valid_pair & (~eye)

    adj1_f = adj1.float()
    adj2_any = torch.bmm(adj1_f, adj1_f) > 0
    hop2 = adj2_any & valid_pair & (~eye) & (~adj1)

    hop_adjs = [(2, hop2)]
    if max_hop >= 3:
        adj3_any = torch.bmm(adj2_any.float(), adj1_f) > 0
        hop3 = adj3_any & valid_pair & (~eye) & (~adj1) & (~hop2)
        hop_adjs.append((3, hop3))

    for hop_type, hop_adj in hop_adjs:
        graph, src, dst = hop_adj.nonzero(as_tuple=True)
        base = ptr[graph]
        edge_indices.append(torch.stack([base + src, base + dst], dim=0))
        edge_types.append(torch.full((src.numel(),), hop_type, dtype=torch.long, device=device))

    if not edge_indices:
        empty_idx = torch.empty((2, 0), dtype=torch.long, device=device)
        empty_type = torch.empty((0,), dtype=torch.long, device=device)
        return empty_idx, empty_type
    return torch.cat(edge_indices, dim=1), torch.cat(edge_types, dim=0)


def extend_bond_index(
    pos: torch.Tensor,
    bond_index: torch.Tensor,
    batch: torch.Tensor,
    bond_attr: torch.Tensor,
    device: torch.device,
    one_hot: bool = False,
    one_hot_types: int = 5,
    cutoff: float = 10.0,
    max_num_neighbors: int = 32,
    topology: Tuple[torch.Tensor, torch.Tensor] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Hop-1/2/3 + nonbonded ablation: keep the charge+chirality-only node
    # schema and the static topological 1/2/3-hop covalent graph from the
    # previous run, but re-enable dynamic radius/nonbonded edges as type 0.
    # RDKit bond-order edge_attr, if present upstream, is still ignored so
    # covalent bonds are one category independent of single/double/aromatic
    # order. Edge types passed to the network are:
    #   0 = dynamic radius/nonbonded edge, 1 = covalent bond,
    #   2 = shortest-path 2-hop, 3 = shortest-path 3-hop.
    if topology is None:
        topology = _topological_hop_edges(bond_index, batch, pos.size(0), device)
    edge_index, edge_type = topology
    assert int((edge_type == 1).sum().item()) == bond_index.shape[1], (
        "Type-1 covalent edge count should match input molecular bond_index before radius extension."
    )

    edge_index, edge_type = extend_graph_order_radius(
        pos=pos,
        edge_index=edge_index,
        edge_type=edge_type,
        batch=batch,
        cutoff=cutoff,
        max_num_neighbors=max_num_neighbors,
        extend_radius=True,
    )
    unique_types = set(int(v) for v in edge_type.unique().tolist()) if edge_type.numel() else set()
    assert unique_types.issubset({0, 1, 2, 3}), f"Unexpected runtime edge types: {sorted(unique_types)}"
    assert int((edge_type == 1).sum().item()) == bond_index.shape[1], (
        "Type-1 covalent edge count should match input molecular bond_index after radius extension."
    )

    if one_hot:
        edge_type = torch.nn.functional.one_hot(
            edge_type, num_classes=one_hot_types + 1
        ).float()

    return edge_index, edge_type

def unsqueeze_like(x: torch.Tensor, target: torch.Tensor):
    shape = (x.size(0), *([1] * (target.dim() - 1)))
    return x.view(shape)


"""
Following code adapted from HarmonicFlow
https://github.com/HannesStark/FlowSite/blob/main/utils/diffusion.py
"""


class HarmonicSampler:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.eig_val_cache = {}
        self.eig_vec_cache = {}

    def diagonalize(self, n_nodes, edges=[], batch=None, smiles=None):
        a = self.alpha * torch.ones((edges.shape[0],), device=edges.device)
        edge_index, edge_weight = get_laplacian(
            edges.T,
            a,
            num_nodes=n_nodes,
        )

        H = to_dense_adj(
            edge_index=edge_index, edge_attr=edge_weight, max_num_nodes=n_nodes
        ).squeeze()

        if batch is None:
            D, P = torch.linalg.eigh(H)
            return D, P

        Ds, Ps = [], []
        batch_size = batch.max() + 1

        for i in range(batch_size):
            idx = torch.where(batch == i)[0]
            start = idx.min()
            end = idx.max() + 1

            D, P = None, None
            if smiles is not None:
                D, P = self.check_cache(smiles[i])

                if (D is not None) and (P is not None):
                    D = D.to(edge_index.device)
                    P = P.to(edge_index.device)

            if (D is None) or (P is None):
                D, P = torch.linalg.eigh(H[start:end, start:end])

                if smiles is not None:
                    self.eig_val_cache[smiles[i]] = D.cpu()
                    self.eig_vec_cache[smiles[i]] = P.cpu()

            Ds.append(D)
            Ps.append(P)

        return torch.cat(Ds), torch.block_diag(*Ps)

    def check_cache(self, smiles):
        D = self.eig_val_cache.get(smiles, None)
        P = self.eig_vec_cache.get(smiles, None)
        return D, P

    def sample(self, size, edge_index, batch=None, smiles=None):
        # transpose if (2, n_edges)
        if edge_index.size(0) == 2:
            edge_index = edge_index.T

        n_nodes = size[0]
        D, P = self.diagonalize(
            n_nodes=n_nodes, edges=edge_index, batch=batch, smiles=smiles
        )

        # get starting index per sample in batch
        start_index = 0
        if batch is not None:
            _, counts = torch.unique(batch, return_counts=True)
            cum_sum = counts.cumsum(0)[:-1]
            zero = torch.zeros(1).to(D.device)
            start_index = torch.concat((zero, cum_sum)).long()

        std = 1.0 / torch.sqrt(D)
        std[start_index] = 0.0

        noise = torch.randn(size).to(D.device)
        noise = std[:, None] * noise
        noise[noise.isnan()] = 0.0
        sample = P @ (noise)

        return sample
