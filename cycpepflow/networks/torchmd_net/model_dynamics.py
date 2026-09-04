from typing import Optional, Tuple

import os

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

from .modules import CoorsNorm, EquivariantVectorOutput
from .utils import CosineCutoff, NeighborEmbedding, act_class_mapping, rbf_class_mapping


def center(pos, batch):
    pos_center = pos - scatter(pos, batch, dim=0, reduce="mean")[batch]
    return pos_center


def all_pair_graph_geodesic(
    edge_index: Tensor,
    edge_type: Tensor,
    batch: Tensor,
    num_nodes: int,
    max_distance: int,
) -> Tensor:
    """Return padded all-pair covalent shortest-path buckets per molecule.

    Only runtime edge type 1 (the original covalent bond graph) is used. The
    dynamic type-0 radius edges and expanded hop-2/hop-3 edges are excluded, so
    the result is independent of noisy 3D coordinates. Connected distances
    above ``max_distance`` share the final connected bucket; disconnected and
    padded pairs use one extra bucket.
    """
    if max_distance < 1:
        raise ValueError("max_distance must be positive")
    if batch is None:
        batch = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    batch = batch.to(device=edge_index.device, dtype=torch.long)
    num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
    counts = torch.bincount(batch, minlength=num_graphs)
    max_nodes = int(counts.max().item()) if counts.numel() else int(num_nodes)
    if max_nodes <= 0:
        return torch.empty((num_graphs, 0, 0), dtype=torch.long, device=edge_index.device)

    ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    local_index = torch.arange(num_nodes, device=edge_index.device) - ptr[batch]
    inf = max_nodes + 1
    dist = torch.full(
        (num_graphs, max_nodes, max_nodes), inf, dtype=torch.long, device=edge_index.device
    )
    diagonal = torch.arange(max_nodes, device=edge_index.device)
    valid_nodes = diagonal.unsqueeze(0) < counts.unsqueeze(1)
    dist[:, diagonal, diagonal] = torch.where(
        valid_nodes,
        torch.zeros_like(valid_nodes, dtype=torch.long),
        torch.full_like(valid_nodes, inf, dtype=torch.long),
    )

    if edge_type.dim() > 1:
        if edge_type.size(-1) == 1:
            edge_type = edge_type[:, 0]
        else:
            edge_type = edge_type.argmax(dim=-1)
    covalent = edge_type.to(device=edge_index.device).round().long() == 1
    if covalent.any():
        src = edge_index[0, covalent].long()
        dst = edge_index[1, covalent].long()
        graph = batch[src]
        src_local = local_index[src]
        dst_local = local_index[dst]
        dist[graph, src_local, dst_local] = 1
        dist[graph, dst_local, src_local] = 1

    # Dense Floyd--Warshall is computed once per network forward and reused by
    # every global-attention layer.
    for k in range(max_nodes):
        through_k = dist[:, :, k].unsqueeze(-1) + dist[:, k, :].unsqueeze(-2)
        dist = torch.minimum(dist, through_k)

    unreachable = dist >= inf
    dist = dist.clamp(max=max_distance)
    dist = torch.where(unreachable, torch.full_like(dist, max_distance + 1), dist)
    return dist


class ScalarGlobalAttention(nn.Module):
    """Per-molecule all-pairs scalar self-attention with headwise output gating.

    This gated global-attention branch follows the qiuzh20/gated_attention idea:
    after scaled dot-product attention, each query atom/head receives a learned
    sigmoid gate that modulates the head output before head concatenation and
    output projection. It updates invariant scalar node embeddings only; the
    local equivariant TorchMD edge/vector pathway remains unchanged.

    Gate multiplier uses 2*sigmoid(logit), so zero-initialized gate logits start
    at an identity multiplier of 1.0 instead of suppressing the new branch by 0.5.
    The final residual MLP is still zero-initialized so the full model starts
    exactly from the hop1/2/3 no-nonbonded local model and learns global gated
    communication during training.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        activation,
        geodesic_bias: bool = False,
        geodesic_max_distance: int = 32,
    ):
        super().__init__()
        if hidden_channels % num_heads != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by num_heads ({num_heads})"
            )
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        self.headwise_attn_output_gate = True
        self.gate_multiplier_identity_init = True
        self.geodesic_bias = bool(geodesic_bias)
        self.geodesic_max_distance = int(geodesic_max_distance)
        if self.geodesic_bias and self.geodesic_max_distance < 1:
            raise ValueError("geodesic_max_distance must be positive")
        self.norm = nn.LayerNorm(hidden_channels)
        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.gate_proj = nn.Linear(hidden_channels, num_heads)
        self.geodesic_bias_embedding = (
            nn.Embedding(self.geodesic_max_distance + 2, num_heads)
            if self.geodesic_bias
            else None
        )
        self.o_proj = nn.Linear(hidden_channels, hidden_channels)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            activation(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.norm.reset_parameters()
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                layer.bias.data.fill_(0)
        # Zero logits give a 2*sigmoid(0)=1 multiplier, i.e. identity gating.
        self.gate_proj.weight.data.zero_()
        self.gate_proj.bias.data.zero_()
        if self.geodesic_bias_embedding is not None:
            # Start from the parent's unbiased global-attention logits.
            self.geodesic_bias_embedding.weight.data.zero_()
        nn.init.xavier_uniform_(self.out_proj[0].weight)
        self.out_proj[0].bias.data.fill_(0)
        # Zero-init the residual branch so training starts exactly from the
        # local hop1/2/3 model and learns gated global communication if useful.
        self.out_proj[2].weight.data.zero_()
        self.out_proj[2].bias.data.zero_()

    def _attend_padded(
        self,
        padded: Tensor,
        pad_mask: Optional[Tensor],
        geodesic_distance: Optional[Tensor],
    ) -> Tensor:
        h = self.norm(padded)
        bsz, max_nodes, _ = h.shape
        q = self.q_proj(h).view(bsz, max_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(h).view(bsz, max_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(h).view(bsz, max_nodes, self.num_heads, self.head_dim)
        attn_logits = torch.einsum("bqhd,bkhd->bhqk", q, k) * (self.head_dim ** -0.5)
        if self.geodesic_bias:
            if geodesic_distance is None:
                raise ValueError("geodesic_distance is required when geodesic_bias=True")
            if tuple(geodesic_distance.shape) != (bsz, max_nodes, max_nodes):
                raise ValueError(
                    f"geodesic_distance shape {tuple(geodesic_distance.shape)} "
                    f"does not match {(bsz, max_nodes, max_nodes)}"
                )
            buckets = geodesic_distance.to(device=h.device, dtype=torch.long).clamp(
                0, self.geodesic_max_distance + 1
            )
            pair_bias = self.geodesic_bias_embedding(buckets).permute(0, 3, 1, 2)
            attn_logits = attn_logits + pair_bias.to(dtype=attn_logits.dtype)
        if pad_mask is not None:
            attn_logits = attn_logits.masked_fill(pad_mask[:, None, None, :], torch.finfo(attn_logits.dtype).min)
        attn = torch.softmax(attn_logits, dim=-1)
        y = torch.einsum("bhqk,bkhd->bqhd", attn, v)
        gate = 2.0 * torch.sigmoid(self.gate_proj(h)).unsqueeze(-1)  # [B, N, H, 1]
        y = y * gate
        y = y.reshape(bsz, max_nodes, self.hidden_channels)
        y = self.o_proj(y)
        y = self.out_proj(y)
        if pad_mask is not None:
            y = y.masked_fill(pad_mask.unsqueeze(-1), 0)
        return y.to(dtype=padded.dtype)

    def _forward_impl(
        self,
        x: Tensor,
        batch: Optional[Tensor],
        geodesic_distance: Optional[Tensor],
    ) -> Tensor:
        if x.numel() == 0:
            return x
        if batch is None:
            return self._attend_padded(x.unsqueeze(0), None, geodesic_distance).squeeze(0)

        batch = batch.to(device=x.device, dtype=torch.long)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        counts = torch.bincount(batch, minlength=num_graphs)
        max_nodes = int(counts.max().item()) if counts.numel() else x.size(0)
        if max_nodes <= 0:
            return torch.zeros_like(x)

        padded = x.new_zeros((num_graphs, max_nodes, x.size(-1)))
        pad_mask = torch.ones((num_graphs, max_nodes), dtype=torch.bool, device=x.device)
        indices = []
        for graph_idx in range(num_graphs):
            idx = torch.nonzero(batch == graph_idx, as_tuple=False).view(-1)
            indices.append(idx)
            n = idx.numel()
            if n > 0:
                padded[graph_idx, :n] = x[idx]
                pad_mask[graph_idx, :n] = False

        y = self._attend_padded(padded, pad_mask, geodesic_distance)
        out = torch.zeros_like(x)
        for graph_idx, idx in enumerate(indices):
            n = idx.numel()
            if n > 0:
                out[idx] = y[graph_idx, :n].to(dtype=out.dtype)
        return out

    def forward(
        self,
        x: Tensor,
        batch: Optional[Tensor],
        geodesic_distance: Optional[Tensor] = None,
    ) -> Tensor:
        output_dtype = x.dtype
        force_fp32 = os.environ.get("CYCPEPFLOW_GLOBAL_ATTN_FORCE_FP32", "").strip().lower() in {"1", "true", "yes", "on"}
        if force_fp32:
            with torch.autocast(device_type=x.device.type, enabled=False):
                return self._forward_impl(x.float(), batch, geodesic_distance).to(dtype=output_dtype)
        return self._forward_impl(x, batch, geodesic_distance).to(dtype=output_dtype)



class DepthAttentionResidual(nn.Module):
    """Invariant full-depth AttnRes router for CycPepFlow scalar/vector streams.

    Depth-attention weights are computed only from invariant scalar node states.
    The same scalar weights mix the equivariant vector stream, preserving
    equivariance because no raw orientation-dependent vector component is used
    to produce the weights.
    """

    def __init__(self, hidden_channels: int, num_queries: int):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.num_queries = int(num_queries)
        self.key_norm = nn.LayerNorm(hidden_channels)
        self.query = nn.Parameter(torch.empty(num_queries, hidden_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.key_norm.reset_parameters()
        # Small random queries start close to uniform depth mixing but allow each
        # layer/final readout to learn a different source preference.
        nn.init.normal_(self.query, mean=0.0, std=self.hidden_channels ** -0.5)

    def forward(self, query_idx: int, x_sources, vec_sources) -> Tuple[Tensor, Tensor, Tensor]:
        if len(x_sources) != len(vec_sources):
            raise ValueError("x_sources and vec_sources must have the same length")
        if not (0 <= int(query_idx) < self.num_queries):
            raise IndexError(f"query_idx {query_idx} outside [0, {self.num_queries})")
        x_stack = torch.stack(tuple(x_sources), dim=0)      # [S, N, H]
        vec_stack = torch.stack(tuple(vec_sources), dim=0)  # [S, N, 3, H]
        q = self.query[int(query_idx)].to(device=x_stack.device, dtype=x_stack.dtype)
        keys = self.key_norm(x_stack)
        logits = (keys * q.view(1, 1, -1)).sum(dim=-1) * (self.hidden_channels ** -0.5)  # [S, N]
        weights = torch.softmax(logits.float(), dim=0).to(dtype=x_stack.dtype).unsqueeze(-1)  # [S, N, 1]
        x_mix = (weights * x_stack).sum(dim=0)
        vec_mix = (weights.unsqueeze(2) * vec_stack).sum(dim=0)
        return x_mix, vec_mix, weights.squeeze(-1)


class MACEProductLiteResidual(nn.Module):
    """Cheap MACE-inspired product-basis residual for TorchMD scalar/vector streams.

    The branch is deliberately dependency-light: it builds invariant many-body
    features from the current scalar stream, vector-channel squared norms,
    scalar-vector products, transformed node attributes, and CycPepFlow time. It then
    returns a scalar residual and an invariant channel gate on the existing vector
    stream. Because all gates are invariant and the vector update only rescales
    the equivariant vector stream, the output remains equivariant while adding a
    local many-body/product-style pathway every few TorchMD layers.
    """

    def __init__(self, hidden_channels: int, activation):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.x_norm = nn.LayerNorm(hidden_channels)
        self.vec_norm = nn.LayerNorm(hidden_channels)
        self.prod_norm = nn.LayerNorm(hidden_channels)
        self.node_norm = nn.LayerNorm(hidden_channels)
        self.in_proj = nn.Linear(hidden_channels * 4 + 1, hidden_channels)
        self.act = activation()
        self.out_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        self.zero_initialized_residual = True
        self.reset_parameters()

    def reset_parameters(self):
        self.x_norm.reset_parameters()
        self.vec_norm.reset_parameters()
        self.prod_norm.reset_parameters()
        self.node_norm.reset_parameters()
        nn.init.xavier_uniform_(self.in_proj.weight)
        self.in_proj.bias.data.fill_(0)
        self.out_proj.weight.data.zero_()
        self.out_proj.bias.data.zero_()

    def forward(self, x: Tensor, vec: Tensor, t: Optional[Tensor], node_attr: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        if x.numel() == 0:
            return torch.zeros_like(x), torch.zeros_like(vec)
        out_dtype = x.dtype
        vec_norm_sq = (vec.float() * vec.float()).sum(dim=1).to(dtype=out_dtype)
        product_inv = x * vec_norm_sq
        node_hidden = node_attr if node_attr is not None else torch.zeros_like(x)
        if t is None:
            t_feat = x.new_zeros((x.size(0), 1))
        else:
            t_feat = t
            if t_feat.dim() == 1:
                t_feat = t_feat.unsqueeze(-1)
            t_feat = t_feat.to(device=x.device, dtype=out_dtype)
        feats = torch.cat([self.x_norm(x), self.vec_norm(vec_norm_sq), self.prod_norm(product_inv), self.node_norm(node_hidden), t_feat], dim=-1)
        h = self.act(self.in_proj(feats))
        dx_raw, scalar_gate, vector_gate = self.out_proj(h).chunk(3, dim=-1)
        dx = dx_raw * torch.sigmoid(scalar_gate)
        dvec = vec * torch.tanh(vector_gate).unsqueeze(1)
        return dx.to(dtype=out_dtype), dvec.to(dtype=vec.dtype)


class EquivariantMultiHeadAttention(MessagePassing):
    def __init__(
        self,
        hidden_channels: int,
        num_rbf: int,
        distance_influence: str,
        num_heads: int,
        activation: str,
        attn_activation: str,
        cutoff_lower: float,
        cutoff_upper: float,
        node_attr_dim: int = 0,
        qk_norm: bool = False,
        norm_coors: bool = False,
        norm_coors_scale_init: float = 1e-2,
        so3_equivariant: bool = False,
    ):
        super(EquivariantMultiHeadAttention, self).__init__(aggr="add", node_dim=0)
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.so3_equivariant = so3_equivariant
        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads

        self.layernorm = nn.LayerNorm(hidden_channels)
        self.node_attr_dim = node_attr_dim
        self.norm_coors = norm_coors  # boolean
        self.coors_norm = (
            CoorsNorm(scale_init=norm_coors_scale_init) if norm_coors else nn.Identity()
        )
        self.act = activation()
        self.attn_activation = act_class_mapping[attn_activation]()
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        self.qk_norm = qk_norm

        input_channels = (
            hidden_channels + 1 + (hidden_channels if node_attr_dim > 0 else 0)
        )
        self.mixing_mlp = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        if qk_norm:
            # add layer norm to q and k projections
            # based on https://arxiv.org/pdf/2302.05442.pdf
            self.q_proj = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
            )
            self.k_proj = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
            )
        else:
            self.q_proj = nn.Linear(hidden_channels, hidden_channels)
            self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(
            hidden_channels, hidden_channels * (3 + int(so3_equivariant))
        )
        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 3, bias=False)

        # projection linear layers for edge attributes
        self.dk_proj = nn.Linear(num_rbf, hidden_channels)
        self.dv_proj = nn.Linear(num_rbf, hidden_channels * (3 + int(so3_equivariant)))

        self.reset_parameters()

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        if self.qk_norm:
            self.q_proj[0].bias.data.fill_(0)
            nn.init.xavier_uniform_(self.q_proj[0].weight)
            self.k_proj[0].bias.data.fill_(0)
            nn.init.xavier_uniform_(self.k_proj[0].weight)
        else:
            self.q_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.q_proj.weight)
            self.k_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.k_proj.weight)

        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.vec_proj.weight)
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)
        if self.dv_proj:
            nn.init.xavier_uniform_(self.dv_proj.weight)
            self.dv_proj.bias.data.fill_(0)

    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij, t, node_attr):
        # Mix x with node_attr and time
        x = self.mixing_mlp(torch.cat([x, t, node_attr], dim=1))

        # Input features: (num_atoms, hidden_channels)
        x = self.layernorm(x)
        # key/query features: (num_atoms, num_heads, head_dim)
        # where head_dim * num_heads == hidden_channels
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        # value features: (num_atoms, num_heads, 3 * head_dim)
        v = self.v_proj(x).reshape(
            -1, self.num_heads, self.head_dim * (3 + int(self.so3_equivariant))
        )

        # vec features: (num_atoms, 3, hidden_channels) (all invariant)
        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec = vec.reshape(-1, 3, self.num_heads, self.head_dim)
        vec_dot = (vec1 * vec2).sum(dim=1)

        # transform edge attributes (relative distances and user provided edge attributes)
        # into dk and dv vectors with shape (num_edges, num_heads, head_dim)
        # and (num_edges, num_heads, 3 * head_dim) respectively
        dk = self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
        dv = self.act(self.dv_proj(f_ij)).reshape(
            -1, self.num_heads, self.head_dim * (3 + int(self.so3_equivariant))
        )

        # Message Passing Propagate
        x, vec = self.propagate(
            edge_index,  # (2, edges)
            q=q,
            k=k,
            v=v,
            vec=vec,
            dk=dk,
            dv=dv,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
        )
        # new shape: (num_atoms, hidden_channels)
        x = x.reshape(-1, self.hidden_channels)
        # new shape: (num_atoms, 3, hidden_channels)
        vec = vec.reshape(-1, 3, self.hidden_channels)
        # normalize the vec if norm_coors is True
        vec = self.coors_norm(vec)

        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dvec = vec3 * o1.unsqueeze(1) + vec
        dx = vec_dot * o2 + o3
        return dx, dvec

    def message(
        self,
        q_i: Tensor,  # (num_edges, num_heads, head_dim)
        k_j: Tensor,  # (num_edges, num_heads, head_dim)
        v_j: Tensor,  # (num_edges, num_heads, head_dim * 3)
        vec_j: Tensor,  # (num_edges, 3, num_heads, head_dim)
        dk: Tensor,  # (num_edges, num_heads, head_dim)
        dv: Tensor,  # (num_edges, num_heads, head_dim * 3)
        r_ij: Tensor,  # (num_edges,) edge distances
        d_ij: Tensor,  # (num_edges, 3) edge vectors (unit vectors)
    ):
        # dot product attention, a score for each edge
        attn = (q_i * k_j * dk).sum(dim=-1)  # (num_edges, num_heads)

        # apply attention activation function
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

        # value pathway
        v_j = v_j * dv  # multiply with edge attr features

        if self.so3_equivariant:
            x, vec1, vec2, vec3 = torch.split(v_j, self.head_dim, dim=2)
        else:
            x, vec1, vec2 = torch.split(v_j, self.head_dim, dim=2)
            vec3 = None

        # update scalar features
        x = x * attn.unsqueeze(2)  # (num_edges, num_heads, head_dim)
        # update vector features (num_edges, 3, num_heads, head_dim)
        if self.so3_equivariant:
            vec = (
                vec_j * vec1.unsqueeze(1)
                + vec2.unsqueeze(1) * d_ij.unsqueeze(2).unsqueeze(3)
                + vec3.unsqueeze(1)
                * torch.cross(d_ij.unsqueeze(2).unsqueeze(3), vec_j, dim=1)
            )
        else:
            vec = vec_j * vec1.unsqueeze(1) + vec2.unsqueeze(1) * d_ij.unsqueeze(
                2
            ).unsqueeze(3)
        return x, vec

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        # scatter edge-level features (for x and vec) to node-level
        # x shape: (num_atoms, num_heads, head_dim)
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        # vec shape: (num_atoms, 3, num_heads, head_dim)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs


class TorchMD_ET_dynamics(nn.Module):
    r"""The TorchMD equivariant Transformer architecture.

    Parameters
    ----------
    hidden_channels (int, optional): Hidden embedding size.
        (default: :obj:`128`)
    num_layers (int, optional): The number of attention layers.
        (default: :obj:`6`)
    num_rbf (int, optional): The number of radial basis functions :math:`\mu`.
        (default: :obj:`50`)
    rbf_type (string, optional): The type of radial basis function to use.
        (default: :obj:`"expnorm"`)
    trainable_rbf (bool, optional): Whether to train RBF parameters with
        backpropagation. (default: :obj:`True`)
    activation (string, optional): The type of activation function to use.
        (default: :obj:`"silu"`)
    attn_activation (string, optional): The type of activation function to use
        inside the attention mechanism. (default: :obj:`"silu"`)
    neighbor_embedding (bool, optional): Whether to perform an initial neighbor
        embedding step. (default: :obj:`True`)
    num_heads (int, optional): Number of attention heads.
        (default: :obj:`8`)
    distance_influence (string, optional): Where distance information is used inside
        the attention mechanism. (default: :obj:`"both"`)
    cutoff_lower (float, optional): Lower cutoff distance for interatomic interactions.
        (default: :obj:`0.0`)
    cutoff_upper (float, optional): Upper cutoff distance for interatomic interactions.
        (default: :obj:`5.0`)
    max_z (int, optional): Maximum atomic number. Used for initializing embeddings.
        (default: :obj:`100`)
    qk_norm (bool, optional):
        Applies layer norm to q and k projections. Supposed to
        stabilize the training based on
        https://arxiv.org/pdf/2302.05442.pdf. (default: :obj:`False`)
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 6,
        num_rbf: int = 50,
        rbf_type: str = "expnorm",
        trainable_rbf: bool = True,
        activation: str = "silu",
        attn_activation: str = "silu",
        neighbor_embedding: bool = True,
        num_heads: int = 8,
        distance_influence: str = "both",
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 10.0,
        max_z: int = 100,
        node_attr_dim: int = 0,
        edge_attr_dim: int = 0,
        qk_norm: bool = False,
        norm_coors: bool = False,
        norm_coors_scale_init: float = 1e-2,
        clip_during_norm: bool = False,
        so3_equivariant: bool = False,
        global_attention: bool = False,
        global_attention_every_n_layers: int = 1,
        global_attention_residual_scale: float = 1.0,
        global_geodesic_bias: bool = False,
        global_geodesic_max_distance: int = 32,
        depth_attnres: bool = False,
        depth_attnres_final: bool = True,
        mace_product_lite: bool = False,
        mace_product_lite_every_n_layers: int = 5,
        mace_product_lite_residual_scale: float = 1.0,
    ):
        super(TorchMD_ET_dynamics, self).__init__()

        assert distance_influence in ["keys", "values", "both", "none"]
        assert rbf_type in rbf_class_mapping, (
            f'Unknown RBF type "{rbf_type}". '
            f'Choose from {", ".join(rbf_class_mapping.keys())}.'
        )
        assert activation in act_class_mapping, (
            f'Unknown activation function "{activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )
        assert attn_activation in act_class_mapping, (
            f'Unknown attention activation function "{attn_activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.attn_activation = attn_activation
        self.neighbor_embedding = neighbor_embedding
        self.num_heads = num_heads
        self.distance_influence = distance_influence
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_z = max_z
        self.node_attr_dim = node_attr_dim
        self.edge_attr_dim = edge_attr_dim
        self.clip_during_norm = clip_during_norm
        self.global_attention = bool(global_attention)
        self.global_attention_every_n_layers = int(global_attention_every_n_layers)
        self.global_attention_residual_scale = float(global_attention_residual_scale)
        self.global_geodesic_bias = bool(global_geodesic_bias)
        self.global_geodesic_max_distance = int(global_geodesic_max_distance)
        self.use_depth_attnres = bool(depth_attnres)
        self.depth_attnres_final = bool(depth_attnres_final)
        self.mace_product_lite = bool(mace_product_lite)
        self.mace_product_lite_every_n_layers = int(mace_product_lite_every_n_layers)
        self.mace_product_lite_residual_scale = float(mace_product_lite_residual_scale)
        if self.global_attention and self.global_attention_every_n_layers <= 0:
            raise ValueError("global_attention_every_n_layers must be positive")
        if self.global_geodesic_bias and not self.global_attention:
            raise ValueError("global_geodesic_bias requires global_attention")
        if self.global_geodesic_bias and self.global_geodesic_max_distance < 1:
            raise ValueError("global_geodesic_max_distance must be positive")
        if self.mace_product_lite and self.mace_product_lite_every_n_layers <= 0:
            raise ValueError("mace_product_lite_every_n_layers must be positive")

        act_class = act_class_mapping[activation]

        self.embedding = nn.Embedding(self.max_z, self.hidden_channels)

        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )
        self.neighbor_embedding = (
            NeighborEmbedding(
                hidden_channels,
                num_rbf + edge_attr_dim,
                cutoff_lower,
                cutoff_upper,
                self.max_z,
            )
            if neighbor_embedding
            else None
        )

        if self.node_attr_dim > 0:
            self.node_mlp = nn.Sequential(
                nn.Linear(node_attr_dim, hidden_channels),
                act_class(),
                nn.LayerNorm(hidden_channels),
                nn.Linear(hidden_channels, hidden_channels),
            )

        self.attention_layers = nn.ModuleList()
        self.global_attention_layers = nn.ModuleList()
        self.global_attention_layer_mask = []
        self.mace_product_lite_layers = nn.ModuleList()
        self.mace_product_lite_layer_mask = []
        self.depth_attnres = (
            DepthAttentionResidual(hidden_channels, num_layers + 1)
            if self.use_depth_attnres
            else None
        )
        for layer_idx in range(num_layers):
            layer = EquivariantMultiHeadAttention(
                hidden_channels,
                num_rbf + edge_attr_dim,
                distance_influence,
                num_heads,
                act_class,
                attn_activation,
                cutoff_lower,
                cutoff_upper,
                node_attr_dim=node_attr_dim,
                qk_norm=qk_norm,
                norm_coors=norm_coors,
                norm_coors_scale_init=norm_coors_scale_init,
                so3_equivariant=so3_equivariant,
            )  # .jittable() TODO: Removing for now
            self.attention_layers.append(layer)
            use_global_layer = self.global_attention and ((layer_idx + 1) % self.global_attention_every_n_layers == 0)
            self.global_attention_layer_mask.append(use_global_layer)
            self.global_attention_layers.append(
                ScalarGlobalAttention(
                    hidden_channels,
                    num_heads,
                    act_class,
                    geodesic_bias=self.global_geodesic_bias,
                    geodesic_max_distance=self.global_geodesic_max_distance,
                )
                if use_global_layer
                else nn.Identity()
            )
            use_product_lite = self.mace_product_lite and ((layer_idx + 1) % self.mace_product_lite_every_n_layers == 0)
            self.mace_product_lite_layer_mask.append(use_product_lite)
            self.mace_product_lite_layers.append(
                MACEProductLiteResidual(hidden_channels, act_class)
                if use_product_lite
                else nn.Identity()
            )

        self.out_norm = nn.LayerNorm(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        if self.neighbor_embedding is not None:
            self.neighbor_embedding.reset_parameters()
        for attn in self.attention_layers:
            attn.reset_parameters()
        if getattr(self, "global_attention", False):
            for use_global, layer in zip(self.global_attention_layer_mask, self.global_attention_layers):
                if use_global and hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
        if getattr(self, "mace_product_lite", False):
            for use_product, layer in zip(self.mace_product_lite_layer_mask, self.mace_product_lite_layers):
                if use_product and hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
        if getattr(self, "depth_attnres", None) is not None:
            self.depth_attnres.reset_parameters()
        self.out_norm.reset_parameters()

    def forward(
        self,
        z: Tensor,
        t: Tensor,
        pos: Tensor,
        batch: Tensor,
        edge_index: Optional[Tensor] = None,
        node_attr: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        geodesic_distance: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # embed atomic numbers using an embedding layer
        if z.dim() > 1:
            z = z.squeeze()  # (num_atoms,)
        x = self.embedding(z)  # (num_atoms, hidden_channels)

        # append time to node features
        if self.node_attr_dim > 0:
            node_attr = self.node_mlp(node_attr)
        else:
            node_attr = None

        # Compute the static all-pair covalent graph distance once, before the
        # runtime scalar edge type is concatenated with radial features.
        if self.global_geodesic_bias and geodesic_distance is None:
            if edge_attr is None:
                raise ValueError("global_geodesic_bias requires scalar runtime edge types")
            geodesic_distance = all_pair_graph_geodesic(
                edge_index=edge_index,
                edge_type=edge_attr,
                batch=batch,
                num_nodes=pos.size(0),
                max_distance=self.global_geodesic_max_distance,
            )

        # compute distances
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        edge_weight = (edge_vec**2).sum(dim=-1, keepdim=False)

        # update edge_attributes with user input if they are given
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(1)  # (num_edges, 1)
            # (num_edges, num_rbf + edge_attr_dim)
            edge_attr = torch.cat(
                [self.distance_expansion(edge_weight), edge_attr], dim=-1
            )
        else:
            edge_attr = self.distance_expansion(edge_weight)

        mask = edge_index[0] == edge_index[1]
        masked_edge_weight = edge_weight.masked_fill(mask, 1).unsqueeze(1)

        if self.clip_during_norm:
            # clip edge_weight to avoid exploding values if two nodes are close
            masked_edge_weight = masked_edge_weight.clamp(min=1.0e-2)

        edge_vec = edge_vec / masked_edge_weight

        if self.neighbor_embedding is not None:
            x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)

        # vec here is invariant values, we are not modifying the vectors.
        # (num_atoms, 3, hidden_channels)
        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)
        if self.use_depth_attnres:
            x_sources = [x]
            vec_sources = [vec]
        for layer_idx, attn in enumerate(self.attention_layers):
            if self.use_depth_attnres:
                x_in, vec_in, _ = self.depth_attnres(layer_idx, x_sources, vec_sources)
            else:
                x_in, vec_in = x, vec
            dx, dvec = attn(
                x_in,
                vec_in,
                edge_index,
                edge_weight,
                edge_attr,
                edge_vec,
                node_attr=node_attr,
                t=t,
            )
            x = x_in + dx
            vec = vec_in + dvec
            if self.global_attention_layer_mask[layer_idx]:
                x = x + self.global_attention_residual_scale * self.global_attention_layers[layer_idx](
                    x, batch, geodesic_distance=geodesic_distance
                )
            if self.mace_product_lite_layer_mask[layer_idx]:
                dx_prod, dvec_prod = self.mace_product_lite_layers[layer_idx](x, vec, t=t, node_attr=node_attr)
                x = x + self.mace_product_lite_residual_scale * dx_prod
                vec = vec + self.mace_product_lite_residual_scale * dvec_prod
            if self.use_depth_attnres:
                x_sources.append(x)
                vec_sources.append(vec)
        if self.use_depth_attnres and self.depth_attnres_final:
            x, vec, _ = self.depth_attnres(self.num_layers, x_sources, vec_sources)
        x = self.out_norm(x)  # apply layer norm in the end.

        return x, vec, z, pos, batch

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"num_rbf={self.num_rbf}, "
            f"rbf_type={self.rbf_type}, "
            f"trainable_rbf={self.trainable_rbf}, "
            f"activation={self.activation}, "
            f"attn_activation={self.attn_activation}, "
            f"neighbor_embedding={self.neighbor_embedding}, "
            f"num_heads={self.num_heads}, "
            f"distance_influence={self.distance_influence}, "
            f"cutoff_lower={self.cutoff_lower}, "
            f"cutoff_upper={self.cutoff_upper}, "
            f"global_attention={self.global_attention}, "
            f"global_geodesic_bias={self.global_geodesic_bias}, "
            f"depth_attnres={self.use_depth_attnres}, "
            f"mace_product_lite={self.mace_product_lite})"
        )


class TorchMDDynamics(nn.Module):
    r"""
    TorchMDDynamics velocity network for flow-matching inference.

    Parameters
    ----------
    hidden_channels (int, optional):
        Hidden embedding size. (default: :obj:`128`)
    num_layers (int, optional):
        The number of attention layers. (default: :obj:`8`)
    num_rbf (int, optional):
        The number of radial basis functions :math:`\mu`.
        (default: :obj:`64`)
    rbf_type (string, optional):
        The type of radial basis function to use.
        (default: :obj:`"expnorm"`)
    trainable_rbf (bool, optional):
        Whether to train RBF parameters with backpropagation.
        (default: :obj:`False`)
    activation (string, optional):
        The type of activation function to use. (default: :obj:`"silu"`)
    neighbor_embedding (bool, optional):
        Whether to perform an initial neighbor embedding step.
        (default: :obj:`True`)
    cutoff_lower (float, optional):
        Lower cutoff distance for interatomic interactions.
        (default: :obj:`0.0`)
    cutoff_upper (float, optional):
        Upper cutoff distance for interatomic interactions.
        (default: :obj:`5.0`)
    max_z (int, optional):
        Maximum atomic number. Used for initializing embeddings.
        (default: :obj:`100`)
    node_attr_dim (int, optional):
        Dimension of additional input node  features (non-atomic numbers).
    attn_activation (string, optional):
        The type of activation function to use inside the attention
        mechanism. (default: :obj:`"silu"`)
    num_heads (int, optional):
        Number of attention heads. (default: :obj:`8`)
    distance_influence (string, optional):
        Where distance information is used inside the attention
        mechanism. (default: :obj:`"both"`)
    qk_norm (bool, optional):
        Applies layer norm to q and k projections. Supposed to
        stabilize the training based on
        https://arxiv.org/pdf/2302.05442.pdf. (default: :obj:`False`)
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 8,
        num_rbf: int = 64,
        rbf_type: str = "expnorm",
        trainable_rbf: bool = False,
        activation: str = "silu",
        neighbor_embedding: int = True,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 10.0,
        max_z: int = 100,
        node_attr_dim: int = 0,
        edge_attr_dim: int = 0,
        attn_activation: str = "silu",
        num_heads: int = 8,
        distance_influence: str = "both",
        reduce_op: str = "sum",
        qk_norm: bool = False,
        output_layer_norm: bool = True,
        clip_during_norm: bool = False,
        so3_equivariant: bool = False,
        global_attention: bool = False,
        global_attention_every_n_layers: int = 1,
        global_attention_residual_scale: float = 1.0,
        global_geodesic_bias: bool = False,
        global_geodesic_max_distance: int = 32,
        depth_attnres: bool = False,
        depth_attnres_final: bool = True,
        mace_product_lite: bool = False,
        mace_product_lite_every_n_layers: int = 5,
        mace_product_lite_residual_scale: float = 1.0,
    ):
        super().__init__()
        self.representation_model = TorchMD_ET_dynamics(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_rbf=num_rbf,
            rbf_type=rbf_type,
            trainable_rbf=trainable_rbf,
            activation=activation,
            neighbor_embedding=neighbor_embedding,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            max_z=max_z,
            attn_activation=attn_activation,
            num_heads=num_heads,
            distance_influence=distance_influence,
            node_attr_dim=node_attr_dim,
            edge_attr_dim=edge_attr_dim,
            qk_norm=qk_norm,
            clip_during_norm=clip_during_norm,
            so3_equivariant=so3_equivariant,
            global_attention=global_attention,
            global_attention_every_n_layers=global_attention_every_n_layers,
            global_attention_residual_scale=global_attention_residual_scale,
            global_geodesic_bias=global_geodesic_bias,
            global_geodesic_max_distance=global_geodesic_max_distance,
            depth_attnres=depth_attnres,
            depth_attnres_final=depth_attnres_final,
            mace_product_lite=mace_product_lite,
            mace_product_lite_every_n_layers=mace_product_lite_every_n_layers,
            mace_product_lite_residual_scale=mace_product_lite_residual_scale,
        )
        self.output_model = EquivariantVectorOutput(
            hidden_channels=hidden_channels,
            activation=activation,
            reduce_op=reduce_op,
            layer_norm=output_layer_norm,
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()

    def forward(
        self,
        z: Tensor,
        t: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Optional[Tensor] = None,
        node_attr: Optional[Tensor] = None,
        geodesic_distance: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass over torchmd-net model.

        Parameters
        ----------
        z: torch.Tensor
            Atomic numbers, shape (num_atoms,)
        t: torch.Tensor
            Time steps of diffusion, shape (num_atoms,)
        pos: torch.Tensor
            Atomic positions, shape (num_atoms, 3)
        edge_index: torch.Tensor
            Edge index, shape (2, num_edges)
        batch: torch.Tensor, optional
            Batch vector representing which atoms belong to which molecule,
            shape (num_atoms,). If not given, all atoms are assumed to belong
            to the same molecule.
        edge_attr: torch.Tensor, optional
            Edge attributes, shape (num_edges, edge_attr_dim)
        node_attr: torch.Tensor, optional
            Node attributes, shape (num_atoms, node_attr_dim)
        """
        # run the potentially wrapped representation model
        x, v, z, pos, batch = self.representation_model(
            z=z,
            t=t,
            pos=pos,
            batch=batch,
            node_attr=node_attr,
            edge_index=edge_index,
            edge_attr=edge_attr,
            geodesic_distance=geodesic_distance,
        )

        # latent representation
        _, v = self.output_model.pre_reduce(x, v, z, pos, batch)
        return center(v - pos, batch)
