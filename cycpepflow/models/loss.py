"""Loss Functions"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter


def correct_tensor_shape(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 1:
        return t.unsqueeze(1)
    return t


def mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # if shape of predictions is (N,), unsqueeze to (N, 1)
    prediction = correct_tensor_shape(prediction)
    target = correct_tensor_shape(target)
    return ((prediction - target) ** 2).sum(dim=-1).mean(dim=0)


def l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # if shape of predictions is (N,), unsqueeze to (N, 1)
    prediction = correct_tensor_shape(prediction)
    target = correct_tensor_shape(target)

    return (torch.abs(prediction - target)).sum(dim=-1).mean(dim=0)


def l2_loss(prediction: torch.Tensor, target: torch.Tensor):
    # if shape of predictions is (N,), unsqueeze to (N, 1)
    prediction = correct_tensor_shape(prediction)
    target = correct_tensor_shape(target)
    return torch.norm(prediction - target, p=2, dim=-1).mean(dim=0)


def batchwise_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: Optional[torch.Tensor] = None,
    reduce: bool = "mean",
) -> torch.Tensor:
    """Mean Squared Error Loss
    This computes the average MSE loss per molecule and then
    averages over number of molecules in the batch.
    """
    if batch is None:
        batch = torch.zeros(
            size=(prediction.size(0),), dtype=torch.long, device=prediction.device
        )

    # if shape of predictions is (N,), unsqueeze to (N, 1)
    prediction = correct_tensor_shape(prediction)
    target = correct_tensor_shape(target)

    return scatter(
        ((prediction - target) ** 2).sum(dim=-1), index=batch, reduce=reduce
    ).mean(dim=0)


def batchwise_l2_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: Optional[torch.Tensor] = None,
    reduce: bool = "mean",
) -> torch.Tensor:
    if batch is None:
        batch = torch.zeros(
            size=(prediction.size(0),), dtype=torch.long, device=prediction.device
        )

    # if shape of predictions is (N,), unsqueeze to (N, 1)
    prediction = correct_tensor_shape(prediction)
    target = correct_tensor_shape(target)

    return scatter(
        torch.norm(prediction - target, p=2, dim=-1), index=batch, reduce=reduce
    ).mean(dim=0)


def stp_raw_signed_volume(
    pos: torch.Tensor,
    chiral_index: torch.Tensor,
    chiral_nbr_index: torch.Tensor,
) -> torch.Tensor:
    """Raw evaluator-compatible scalar triple products at tetrahedral centers.

    The RINGER/CREMP STP evaluator uses the source-specified center atom and the
    sorted first-three neighbor atom indices. CycPepFlow's PyG data stores center
    indices as ``chiral_index`` and all four neighbors as ``chiral_nbr_index``;
    after PyG batching both tensors are already atom-index offset into the
    concatenated coordinate tensor.
    """
    centers = chiral_index.reshape(-1).long()
    nbrs4 = chiral_nbr_index.reshape(-1, 4).long()
    if centers.numel() == 0 or nbrs4.numel() == 0:
        return pos.new_zeros((0,))

    nbrs3 = torch.sort(nbrs4, dim=1).values[:, :3]
    c = pos[centers]
    v1 = pos[nbrs3[:, 0]] - c
    v2 = pos[nbrs3[:, 1]] - c
    v3 = pos[nbrs3[:, 2]] - c
    return torch.sum(torch.cross(v1, v2, dim=-1) * v3, dim=-1)


def stp_normalized_signed_volume(
    pos: torch.Tensor,
    chiral_index: torch.Tensor,
    chiral_nbr_index: torch.Tensor,
    eps: float = 1.0e-7,
) -> torch.Tensor:
    """Scale-robust STP value in approximately [-1, 1]."""
    centers = chiral_index.reshape(-1).long()
    nbrs4 = chiral_nbr_index.reshape(-1, 4).long()
    if centers.numel() == 0 or nbrs4.numel() == 0:
        return pos.new_zeros((0,))

    nbrs3 = torch.sort(nbrs4, dim=1).values[:, :3]
    c = pos[centers]
    v1 = pos[nbrs3[:, 0]] - c
    v2 = pos[nbrs3[:, 1]] - c
    v3 = pos[nbrs3[:, 2]] - c
    raw = torch.sum(torch.cross(v1, v2, dim=-1) * v3, dim=-1)
    denom = torch.linalg.norm(v1, dim=-1) * torch.linalg.norm(v2, dim=-1) * torch.linalg.norm(v3, dim=-1)
    return raw / denom.clamp_min(eps)


def stp_chirality_loss(
    pos_pred: torch.Tensor,
    pos_ref: torch.Tensor,
    chiral_index: Optional[torch.Tensor],
    chiral_nbr_index: Optional[torch.Tensor],
    scale: float = 1.0,
    eps: float = 1.0e-7,
) -> torch.Tensor:
    """Differentiable STP chirality loss for source-specified centers.

    ``pos_ref`` supplies the desired handedness for the current training
    conformer. ``pos_pred`` is penalized with a softplus margin if its scalar
    triple product has the opposite sign. This is a training objective, not a
    post-hoc coordinate projection.
    """
    if chiral_index is None or chiral_nbr_index is None:
        return pos_pred.new_zeros(())

    ref_norm = stp_normalized_signed_volume(
        pos_ref, chiral_index, chiral_nbr_index, eps=eps
    ).detach()
    if ref_norm.numel() == 0:
        return pos_pred.new_zeros(())

    valid = ref_norm.abs() > eps
    if not bool(valid.any()):
        return pos_pred.new_zeros(())

    pred_norm = stp_normalized_signed_volume(
        pos_pred, chiral_index, chiral_nbr_index, eps=eps
    )[valid]
    target_sign = torch.sign(ref_norm[valid])
    denom = max(float(scale), eps)
    signed_margin = target_sign * pred_norm / denom
    return F.softplus(-signed_margin).mean()
