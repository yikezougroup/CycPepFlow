import math
import os
from itertools import chain
from typing import List, Optional

import torch

_backend = os.environ.get("SOAP_CUDA_LINALG_BACKEND")
if _backend and hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "preferred_linalg_library"):
    try:
        torch.backends.cuda.preferred_linalg_library(_backend)
    except Exception as exc:
        print(f"SOAP_CUDA_LINALG_BACKEND={_backend!r} could not be set: {exc!r}", flush=True)
from torch.optim import Optimizer


def _merge_small_dims(shape, max_dim):
    """Merge adjacent dims while product stays <= max_dim; adapted for SOAP merge_dims."""
    merged = []
    for dim in shape:
        dim = int(dim)
        if not merged or merged[-1] * dim > max_dim:
            merged.append(dim)
        else:
            merged[-1] *= dim
    return tuple(merged)


class SOAP(Optimizer):
    """SOAP: Shampoo preconditioning in an AdamW-style optimizer.

    Minimal standalone implementation adapted from pytorch_optimizer's SOAP
    implementation (Vyas et al. SOAP defaults) so this run does not modify the
    shared Python environment. 1D/non-preconditioned parameters reduce to an
    AdamW-like update; matrix/tensor parameters receive Shampoo eigenbasis
    projection with default preconditioner refresh frequency 10.
    """

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.95, 0.95),
        shampoo_beta: Optional[float] = None,
        weight_decay: float = 0.0,
        precondition_frequency: int = 10,
        max_precondition_dim: int = 4096,
        merge_dims: bool = False,
        precondition_1d: bool = False,
        correct_bias: bool = True,
        normalize_gradient: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if shampoo_beta is not None and not 0.0 <= shampoo_beta < 1.0:
            raise ValueError(f"Invalid shampoo_beta: {shampoo_beta}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if precondition_frequency <= 0:
            raise ValueError("precondition_frequency must be positive")
        if max_precondition_dim <= 0:
            raise ValueError("max_precondition_dim must be positive")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        defaults = dict(
            lr=lr,
            betas=betas,
            shampoo_beta=shampoo_beta,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precondition_dim=max_precondition_dim,
            merge_dims=merge_dims,
            precondition_1d=precondition_1d,
            correct_bias=correct_bias,
            normalize_gradient=normalize_gradient,
            eps=eps,
            maximize=maximize,
            step=0,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _debias(beta: float, step: int) -> float:
        return 1.0 - beta ** step

    @staticmethod
    def _init_preconditioner(
        grad: torch.Tensor,
        state: dict,
        precondition_frequency: int,
        shampoo_beta: float,
        max_precondition_dim: int,
        precondition_1d: bool,
        merge_dims: bool,
    ) -> None:
        state["GG"] = []
        g = grad
        if g.dim() == 1:
            if not precondition_1d or g.shape[0] > max_precondition_dim:
                state["GG"].append([])
            else:
                state["GG"].append(torch.zeros(g.shape[0], g.shape[0], device=g.device, dtype=g.dtype))
        else:
            if merge_dims:
                g = g.reshape(_merge_small_dims(g.size(), max_precondition_dim))
            for sh in g.shape:
                if sh > max_precondition_dim:
                    state["GG"].append([])
                else:
                    state["GG"].append(torch.zeros(sh, sh, device=g.device, dtype=g.dtype))
        state["Q"] = None
        state["precondition_frequency"] = precondition_frequency
        state["shampoo_beta"] = shampoo_beta

    @staticmethod
    def _get_orthogonal_matrix(mats: List[torch.Tensor]) -> List[torch.Tensor]:
        matrices: List = []
        for m in mats:
            if len(m) == 0:
                matrices.append([])
                continue
            # On this cluster/PyTorch build, full DDP training hit a CUDA
            # illegal-memory-access inside torch.linalg.eigh during first SOAP
            # Shampoo-basis setup. Keep eigensolves on CPU, then move the
            # orthogonal basis back to the parameter device for projections.
            device, dtype = m.device, m.dtype
            m_cpu = m.detach().to(device='cpu', dtype=torch.float64)
            eye = torch.eye(m_cpu.shape[0], device=m_cpu.device, dtype=m_cpu.dtype)
            _, q = torch.linalg.eigh(m_cpu + 1e-30 * eye)
            q = torch.flip(q, dims=[1]).to(device=device, dtype=dtype)
            matrices.append(q)
        return matrices

    @staticmethod
    def _project(
        grad: torch.Tensor,
        state: dict,
        merge_dims: bool = False,
        max_precondition_dim: int = 4096,
        project_type: str = "forward",
    ) -> torch.Tensor:
        original_shape = grad.shape
        if merge_dims:
            grad = grad.reshape(_merge_small_dims(grad.size(), max_precondition_dim))
        for mat in state["Q"]:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat, dims=[[0], [0 if project_type == "forward" else 1]])
            else:
                grad = grad.permute([*list(range(1, len(grad.shape))), 0])
        if merge_dims:
            grad = grad.reshape(original_shape)
        return grad

    @staticmethod
    def _get_orthogonal_matrix_qr(state: dict, max_precondition_dim: int = 4096, merge_dims: bool = False):
        """Hybrid safe path: CPU eig for initial Q, but CUDA QR refresh.

        The initial cuSOLVER eigh path corrupts CUDA state in 4-GPU DDP on this
        PyTorch 2.4.1/4090D stack. QR refreshes are much cheaper on GPU and do
        not call eigensolve, so test them separately with synchronization.
        """
        original_shape = state["exp_avg_sq"].shape
        exp_avg_sq = state["exp_avg_sq"]
        if merge_dims:
            exp_avg_sq = exp_avg_sq.reshape(_merge_small_dims(exp_avg_sq.size(), max_precondition_dim))
        matrices = []
        for ind, (m, q_old) in enumerate(zip(state["GG"], state["Q"])):
            if len(m) == 0:
                matrices.append([])
                continue
            device, dtype = m.device, m.dtype
            if device.type == "cuda":
                m_work = m.detach().to(device=device, dtype=torch.float32).contiguous()
                q_old_work = q_old.detach().to(device=device, dtype=torch.float32).contiguous()
                est_eig = torch.diag(q_old_work.T @ m_work @ q_old_work)
                sort_idx = torch.argsort(est_eig, descending=True)
                exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx.to(exp_avg_sq.device))
                power_iter = (m_work @ q_old_work[:, sort_idx]).contiguous()
                q, _ = torch.linalg.qr(power_iter)
                torch.cuda.synchronize(device)
                matrices.append(q.to(device=device, dtype=dtype))
            else:
                est_eig = torch.diag(q_old.T @ m @ q_old)
                sort_idx = torch.argsort(est_eig, descending=True)
                exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)
                power_iter = m @ q_old[:, sort_idx]
                q, _ = torch.linalg.qr(power_iter.to(torch.float32))
                matrices.append(q.to(power_iter.dtype))
        if merge_dims:
            exp_avg_sq = exp_avg_sq.reshape(original_shape)
        state["exp_avg_sq"] = exp_avg_sq
        return matrices

    def _update_preconditioner(
        self,
        grad: torch.Tensor,
        state: dict,
        step: int,
        max_precondition_dim: int,
        precondition_1d: bool,
        merge_dims: bool,
    ) -> None:
        g = grad
        if g.dim() == 1:
            if precondition_1d and g.shape[0] <= max_precondition_dim:
                state["GG"][0].lerp_((g.unsqueeze(1) @ g.unsqueeze(0)).to(state["GG"][0].dtype), weight=1.0 - state["shampoo_beta"])
        else:
            if merge_dims:
                g = g.reshape(_merge_small_dims(g.size(), max_precondition_dim))
            for idx, dim in enumerate(g.shape):
                if dim <= max_precondition_dim:
                    outer_product = torch.tensordot(g, g, dims=[[*chain(range(idx), range(idx + 1, len(g.shape)))]] * 2)
                    state["GG"][idx].lerp_(outer_product.to(state["GG"][idx].dtype), weight=1.0 - state["shampoo_beta"])
        if state["Q"] is None:
            state["Q"] = self._get_orthogonal_matrix(state["GG"])
        if step > 0 and step % state["precondition_frequency"] == 0:
            state["Q"] = self._get_orthogonal_matrix_qr(state, max_precondition_dim, merge_dims)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            shampoo_beta = group["shampoo_beta"] if group["shampoo_beta"] is not None else beta2

            # Lazy state init from current gradients.
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SOAP does not support sparse gradients")
                if torch.is_complex(p):
                    raise RuntimeError("SOAP does not support complex parameters")
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    self._init_preconditioner(
                        grad,
                        state,
                        precondition_frequency=group["precondition_frequency"],
                        shampoo_beta=shampoo_beta,
                        max_precondition_dim=group["max_precondition_dim"],
                        precondition_1d=group["precondition_1d"],
                        merge_dims=group["merge_dims"],
                    )
                    self._update_preconditioner(
                        grad,
                        state,
                        step=group["step"],
                        max_precondition_dim=group["max_precondition_dim"],
                        precondition_1d=group["precondition_1d"],
                        merge_dims=group["merge_dims"],
                    )

            group["step"] += 1
            if group["step"] == 1:
                # Match common SOAP implementations: first gradient initializes Shampoo bases.
                continue

            step_size = group["lr"]
            if group["correct_bias"]:
                step_size *= math.sqrt(self._debias(beta2, group["step"])) / self._debias(beta1, group["step"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if group.get("maximize", False):
                    grad = -grad
                state = self.state[p]

                grad_projected = self._project(
                    grad,
                    state,
                    merge_dims=group["merge_dims"],
                    max_precondition_dim=group["max_precondition_dim"],
                )
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).add_(grad_projected.square(), alpha=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                exp_avg_projected = self._project(
                    exp_avg,
                    state,
                    merge_dims=group["merge_dims"],
                    max_precondition_dim=group["max_precondition_dim"],
                )
                norm_grad = self._project(
                    exp_avg_projected / denom,
                    state,
                    merge_dims=group["merge_dims"],
                    max_precondition_dim=group["max_precondition_dim"],
                    project_type="backward",
                )
                if group["normalize_gradient"]:
                    norm_grad = norm_grad / (norm_grad.square().mean().sqrt() + group["eps"])

                p.add_(norm_grad, alpha=-step_size)
                if group["weight_decay"] != 0.0:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])

                self._update_preconditioner(
                    grad,
                    state,
                    step=group["step"],
                    max_precondition_dim=group["max_precondition_dim"],
                    precondition_1d=group["precondition_1d"],
                    merge_dims=group["merge_dims"],
                )
        return loss
