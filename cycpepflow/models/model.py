from os import PathLike
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from cycpepflow.commons.utils import signed_volume
from cycpepflow.models import utils as flow_utils
from cycpepflow.models.utils import (
    HarmonicSampler,
    center_of_mass,
    extend_bond_index,
    unsqueeze_like,
)
from cycpepflow.networks.torchmd_net import TorchMDDynamics
from cycpepflow.networks.torchmd_net import model_dynamics

__all__ = ["BaseFlow"]

Config = str | PathLike[str] | Dict[str, Any]

# Archived optimizer/scheduler options remain valid config metadata. They do not
# construct training objects or affect network initialization or sampling.
_ARCHIVED_TRAINING_OPTIONS = frozenset(
    {
        'optimizer_type',
        'lr',
        'beta1',
        'beta2',
        'weight_decay',
        'ams_grad',
        'grad_norm_max_val',
        'lr_scheduler_type',
        'factor',
        'patience',
        'first_cycle_steps',
        'cycle_mult',
        'max_lr',
        'min_lr',
        'warmup_steps',
        'gamma',
        'last_epoch',
        'lr_scheduler_monitor',
        'lr_scheduler_interval',
        'lr_scheduler_frequency',
    }
)


class BaseFlow(torch.nn.Module):
    """Checkpoint-compatible flow-matching inference model."""

    __prior_types__ = ["gaussian", "harmonic"]

    def __init__(
        self,
        # flow matching network args
        network_type: str = "TorchMDDynamics",
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
        output_layer_norm: bool = False,
        clip_during_norm: bool = False,
        max_num_neighbors: int = 32,
        so3_equivariant: bool = False,
        # optional scalar all-pairs attention branch; keeps local equivariant graph attention intact
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
        # Flow/prior options. Training-only path/loss values are accepted below
        # for archived configuration compatibility, without being stored or used.
        sigma: float = 0.1,
        prior_type: str = "gaussian",
        sample_time_dist: str = "uniform",
        harmonic_alpha: float = 1.0,
        parity_switch: Optional[str] = None,
        chirality_loss_weight: float = 0.0,
        chirality_loss_scale: float = 1.0,
        chirality_loss_eps: float = 1.0e-7,
        chirality_warmup_steps: int = 0,
        chirality_ramp_steps: int = 0,
        # make edge_type one_hot
        edge_one_hot: bool = False,
        edge_one_hot_types: int = 5,
        **kwargs,
    ):
        unknown_options = set(kwargs) - _ARCHIVED_TRAINING_OPTIONS
        if unknown_options:
            names = ", ".join(sorted(unknown_options))
            raise TypeError(f"Unexpected model option(s): {names}")
        super().__init__()
        # setup network
        if network_type == "TorchMDDynamics":
            self.network = TorchMDDynamics(
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
                node_attr_dim=node_attr_dim,
                edge_attr_dim=edge_attr_dim,
                attn_activation=attn_activation,
                num_heads=num_heads,
                distance_influence=distance_influence,
                reduce_op=reduce_op,
                qk_norm=qk_norm,
                output_layer_norm=output_layer_norm,
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
        else:
            raise NotImplementedError(f"Network {network_type} not implemented.")

        self.cutoff = cutoff_upper
        self.parity_switch = parity_switch
        self.prior_type = prior_type
        self.edge_one_hot = edge_one_hot
        self.edge_one_hot_types = edge_one_hot_types
        self.max_num_neighbors = max_num_neighbors
        # Inference-only network AMP. Default keeps FP32 inference unchanged;
        # generation scripts may set this to "fp16" or "bf16" to autocast only
        # the TorchMD network forward and cast the velocity back to fp32.
        self.network_amp = "none"

        if parity_switch is not None:
            assert (
                parity_switch == "post_hoc"
            ), f"Parity switch {parity_switch} not implemented"

        assert (
            self.prior_type in self.__prior_types__
        ), f"""\nPrior type {prior_type} not available.
            This is the list of implemented prior types {self.__prior_types__}.\n"""

        if prior_type == "harmonic":
            self.harmonic_sampler = HarmonicSampler(alpha=harmonic_alpha)

    @property
    def device(self) -> torch.device:
        """Device of the model parameters, including after ``to``/``cpu``/``cuda``."""
        return next(self.parameters()).device

    @classmethod
    def from_config(cls, cfg: Config):
        """Construct from a release config dictionary or a YAML filesystem path."""
        import yaml

        if isinstance(cfg, (str, PathLike)):
            with open(cfg) as handle:
                cfg = yaml.safe_load(handle)
        if not isinstance(cfg, dict):
            raise ValueError("cfg should be a dictionary or a path to a yaml file")
        return cls(**cfg["model_args"])

    def switch_parity_of_pos(
        self, pos, chiral_index, chiral_nbr_index, chiral_tag, batch
    ):
        assert all(
            [
                key is not None
                for key in [chiral_index, chiral_nbr_index, chiral_tag, batch]
            ]
        )
        num_graphs = batch.max().item() + 1
        sv = signed_volume(
            pos[chiral_nbr_index.view(chiral_index.shape[1], 4)].unsqueeze(2)
        ).squeeze()
        ct = chiral_tag
        z_flip = sv * ct

        graph_diag = torch.ones(num_graphs, device=self.device)
        graph_diag[batch[chiral_index][:, (z_flip == -1.0)].squeeze()] = -1.0
        node_factor = graph_diag[batch].unsqueeze(1)

        return pos * node_factor

    def sample_base_dist(
        self,
        size: torch.Size,
        edge_index: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        smiles: Optional[str] = None,
    ) -> Tensor:
        """Sample from prior distribution (Either harmonic or gaussian)"""
        if self.prior_type == "harmonic":
            assert (edge_index is not None) and (batch is not None)
            x0 = self.harmonic_sampler.sample(
                size=size, edge_index=edge_index, batch=batch, smiles=smiles
            ).to(self.device)

            # check if x0 is nan
            if torch.isnan(x0).any():
                raise ValueError("x0 is NaN. Check edge_index for disconnected graphs!")

            return x0

        # gaussian prior if not harmonic
        return torch.randn(size=size, device=self.device)

    def forward(
        self,
        z: Tensor,
        t: Tensor,
        pos: Tensor,
        bond_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        node_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        topology: Optional[tuple[Tensor, Tensor]] = None,
        geodesic_distance: Optional[Tensor] = None,
    ):
        # center the positions at 0
        pos = center_of_mass(pos, batch=batch)

        # compute extended bond index
        edge_index, edge_type = extend_bond_index(
            pos=pos,
            bond_index=bond_index,
            batch=batch,
            bond_attr=edge_attr,
            device=self.device,
            one_hot=self.edge_one_hot,
            one_hot_types=self.edge_one_hot_types,
            cutoff=self.cutoff,
            max_num_neighbors=self.max_num_neighbors,
            topology=topology,
        )

        # compute energy and score from network
        network_kwargs = dict(
            z=z,
            t=t[batch],
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_type,
            node_attr=node_attr,
            batch=batch,
            geodesic_distance=geodesic_distance,
        )
        network_amp = getattr(self, "network_amp", "none")
        if network_amp in (None, "", "none") or pos.device.type != "cuda":
            v_t = self.network(**network_kwargs)
        else:
            if network_amp == "fp16":
                amp_dtype = torch.float16
            elif network_amp == "bf16":
                amp_dtype = torch.bfloat16
            else:
                raise ValueError(f"Unknown network_amp={network_amp!r}; expected none/fp16/bf16")
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                v_t = self.network(**network_kwargs)
            # Keep the ODE integrator/state in fp32; only the network forward is autocast.
            v_t = v_t.float()

        return v_t

    def _compute_delta_t(self, t_schedule: Tensor, t: Tensor):
        if t + 1 >= t_schedule.size(0):
            return 0.0

        t_curr, t_next = t_schedule[t : t + 2]
        return t_next - t_curr

    @torch.no_grad()
    def sample(
        self,
        z: Tensor,
        bond_index: Tensor,
        batch: Tensor,
        node_attr: Tensor = None,
        edge_attr: Tensor = None,
        chiral_index: Tensor = None,
        chiral_nbr_index: Tensor = None,
        chiral_tag: Tensor = None,
        n_timesteps: int = 50,
        s_churn: float = 1.0,
        t_min: float = 1.0,
        t_max: float = 1.0,
        std: float = 1.0,
        sampler_type: str = "ode",
        smiles: Optional[List[str]] = None,
    ):
        """
        By default performs ODE (sampler_type="ode") sampling
        If sampler_type is set to "stochastic", then it performs stochastic sampling
        """
        t_schedule = torch.linspace(0, 1.0, steps=n_timesteps + 1, device=self.device)

        smiles_for_prior = smiles
        if isinstance(smiles_for_prior, str):
            num_graphs = int(batch.max().item()) + 1 if batch is not None else 1
            smiles_for_prior = [smiles_for_prior] * num_graphs

        x = center_of_mass(
            self.sample_base_dist(
                (z.size(0), 3),
                bond_index,
                batch,
                smiles=smiles_for_prior,
            ),
            batch=batch,
        )
        gamma = torch.tensor(s_churn / n_timesteps).to(self.device)

        # Covalent topology is constant during integration; radius edges are not.
        # Keep this state local so another molecule/device cannot reuse stale data.
        topology = flow_utils._topological_hop_edges(
            bond_index, batch, z.size(0), self.device,
        )
        representation = self.network.representation_model
        geodesic_distance = None
        if representation.global_geodesic_bias:
            geodesic_distance = model_dynamics.all_pair_graph_geodesic(
                topology[0], topology[1], batch, z.size(0),
                representation.global_geodesic_max_distance,
            )

        n = t_schedule.size(0) - 1
        for i in range(n):
            t = t_schedule[i].repeat(x.size(0))
            t = unsqueeze_like(t, x)
            delta_t = self._compute_delta_t(t_schedule, t=i)

            # We do ODE when t is outside of [s_min, s_max]
            if sampler_type == "ode" or (
                t_schedule[i] < t_min or t_schedule[i] >= t_max
            ):
                v_t = self(
                    z=z,
                    t=t,
                    pos=x,
                    bond_index=bond_index,
                    edge_attr=edge_attr,
                    node_attr=node_attr,
                    batch=batch,
                    topology=topology,
                    geodesic_distance=geodesic_distance,
                )
                x = x + delta_t * v_t

            # Stochastic sampling
            else:
                # delta_hat = gamma*delta_t
                delta_hat = gamma * (1 - t_schedule[i])
                t_prev_int = t_schedule[i] - delta_hat
                t_prev = t_prev_int.repeat(x.size(0))
                t_prev = unsqueeze_like(t_prev, x)
                """linear noise"""
                sig_t_sq = t_schedule[i] ** 2
                sig_t_prev_sq = t_prev_int**2
                mean = torch.zeros_like(x)
                noise = torch.normal(mean=mean, std=std)
                noise = center_of_mass(noise, batch=batch)
                x_prev = (
                    x
                    + torch.sqrt(torch.abs(sig_t_sq - sig_t_prev_sq))
                    * noise
                    * delta_hat
                )  # quadratic + linear decay

                v_t_prev = self(
                    z=z,
                    t=t_prev,
                    pos=x_prev,
                    bond_index=bond_index,
                    edge_attr=edge_attr,
                    node_attr=node_attr,
                    batch=batch,
                    topology=topology,
                    geodesic_distance=geodesic_distance,
                )
                # update step
                x = x_prev + v_t_prev * (delta_t + delta_hat)

        if self.parity_switch == "post_hoc":
            x = self.switch_parity_of_pos(
                x, chiral_index, chiral_nbr_index, chiral_tag, batch
            )

        return x
