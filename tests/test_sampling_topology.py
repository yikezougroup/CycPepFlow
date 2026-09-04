import pytest
import torch
from torch_geometric.data import Batch

from cycpepflow.commons.featurization import MoleculeFeaturizer
from cycpepflow.models import BaseFlow
from cycpepflow.models import utils
from cycpepflow.networks.torchmd_net import model_dynamics


def graph_batch():
    data = MoleculeFeaturizer().get_data_from_smiles('C1CC1')
    data.num_nodes = data.atomic_numbers.numel()
    return Batch.from_data_list([data, data])


def test_empty_bond_graph_retains_radius_edges():
    pos = torch.tensor([[0., 0., 0.], [1., 0., 0.]])
    edges, types = utils.extend_bond_index(
        pos, torch.empty((2, 0), dtype=torch.long), torch.zeros(2, dtype=torch.long),
        None, pos.device,
    )
    assert edges.shape == (2, 2)
    assert torch.equal(types, torch.zeros(2, dtype=torch.long))


def test_precomputed_topology_preserves_dynamic_edges():
    batch = graph_batch()
    pos = torch.randn(batch.num_nodes, 3)
    topology = utils._topological_hop_edges(batch.edge_index, batch.batch,
                                            batch.num_nodes, pos.device)
    for scale in (1., 20.):
        args = (pos * scale, batch.edge_index, batch.batch, None, pos.device)
        plain = utils.extend_bond_index(*args)
        cached = utils.extend_bond_index(*args, topology=topology)
        assert all(torch.equal(a, b) for a, b in zip(plain, cached))


@pytest.mark.parametrize('apg', [False, True])
def test_sample_builds_topology_once_and_matches_uncached_euler(monkeypatch, apg):
    torch.manual_seed(11)
    model = BaseFlow(hidden_channels=16, num_heads=2, num_layers=1, num_rbf=8,
                     node_attr_dim=2, edge_attr_dim=1, global_attention=True,
                     global_geodesic_bias=apg).eval()
    batch = graph_batch()
    inputs = dict(z=batch.atomic_numbers, bond_index=batch.edge_index,
                  batch=batch.batch, node_attr=batch.node_attr)
    counts = {'hops': 0, 'apg': 0}
    original_hops = utils._topological_hop_edges
    original_apg = model_dynamics.all_pair_graph_geodesic

    def hops(*args, **kwargs):
        counts['hops'] += 1
        return original_hops(*args, **kwargs)

    def geodesic(*args, **kwargs):
        counts['apg'] += 1
        return original_apg(*args, **kwargs)

    monkeypatch.setattr(utils, '_topological_hop_edges', hops)
    monkeypatch.setattr(model_dynamics, 'all_pair_graph_geodesic', geodesic)
    torch.manual_seed(42)
    actual = model.sample(**inputs, n_timesteps=4)
    assert counts == {'hops': 1, 'apg': int(apg)}
    torch.manual_seed(42)
    with torch.no_grad():
        expected = utils.center_of_mass(
            model.sample_base_dist((batch.num_nodes, 3), batch.edge_index, batch.batch),
            batch=batch.batch,
        )
        schedule = torch.linspace(0, 1, 5)
        for i in range(4):
            t = schedule[i].repeat(expected.size(0)).view(-1, 1)
            expected = expected + (schedule[i + 1] - schedule[i]) * model(
                **inputs, t=t, pos=expected,
            )
    assert torch.equal(actual, expected)
    # A new invocation must rebuild topology, never reuse a previous molecule's cache.
    before = counts.copy()
    model.sample(**inputs, n_timesteps=2)
    assert counts['hops'] - before['hops'] == 1
    assert counts['apg'] - before['apg'] == int(apg)
