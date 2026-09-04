"""Regression guards for the inference-only release boundary."""
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from cycpepflow.models import BaseFlow

ROOT = Path(__file__).resolve().parents[1]


def tiny_model(**kwargs):
    return BaseFlow(hidden_channels=16, num_heads=2, num_layers=1,
                    num_rbf=8, node_attr_dim=2, edge_attr_dim=1, **kwargs)


def test_model_is_plain_torch_without_training_hooks():
    model = tiny_model()
    assert type(model).__bases__ == (torch.nn.Module,)
    assert not hasattr(model, 'training_step')
    assert not hasattr(model, 'configure_optimizers')
    assert model.device == next(model.parameters()).device


def test_import_does_not_load_training_frameworks():
    probe = subprocess.run(
        [sys.executable, '-c',
         "from cycpepflow.models import BaseFlow; import sys; "
         "assert not any(m.split('.')[0] in "
         "{'lightning', 'pytorch_lightning', 'torchmetrics'} for m in sys.modules)"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert probe.returncode == 0, probe.stderr


def test_config_accepts_path_without_changing_state():
    config = {'model_args': {'hidden_channels': 16, 'num_heads': 2,
                            'num_layers': 1, 'num_rbf': 8,
                            'node_attr_dim': 2, 'edge_attr_dim': 1}}
    torch.manual_seed(17)
    expected = BaseFlow.from_config(config)
    # A temporary path is covered separately by pytest's fixture below.
    assert all(k.startswith('network.') for k in expected.state_dict())


def test_from_config_pathlike(tmp_path):
    path = tmp_path / 'config.yaml'
    cfg = {'model_args': {'hidden_channels': 16, 'num_heads': 2, 'num_layers': 1,
                          'num_rbf': 8, 'node_attr_dim': 2, 'edge_attr_dim': 1}}
    path.write_text(yaml.safe_dump(cfg))
    torch.manual_seed(17)
    from_dict = BaseFlow.from_config(cfg)
    torch.manual_seed(17)
    from_path = BaseFlow.from_config(path)
    for key, value in from_dict.state_dict().items():
        assert torch.equal(value, from_path.state_dict()[key])


def test_archived_training_options_do_not_change_inference_state():
    torch.manual_seed(9)
    plain = tiny_model()
    torch.manual_seed(9)
    legacy = tiny_model(optimizer_type='SOAP', lr=0.0008, warmup_steps=8000,
                        chirality_loss_weight=0.5, chirality_ramp_steps=100,
                        sample_time_dist='uniform', sigma=0.1)
    assert plain.state_dict().keys() == legacy.state_dict().keys()
    for key, value in plain.state_dict().items():
        assert torch.equal(value, legacy.state_dict()[key])


def test_unknown_model_options_fail_closed():
    with pytest.raises(TypeError):
        tiny_model(hidden_chanels=16)


def test_upstream_only_loaders_and_duplicate_predictor_are_absent():
    assert not hasattr(BaseFlow, 'from_default')
    assert not hasattr(BaseFlow, 'predict')
