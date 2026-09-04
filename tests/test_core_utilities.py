"""Release-boundary and serialization guards for the inference commons."""
import importlib.util
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_import_does_not_load_xtb_or_create_its_work_directory():
    probe = subprocess.run(
        [sys.executable, "-B", "-c", """
import os
import sys
from pathlib import Path
work_directory = Path('/tmp') / str(os.getpid())
existed = work_directory.exists()
import cycpepflow.commons
created = not existed and work_directory.exists()
if created:
    work_directory.rmdir()
assert 'cycpepflow.commons.xtb' not in sys.modules, 'commons eagerly imported xTB'
assert not created, 'commons created an unused xTB work directory'
"""],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert probe.returncode == 0, probe.stderr


def test_upstream_only_modules_are_absent():
    for module in ("configs", "sample", "xtb"):
        assert importlib.util.find_spec(f"cycpepflow.commons.{module}") is None


def test_retired_helpers_are_not_defined_or_reexported():
    from cycpepflow import commons
    from cycpepflow.commons import io, utils

    retired_io = (
        "load_memmap", "save_memmap", "load_npz", "load_json",
        "get_local_cache", "get_base_data_dir", "CACHE_DIR",
    )
    retired_utils = (
        "Queue", "get_atomic_number_and_charge", "GetNumRings",
        "get_neighbor_ids", "BOND_TYPES",
    )
    for module, names in ((io, retired_io), (utils, retired_utils)):
        for name in names:
            assert not hasattr(module, name), f"{module.__name__}.{name} remains"
            assert not hasattr(commons, name), f"commons still reexports {name}"
    for name in ("batched_sampling", "xtb_energy", "xtb_optimize"):
        assert not hasattr(commons, name), f"commons still reexports {name}"


def test_feature_catalog_retains_exact_released_encodings():
    from cycpepflow.commons.utils import allowable_features

    assert allowable_features == {
        "possible_chirality_list": [
            "CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW",
            "CHI_OTHER", "misc",
        ],
        "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
        "possible_bond_type_list": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
    }


def test_pickle_helpers_preserve_standard_pickle_format(tmp_path):
    from cycpepflow.commons import load_pkl, save_pkl

    payload = [{"smiles": "C", "values": [1, 2.5, None], "trusted_local": True}]
    path = tmp_path / "generated.pkl"
    save_pkl(str(path), payload)
    assert path.read_bytes() == pickle.dumps(payload)
    with path.open("rb") as handle:
        assert pickle.load(handle) == payload
    assert load_pkl(str(path)) == payload


def test_pickle_helpers_roundtrip_generated_pyg_records(tmp_path):
    import numpy as np
    from torch_geometric.data import Data
    from cycpepflow.commons import load_pkl, save_pkl

    positions = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    record = Data(smiles="CC", pos_ref=positions, pos_gen=positions.repeat(2, axis=0))
    path = tmp_path / "generated.pkl"
    save_pkl(str(path), [record])
    restored = load_pkl(str(path))[0]
    assert isinstance(restored, Data)
    assert restored.smiles == record.smiles
    np.testing.assert_array_equal(restored.pos_ref, record.pos_ref)
    np.testing.assert_array_equal(restored.pos_gen, record.pos_gen)


def test_pickle_loader_keeps_missing_file_error(tmp_path):
    from cycpepflow.commons import load_pkl

    path = tmp_path / "missing.pkl"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_pkl(str(path))
