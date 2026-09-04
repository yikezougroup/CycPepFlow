"""Load molecule-level records produced by ``scripts/convert_cremp.py``.

The release only needs deterministic inference access, so this module deliberately
avoids training data modules, random conformer sampling, and split-management
logic that are outside the published checkpoint-reproduction workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import datamol as dm
import torch
from torch import Tensor
from torch_geometric.data import Data

from cycpepflow.commons.featurization import MoleculeFeaturizer
from cycpepflow.commons.covmat import build_conformer


def _get_field(record: Any, name: str) -> Any:
    """Read a named field from either a plain mapping or a legacy PyG record."""
    if isinstance(record, dict):
        if name not in record:
            raise KeyError(name)
        return record[name]
    if not hasattr(record, name):
        raise AttributeError(name)
    return getattr(record, name)


class ProcessedConformerDataset:
    """Deterministic inference view over converted molecule-level ``.pt`` files.

    Current converter output contains only tensors and primitive containers and is
    loaded with PyTorch's restricted ``weights_only`` unpickler. Legacy v0.1.0 PyG
    records require explicit unsafe-pickle opt-in and must come from a trusted local
    conversion; never enable that option for downloaded or otherwise untrusted files.

    Parameters
    ----------
    data_dir:
        Root containing ``<partition>/<split>/*.pt``.
    partition:
        Converted dataset directory name.
    split:
        Dataset split, normally ``test`` for the released benchmark.
    include_chirality_in_node_attr:
        Whether atom chirality tags are included in the two-column node feature.
        Separate stereocenter tensors are always generated.
    allow_unsafe_legacy_pickle:
        Permit fallback loading of trusted v0.1.0 PyG records with
        ``weights_only=False``. This can execute arbitrary code from a malicious file.
    """

    def __init__(
        self,
        data_dir: str | Path,
        partition: str,
        split: str = "test",
        include_chirality_in_node_attr: bool = True,
        allow_unsafe_legacy_pickle: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser()
        self.partition = str(partition)
        self.split = str(split)
        self.include_chirality_in_node_attr = bool(include_chirality_in_node_attr)
        self.allow_unsafe_legacy_pickle = bool(allow_unsafe_legacy_pickle)
        split_dir = self.data_dir / self.partition / self.split
        self.data_files = sorted(split_dir.glob("*.pt"))
        if not self.data_files:
            raise FileNotFoundError(
                f"No processed records found in {split_dir}; expected one .pt file per molecule"
            )
        self.featurizer = MoleculeFeaturizer(
            include_chirality_in_node_attr=self.include_chirality_in_node_attr
        )

    def __len__(self) -> int:
        return len(self.data_files)

    def load_record(self, index: int) -> Any:
        """Load a safe current record, or an explicitly trusted legacy record."""
        path = self.data_files[index]
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            if not self.allow_unsafe_legacy_pickle:
                raise ValueError(
                    f"{path} is not a restricted-load-compatible CycPepFlow record. "
                    "Re-run scripts/convert_cremp.py, or enable unsafe legacy loading "
                    "only for a v0.1.0 record that you created and trust."
                ) from exc
            return torch.load(path, map_location="cpu", weights_only=False)

    def _normalize_record(self, record: Any, index: int) -> tuple[str, Tensor, Tensor]:
        smiles = str(_get_field(record, "smiles"))
        positions = torch.as_tensor(_get_field(record, "pos"), dtype=torch.float32)
        if positions.ndim == 2:
            positions = positions.unsqueeze(0)
        if positions.ndim != 3 or positions.shape[-1] != 3 or positions.shape[0] == 0:
            raise ValueError(
                f"{self.data_files[index]} has invalid pos shape {tuple(positions.shape)}; "
                "expected [n_conformers, n_atoms, 3]"
            )

        atomic_numbers = torch.as_tensor(
            _get_field(record, "atomic_numbers"), dtype=torch.long
        ).reshape(-1)
        if positions.shape[1] != atomic_numbers.numel():
            raise ValueError(
                f"{self.data_files[index]} has {positions.shape[1]} coordinate rows but "
                f"{atomic_numbers.numel()} atomic numbers"
            )
        return smiles, positions, atomic_numbers

    def get_with_references(self, index: int) -> tuple[Data, Tensor]:
        """Return one model-input graph and every normalized reference conformer."""
        record = self.load_record(index)
        smiles, positions, atomic_numbers = self._normalize_record(record, index)

        mol = dm.to_mol(smiles, remove_hs=False, ordered=True)
        graph = self.featurizer.get_data_from_mol(mol, smiles)
        if not torch.equal(graph.atomic_numbers.to(torch.long), atomic_numbers):
            raise ValueError(
                f"{self.data_files[index]} atomic numbers do not match reconstructed SMILES order"
            )

        mol.AddConformer(build_conformer(positions[0]))
        graph.pos = positions[0]
        graph.atomic_numbers = atomic_numbers
        graph.mol = mol
        return graph, positions

    def __getitem__(self, index: int) -> Data:
        return self.get_with_references(index)[0]
