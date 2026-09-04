from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

from cycpepflow.commons.featurization import MoleculeFeaturizer
from cycpepflow.data import ProcessedConformerDataset


def _safe_record(smiles: str, conformer_count: int = 2) -> dict:
    graph = MoleculeFeaturizer().get_data_from_smiles(smiles)
    atom_count = int(graph.atomic_numbers.numel())
    return {
        "smiles": graph.smiles,
        "atomic_numbers": graph.atomic_numbers,
        "pos": torch.zeros((conformer_count, atom_count, 3), dtype=torch.float32),
    }


def _write_record(root: str, record: object, name: str = "000000_record.pt") -> Path:
    split_dir = Path(root) / "tiny" / "test"
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / name
    torch.save(record, path)
    return path


class ProcessedConformerDatasetTest(unittest.TestCase):
    def test_safe_record_loads_and_chiral_indices_batch_correctly(self) -> None:
        record = _safe_record("N[C@@H](C)C(=O)O")
        with tempfile.TemporaryDirectory() as temporary_dir:
            _write_record(temporary_dir, record)
            dataset = ProcessedConformerDataset(temporary_dir, "tiny")
            sample, references = dataset.get_with_references(0)
            batch = Batch.from_data_list([sample, sample])

            atom_count = int(sample.atomic_numbers.numel())
            edge_count = int(sample.edge_index.shape[1])
            center_count = int(sample.chiral_index.numel())
            neighbor_count = int(sample.chiral_nbr_index.numel())

            self.assertGreater(center_count, 0)
            self.assertEqual(tuple(references.shape), (2, atom_count, 3))
            self.assertTrue(torch.equal(batch.atomic_numbers[:atom_count], sample.atomic_numbers))
            self.assertTrue(
                torch.equal(batch.edge_index[:, edge_count:], sample.edge_index + atom_count)
            )
            self.assertTrue(
                torch.equal(
                    batch.chiral_index[:, center_count:], sample.chiral_index + atom_count
                )
            )
            self.assertTrue(
                torch.equal(
                    batch.chiral_nbr_index[:, neighbor_count:],
                    sample.chiral_nbr_index + atom_count,
                )
            )

    def test_chirality_ablation_masks_only_node_attribute(self) -> None:
        record = _safe_record("N[C@@H](C)C(=O)O")
        with tempfile.TemporaryDirectory() as temporary_dir:
            _write_record(temporary_dir, record)
            enabled = ProcessedConformerDataset(temporary_dir, "tiny")[0]
            disabled = ProcessedConformerDataset(
                temporary_dir,
                "tiny",
                include_chirality_in_node_attr=False,
            )[0]

            self.assertFalse(torch.equal(enabled.node_attr[:, 0], disabled.node_attr[:, 0]))
            self.assertTrue(torch.equal(disabled.node_attr[:, 0], torch.zeros_like(disabled.node_attr[:, 0])))
            self.assertTrue(torch.equal(enabled.chiral_index, disabled.chiral_index))
            self.assertTrue(torch.equal(enabled.chiral_tag, disabled.chiral_tag))

    def test_rejects_malformed_coordinates(self) -> None:
        record = _safe_record("C")
        record["pos"] = record["pos"][..., :2]
        with tempfile.TemporaryDirectory() as temporary_dir:
            _write_record(temporary_dir, record)
            dataset = ProcessedConformerDataset(temporary_dir, "tiny")
            with self.assertRaisesRegex(ValueError, "invalid pos shape"):
                dataset[0]

    def test_legacy_pickle_requires_explicit_trust_opt_in(self) -> None:
        record = _safe_record("C")
        legacy = Data(**record)
        with tempfile.TemporaryDirectory() as temporary_dir:
            _write_record(temporary_dir, legacy)
            safe_dataset = ProcessedConformerDataset(temporary_dir, "tiny")
            with self.assertRaisesRegex(ValueError, "unsafe legacy loading"):
                safe_dataset.load_record(0)

            trusted_dataset = ProcessedConformerDataset(
                temporary_dir,
                "tiny",
                allow_unsafe_legacy_pickle=True,
            )
            self.assertIsInstance(trusted_dataset.load_record(0), Data)

    def test_repeated_smiles_keep_each_records_reference_conformer(self) -> None:
        first = _safe_record("N[C@@H](C)C(=O)O", conformer_count=1)
        second = dict(first, pos=first["pos"] + 2.0)
        with tempfile.TemporaryDirectory() as temporary_dir:
            _write_record(temporary_dir, first, "000000_first.pt")
            _write_record(temporary_dir, second, "000001_second.pt")
            dataset = ProcessedConformerDataset(temporary_dir, "tiny")
            dataset[0]
            sample, references = dataset.get_with_references(1)
            actual = torch.as_tensor(sample.mol.GetConformer().GetPositions()).float()
            self.assertTrue(torch.equal(actual, references[0]))

    def test_record_parses_smiles_once(self) -> None:
        from unittest.mock import patch
        import datamol as dm

        record = _safe_record("N[C@@H](C)C(=O)O")
        with tempfile.TemporaryDirectory() as temporary_dir:
            _write_record(temporary_dir, record)
            dataset = ProcessedConformerDataset(temporary_dir, "tiny")
            with patch.object(dm, "to_mol", wraps=dm.to_mol) as parse:
                dataset[0]
            self.assertEqual(parse.call_count, 1)

    def test_rejects_missing_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            with self.assertRaises(FileNotFoundError):
                ProcessedConformerDataset(temporary_dir, "missing")


if __name__ == "__main__":
    unittest.main()
