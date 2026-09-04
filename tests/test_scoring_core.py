"""Guard the scoring functions used by the release, not the retired evaluator."""
import numpy as np

from cycpepflow.commons import covmat
from cycpepflow.networks.torchmd_net import modules, utils


def test_only_used_scoring_and_network_surfaces_remain():
    assert not hasattr(covmat, 'CovMatEvaluator')
    assert not hasattr(covmat, 'get_rmsd')
    assert not hasattr(covmat, 'set_multiple_rdmol_positions')
    assert not hasattr(modules, 'Scalar')
    assert not hasattr(modules, 'EquivariantVectorAndScalarOutput')
    assert not hasattr(utils, 'Distance')


def test_covmat_recall_precision_and_strict_threshold():
    distances = np.array([[0.1, 0.8], [0.6, 1.0]])
    recall, mat_r, precision, mat_p = covmat.calc_performance_stats(
        distances, np.array([0.6, 0.75]),
    )
    np.testing.assert_array_equal(recall, [0.5, 1.0])
    np.testing.assert_array_equal(precision, [0.5, 0.5])
    np.testing.assert_allclose(mat_r, (0.1 + 0.6) / 2)
    np.testing.assert_allclose(mat_p, (0.1 + 0.8) / 2)


def test_conformer_positions_roundtrip():
    from rdkit import Chem

    molecule = Chem.AddHs(Chem.MolFromSmiles('C'))
    positions = np.arange(molecule.GetNumAtoms() * 3, dtype=float).reshape(-1, 3)
    with_positions = covmat.set_rdmol_positions(molecule, positions)
    np.testing.assert_array_equal(with_positions.GetConformer().GetPositions(), positions)
    assert molecule.GetNumConformers() == 0
