"""Atom-order-preserving molecular graphs for checkpoint inference."""
import datamol as dm
import torch
from rdkit import Chem
from torch_geometric.data import Data

from .utils import atom_to_feature_vector, compute_edge_index, get_chiral_tensors


class MoleculeFeaturizer:
    """Build charge/chirality features and covalent edges from one parsed molecule."""

    def __init__(self, include_chirality_in_node_attr: bool = True):
        self.include_chirality_in_node_attr = bool(include_chirality_in_node_attr)

    def get_data_from_mol(self, mol: Chem.Mol, smiles: str) -> Data:
        atomic_numbers = torch.tensor(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.int32,
        )
        node_attr = torch.tensor(
            [atom_to_feature_vector(atom, include_chirality=self.include_chirality_in_node_attr)
             for atom in mol.GetAtoms()], dtype=torch.float32,
        )
        chiral_index, chiral_nbr_index, chiral_tag = get_chiral_tensors(mol)
        edge_index, edge_attr = compute_edge_index(mol, with_edge_attr=False)
        return Data(
            atomic_numbers=atomic_numbers, smiles=smiles, edge_index=edge_index,
            chiral_index=chiral_index, chiral_nbr_index=chiral_nbr_index,
            chiral_tag=chiral_tag, node_attr=node_attr, edge_attr=edge_attr,
            num_nodes=mol.GetNumAtoms(),
        )

    def get_data_from_smiles(self, smiles: str) -> Data:
        """Add explicit hydrogens and retain atom order in the mapped SMILES."""
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        mapped_smiles = dm.to_smiles(
            mol, canonical=False, explicit_hs=True, with_atom_indices=True, isomeric=True,
        )
        return self.get_data_from_mol(mol, mapped_smiles)
