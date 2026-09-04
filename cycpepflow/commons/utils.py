# allowable multiple choice node and edge features
import numpy as np
import torch
from rdkit.Chem.rdchem import ChiralType
from torch_cluster import radius_graph

# similar to GeoMol
chirality = {
    ChiralType.CHI_TETRAHEDRAL_CW: -1.0,
    ChiralType.CHI_TETRAHEDRAL_CCW: 1.0,
    ChiralType.CHI_UNSPECIFIED: 0,
    ChiralType.CHI_OTHER: 0,
}

allowable_features = {
    "possible_chirality_list": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
        "misc",
    ],
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "possible_bond_type_list": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
}


def atom_to_feature_vector(atom, include_chirality: bool = True):
    """Formal-charge + chirality-tag-only node invariant features.

    This ablation keeps only the two baseline OGB-style categorical atom
    descriptors requested by Dizhou:
      0. RDKit chiral tag index, optionally masked to CHI_UNSPECIFIED
      1. RDKit formal charge index over possible_formal_charge_list

    Atomic number is still supplied separately through ``atomic_numbers`` and
    embedded by the model exactly as in the baseline. Edge features are not
    changed: the baseline one-scalar RDKit bond type is preserved.
    """
    chiral_label = str(atom.GetChiralTag()) if include_chirality else "CHI_UNSPECIFIED"
    return [
        safe_index(allowable_features["possible_chirality_list"], chiral_label),
        safe_index(allowable_features["possible_formal_charge_list"], atom.GetFormalCharge()),
    ]

def safe_index(values, item):
    """
    Return the category index, or the last (miscellaneous) category if absent.
    """
    try:
        return values.index(item)
    except Exception:
        return len(values) - 1


def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
        safe_index(
            allowable_features["possible_bond_type_list"], str(bond.GetBondType())
        ),
    ]
    return bond_feature


def compute_edge_index(
    mol, no_reverse: bool = False, with_edge_attr=False
) -> torch.Tensor:
    """Computes edge index from mol object"""
    edge_list = []
    bond_types = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_list.append((i, j))
        bond_types.append(bond_to_feature_vector(bond))

        if not no_reverse:
            edge_list.append((j, i))
            bond_types.append(bond_to_feature_vector(bond))

    if len(edge_list) == 0:
        return torch.empty((2, 0)).long()

    edge_index = torch.from_numpy(np.array(edge_list).T).long()

    if with_edge_attr:
        edge_attr = torch.tensor(bond_types, dtype=torch.float32)  # (num_edges, 1)
        return edge_index, edge_attr

    return edge_index, None


def _extend_to_radius_graph(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    batch: torch.Tensor,
    cutoff: float = 10.0,
    max_num_neighbors: int = 32,
    unspecified_type_number=0,
):
    assert edge_type.dim() == 1
    N = pos.size(0)

    bgraph_adj = torch.sparse_coo_tensor(edge_index, edge_type, torch.Size([N, N]))
    rgraph_edge_index = radius_graph(
        pos, r=cutoff, batch=batch, max_num_neighbors=max_num_neighbors
    )  # (2, E_r)

    rgraph_adj = torch.sparse_coo_tensor(
        rgraph_edge_index,
        torch.ones(rgraph_edge_index.size(1)).long().to(pos.device)
        * unspecified_type_number,
        torch.Size([N, N]),
    )

    composed_adj = (bgraph_adj + rgraph_adj).coalesce()  # Sparse (N, N, T)

    new_edge_index = composed_adj.indices()
    new_edge_type = composed_adj.values().long()

    return new_edge_index, new_edge_type


def extend_graph_order_radius(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    batch: torch.Tensor,
    cutoff: float = 10.0,
    max_num_neighbors: int = 32,
    extend_radius: bool = True,
):
    """Extends bond index"""
    if extend_radius:
        edge_index, edge_type = _extend_to_radius_graph(
            pos=pos,
            edge_index=edge_index,
            edge_type=edge_type,
            cutoff=cutoff,
            batch=batch,
            max_num_neighbors=max_num_neighbors,
        )

    return edge_index, edge_type


def signed_volume(local_coords):
    """
    Compute signed volume given ordered neighbor local coordinates
    From GeoMol

    :param local_coords: (n_tetrahedral_chiral_centers, 4, n_generated_confs, 3)
    :return: signed volume of each tetrahedral center
    (n_tetrahedral_chiral_centers, n_generated_confs)
    """
    v1 = local_coords[:, 0] - local_coords[:, 3]
    v2 = local_coords[:, 1] - local_coords[:, 3]
    v3 = local_coords[:, 2] - local_coords[:, 3]
    cp = v2.cross(v3, dim=-1)
    vol = torch.sum(v1 * cp, dim=-1)
    return torch.sign(vol)


def get_chiral_tensors(mol):
    """Only consider chiral atoms with 4 neighbors"""
    chiral_index = torch.tensor(
        [
            i
            for i, atom in enumerate(mol.GetAtoms())
            if (chirality[atom.GetChiralTag()] != 0 and len(atom.GetNeighbors()) == 4)
        ],
        dtype=torch.int32,
    ).view(
        1, -1
    )  # (1, n_chiral_centers)
    # (n_chiral_centers, 4)
    chiral_nbr_index = torch.tensor(
        [
            [n.GetIdx() for n in atom.GetNeighbors()]
            for atom in mol.GetAtoms()
            if (chirality[atom.GetChiralTag()] != 0 and len(atom.GetNeighbors()) == 4)
        ],
        dtype=torch.int32,
    ).view(
        1, -1
    )  # (1, n_chiral_centers * 4)
    # (n_chiral_centers,)
    chiral_tag = torch.tensor(
        [
            chirality[atom.GetChiralTag()]
            for atom in mol.GetAtoms()
            if (chirality[atom.GetChiralTag()] != 0 and len(atom.GetNeighbors()) == 4)
        ],
        dtype=torch.float32,
    )

    return chiral_index, chiral_nbr_index, chiral_tag
