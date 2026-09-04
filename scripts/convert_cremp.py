#!/usr/bin/env python3
"""Extract official RINGER/CREMP split molecules into CycPepFlow-style .pt files.

Input is either CREMP Zenodo record 7931445 `pickle.tar.gz` or an extracted directory
containing the official per-molecule `*.pickle` files.
For each target sequence in a manifest CSV, this keeps up to --max-confs conformers sorted by
per-conformer energy when an energy array is present in the CREMP pickle dict. Splits listed in
--all-confs-splits keep every conformer (after energy sorting if available), e.g. keep all test
conformers while capping train/val at 30.
"""
import argparse
import csv
import hashlib
import io
import json
import pickle
import tarfile
import traceback
from collections import defaultdict
from pathlib import Path

import torch
import datamol as dm
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

BOND_TYPE_TO_INT = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 12,
}


def read_manifest(path):
    rows = []
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def as_list_like(v):
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        try:
            return v.detach().cpu().reshape(-1).tolist()
        except Exception:
            return None
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return v.reshape(-1).tolist()
    except Exception:
        pass
    if isinstance(v, (list, tuple)):
        # Flatten simple nested singleton arrays/tensors.
        out = []
        for x in v:
            if isinstance(x, torch.Tensor):
                vals = x.detach().cpu().reshape(-1).tolist()
                out.extend(vals)
            elif isinstance(x, (list, tuple)) and len(x) == 1:
                out.append(x[0])
            else:
                out.append(x)
        return out
    return None


def find_mol(obj):
    if isinstance(obj, Chem.Mol):
        return obj
    if isinstance(obj, dict):
        for key in ('mol', 'rdmol', 'rdkit_mol', 'rd_mol', 'conformers', 'ensemble'):
            if key in obj and isinstance(obj[key], Chem.Mol):
                return obj[key]
        for v in obj.values():
            if isinstance(v, Chem.Mol):
                return v
    if hasattr(obj, 'rdmol') and isinstance(obj.rdmol, Chem.Mol):
        return obj.rdmol
    if hasattr(obj, 'mol') and isinstance(obj.mol, Chem.Mol):
        return obj.mol
    raise ValueError(f'Could not find RDKit Mol in pickle object of type {type(obj)}')


def find_source_smiles(obj, mol, manifest_smiles):
    if isinstance(obj, dict):
        for key in ('smiles', 'canonical_smiles', 'SMILES'):
            if key in obj and isinstance(obj[key], str):
                return obj[key]
    if hasattr(obj, 'smiles') and isinstance(obj.smiles, str):
        return obj.smiles
    if manifest_smiles:
        return manifest_smiles
    return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True, canonical=True)


def ordered_explicit_h_smiles(mol):
    """Return CycPepFlow-compatible explicit-H, atom-mapped SMILES preserving RDKit atom order.

    CycPepFlow reconstructs the graph from `data.smiles` with datamol.to_mol(..., remove_hs=False,
    ordered=True), then adds the saved coordinate tensor as a conformer. The CREMP pickles contain
    explicit-H coordinates, so a normal/heavy-atom SMILES causes RDKit atom-count mismatches. The
    atom-map numbers emitted by datamol make datamol's ordered loader recover the original atom order.
    """
    smiles = dm.to_smiles(
        mol,
        canonical=False,
        explicit_hs=True,
        with_atom_indices=True,
        isomeric=True,
    )
    check = dm.to_mol(smiles, remove_hs=False, ordered=True)
    if check is None:
        raise ValueError('datamol could not reparse explicit-H atom-mapped SMILES')
    if check.GetNumAtoms() != mol.GetNumAtoms():
        raise ValueError(
            f'explicit-H SMILES atom-count mismatch: mol={mol.GetNumAtoms()} parsed={check.GetNumAtoms()}'
        )
    return smiles


def dict_candidate_arrays(obj, n):
    if not isinstance(obj, dict):
        return []
    cands = []
    for key, val in obj.items():
        arr = as_list_like(val)
        if arr is not None and len(arr) == n:
            cands.append((key, arr))
    return cands


def conformer_metadata_arrays(obj, n):
    """Return per-conformer metadata arrays from CREMP's conformers list when present."""
    if not isinstance(obj, dict):
        return []
    confs = obj.get('conformers')
    if not (isinstance(confs, list) and len(confs) == n and all(isinstance(c, dict) for c in confs)):
        return []
    keys = sorted(set().union(*(c.keys() for c in confs))) if confs else []
    arrays = []
    for key in keys:
        vals = []
        ok = True
        for c in confs:
            if key not in c:
                ok = False
                break
            v = c[key]
            if isinstance(v, (int, float)):
                vals.append(v)
            else:
                ok = False
                break
        if ok:
            arrays.append((f'conformers.{key}', vals))
    return arrays


def choose_energy_array(obj, n):
    cands = dict_candidate_arrays(obj, n) + conformer_metadata_arrays(obj, n)
    energy_cands = [(k, a) for k, a in cands if 'energy' in k.lower()]
    if not energy_cands:
        return None, None, [k for k, _ in cands]
    # Prefer relativeenergy when present: it is already the xTB energy relative to the lowest conformer.
    # It is also numerically stable and exactly defines the desired low-energy ranking.
    priority = ['relativeenergy', 'relative_energy', 'totalenergy', 'total_energy', 'totalenergies', 'total_energies', 'xtb', 'energy']
    for token in priority:
        for k, a in energy_cands:
            if token in k.lower():
                return k, [float(x) for x in a], [k for k, _ in cands]
    k, a = energy_cands[0]
    return k, [float(x) for x in a], [k for k, _ in cands]


def choose_weight_array(obj, n):
    cands = dict_candidate_arrays(obj, n) + conformer_metadata_arrays(obj, n)
    for k, a in cands:
        lk = k.lower()
        if 'boltz' in lk or 'weight' in lk or 'population' in lk or 'pop' in lk:
            try:
                return k, [float(x) for x in a]
            except Exception:
                pass
    return None, None


def edge_tensors(mol):
    edges = []
    etypes = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx(); j = b.GetEndAtomIdx()
        t = BOND_TYPE_TO_INT.get(b.GetBondType(), 1)
        edges.append((i, j)); etypes.append(t)
        edges.append((j, i)); etypes.append(t)
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(etypes, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)
    return edge_index, edge_type


def convert_one(obj, row, max_confs, keep_all_confs=False):
    mol = find_mol(obj)
    nconf = mol.GetNumConformers()
    if nconf < 1:
        raise ValueError('RDKit Mol has no conformers')
    energy_key, energies, candidate_keys = choose_energy_array(obj, nconf)
    weight_key, weights = choose_weight_array(obj, nconf)
    conf_ids = [c.GetId() for c in mol.GetConformers()]
    n_keep = nconf if keep_all_confs else min(nconf, max_confs)
    if energies is not None:
        order = sorted(range(nconf), key=lambda i: (energies[i], i))[:n_keep]
    else:
        order = list(range(n_keep))
    selected_conf_ids = [conf_ids[i] for i in order]
    pos = []
    for cid in selected_conf_ids:
        conf = mol.GetConformer(cid)
        coords = []
        for atom_i in range(mol.GetNumAtoms()):
            p = conf.GetAtomPosition(atom_i)
            coords.append([p.x, p.y, p.z])
        pos.append(coords)
    pos = torch.tensor(pos, dtype=torch.float32)
    atomic_numbers = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
    charges = torch.tensor([a.GetFormalCharge() for a in mol.GetAtoms()], dtype=torch.long)
    edge_index, edge_type = edge_tensors(mol)
    selected_energies = [energies[i] for i in order] if energies is not None else [float('nan')] * len(order)
    selected_weights = [weights[i] for i in order] if weights is not None else [float('nan')] * len(order)
    smiles = ordered_explicit_h_smiles(mol)
    source_smiles = find_source_smiles(obj, mol, row.get('smiles', ''))
    # Keep the serialized output to tensors and primitive Python containers so
    # inference can load it with torch.load(..., weights_only=True).
    data = {
        'atomic_numbers': atomic_numbers,
        'atom_type': atomic_numbers.clone(),
        'charges': charges,
        'edge_index': edge_index,
        'edge_type': edge_type,
        'pos': pos,
        'energy': torch.tensor(selected_energies, dtype=torch.float32).reshape(-1, 1),
        'totalenergy': torch.tensor(selected_energies, dtype=torch.float32).reshape(-1, 1),
        'boltzmannweight': torch.tensor(selected_weights, dtype=torch.float32).reshape(-1, 1),
        'smiles': smiles,
        'source_smiles': source_smiles,
        'sequence': row.get('sequence', ''),
        'subset': 'ringer_cremp',
        'split': row.get('split', ''),
        'summary_row_1based': int(row.get('summary_row_1based', -1)),
        'num_original_conformers': int(nconf),
        'selected_conformer_indices': torch.tensor(order, dtype=torch.long),
    }
    meta = {
        'sequence': row.get('sequence', ''),
        'split': row.get('split', ''),
        'summary_row_1based': int(row.get('summary_row_1based', -1)),
        'n_atoms': int(mol.GetNumAtoms()),
        'n_h_atoms': int(sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 1)),
        'cycpepflow_smiles_policy': 'datamol explicit_hs=True with_atom_indices=True canonical=False; preserves explicit-H atom order for CycPepFlow',
        'source_smiles': source_smiles,
        'cycpepflow_smiles_len': int(len(smiles)),
        'n_original_conformers_in_pickle_mol': int(nconf),
        'n_saved_conformers': int(pos.shape[0]),
        'keep_all_conformers': bool(keep_all_confs),
        'max_confs_cap': None if keep_all_confs else int(max_confs),
        'energy_key': energy_key,
        'weight_key': weight_key,
        'candidate_array_keys_len_nconf': candidate_keys,
        'selected_energy_min': None if energies is None else float(min(selected_energies)),
        'selected_energy_max': None if energies is None else float(max(selected_energies)),
        'used_manifest_uniqueconfs': row.get('uniqueconfs', ''),
        'used_manifest_totalconfs': row.get('totalconfs', ''),
    }
    return data, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', type=Path, help='CREMP Zenodo pickle.tar.gz')
    ap.add_argument('--pickle-dir', type=Path, help='Extracted directory containing CREMP *.pickle files')
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--out-root', required=True, type=Path)
    ap.add_argument('--max-confs', type=int, default=30)
    ap.add_argument(
        '--all-confs-splits',
        default='',
        help='Comma-separated split names for which every conformer is kept, e.g. test',
    )
    ap.add_argument('--summary-json', required=True, type=Path)
    ap.add_argument('--failures-csv', required=True, type=Path)
    args = ap.parse_args()
    if (args.archive is None) == (args.pickle_dir is None):
        raise SystemExit('Provide exactly one of --archive or --pickle-dir')
    all_confs_splits = {x.strip() for x in args.all_confs_splits.split(',') if x.strip()}

    rows = read_manifest(args.manifest)
    wanted = {r['archive_pickle_basename']: r for r in rows}
    # Also accept names without extension just in case.
    wanted_stems = {Path(k).stem: k for k in wanted}
    out_dirs = {split: args.out_root / split for split in ('train', 'val', 'test')}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    meta_rows = []
    failures = []
    seen = set()

    if args.pickle_dir is not None:
        for key, row in wanted.items():
            p = args.pickle_dir / key
            if not p.is_file():
                failures.append({
                    'sequence': row.get('sequence',''),
                    'split': row.get('split',''),
                    'archive_basename': key,
                    'archive_member': str(p),
                    'error': 'pickle file not found in --pickle-dir',
                    'traceback': '',
                })
                continue
            try:
                obj = pickle.loads(p.read_bytes())
                keep_all = row.get('split', '') in all_confs_splits
                data, meta = convert_one(obj, row, args.max_confs, keep_all_confs=keep_all)
                split = row['split']
                safe = hashlib.md5(row['sequence'].encode()).hexdigest()
                out_path = out_dirs[split] / f"{int(row['split_index']):06d}_{safe}.pt"
                torch.save(data, out_path)
                meta['archive_member'] = str(p)
                meta['output_path'] = str(out_path)
                meta_rows.append(meta)
                seen.add(key)
                if len(seen) % 100 == 0:
                    print(f'converted {len(seen)}/{len(wanted)}', flush=True)
            except Exception as e:
                failures.append({
                    'sequence': row.get('sequence',''),
                    'split': row.get('split',''),
                    'archive_basename': key,
                    'archive_member': str(p),
                    'error': repr(e),
                    'traceback': traceback.format_exc(limit=8),
                })
    else:
      with tarfile.open(args.archive, mode='r:gz') as tar:
        for member in tar:
            if not member.isfile():
                continue
            base = Path(member.name).name
            stem = Path(base).stem
            key = None
            if base in wanted:
                key = base
            elif stem in wanted_stems:
                key = wanted_stems[stem]
            if key is None:
                continue
            row = wanted[key]
            try:
                fh = tar.extractfile(member)
                if fh is None:
                    raise ValueError('tar.extractfile returned None')
                payload = fh.read()
                obj = pickle.load(io.BytesIO(payload))
                keep_all = row.get('split', '') in all_confs_splits
                data, meta = convert_one(obj, row, args.max_confs, keep_all_confs=keep_all)
                split = row['split']
                safe = hashlib.md5(row['sequence'].encode()).hexdigest()
                out_path = out_dirs[split] / f"{int(row['split_index']):06d}_{safe}.pt"
                torch.save(data, out_path)
                meta['archive_member'] = member.name
                meta['output_path'] = str(out_path)
                meta_rows.append(meta)
                seen.add(key)
                if len(seen) % 100 == 0:
                    print(f'converted {len(seen)}/{len(wanted)}', flush=True)
            except Exception as e:
                failures.append({
                    'sequence': row.get('sequence',''),
                    'split': row.get('split',''),
                    'archive_basename': key,
                    'archive_member': member.name,
                    'error': repr(e),
                    'traceback': traceback.format_exc(limit=8),
                })

    missing = sorted(set(wanted) - seen - {f['archive_basename'] for f in failures})
    for key in missing:
        row = wanted[key]
        failures.append({
            'sequence': row.get('sequence',''),
            'split': row.get('split',''),
            'archive_basename': key,
            'archive_member': '',
            'error': 'archive member not found',
            'traceback': '',
        })

    split_counts = defaultdict(int)
    split_confs = defaultdict(int)
    for m in meta_rows:
        split_counts[m['split']] += 1
        split_confs[m['split']] += m['n_saved_conformers']

    summary = {
        'archive': None if args.archive is None else str(args.archive),
        'pickle_dir': None if args.pickle_dir is None else str(args.pickle_dir),
        'manifest': str(args.manifest),
        'out_root': str(args.out_root),
        'max_confs': args.max_confs,
        'all_confs_splits': sorted(all_confs_splits),
        'policy': 'train/val capped by --max-confs; splits in --all-confs-splits keep every original conformer',
        'requested_molecules': len(rows),
        'converted_molecules': len(meta_rows),
        'failed_or_missing_molecules': len(failures),
        'split_counts': dict(sorted(split_counts.items())),
        'saved_conformers_by_split': dict(sorted(split_confs.items())),
        'saved_conformers_total': int(sum(split_confs.values())),
        'energy_keys_observed': sorted(set(m['energy_key'] for m in meta_rows if m.get('energy_key'))),
        'weight_keys_observed': sorted(set(m['weight_key'] for m in meta_rows if m.get('weight_key'))),
        'failures_csv': str(args.failures_csv),
        'meta_first5': meta_rows[:5],
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2))
    with args.failures_csv.open('w', newline='') as fh:
        fieldnames = ['sequence', 'split', 'archive_basename', 'archive_member', 'error', 'traceback']
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader(); w.writerows(failures)
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(f'{len(failures)} molecules failed or were missing; see {args.failures_csv}')

if __name__ == '__main__':
    main()
