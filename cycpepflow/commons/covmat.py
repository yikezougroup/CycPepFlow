import math
import os
from copy import deepcopy
from functools import partial
from multiprocessing import Pool, get_context
from typing import Callable, List

import datamol as dm
import numpy as np
import pandas as pd
import torch
from datamol.types import Mol
from loguru import logger as log
from rdkit.Chem import rdMolAlign as MA
from rdkit.Chem.rdchem import Conformer
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from rdkit.Chem.rdmolops import RemoveHs
from rdkit.Geometry import Point3D
from tqdm import tqdm


def build_conformer(pos):
    if isinstance(pos, torch.Tensor) or isinstance(pos, np.ndarray):
        pos = pos.tolist()

    conformer = Conformer()

    for i, atom_pos in enumerate(pos):
        conformer.SetAtomPosition(i, Point3D(*atom_pos))

    return conformer


def set_multiple_rdmol_positions(rdkit_mol, pos):
    """
    Args:
        rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
        pos: (n, N_atoms, 3)
    """
    mol = deepcopy(rdkit_mol)
    for conf_pos in pos:
        conformer = build_conformer(conf_pos)
        mol.AddConformer(conformer, assignId=True)
    return mol


def set_rdmol_positions(rdkit_mol, pos):
    """
    Args:
        rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
        pos: (N_atoms, 3)
    """
    mol = deepcopy(rdkit_mol)
    conformer = build_conformer(pos)
    mol.AddConformer(conformer)
    return mol


def get_best_rmsd(probe, ref, use_alignmol=False):
    probe = RemoveHs(probe)
    ref = RemoveHs(ref)

    try:
        if use_alignmol:
            return MA.AlignMol(probe, ref)
        else:
            rmsd = MA.GetBestRMS(probe, ref)
    except:  # noqa
        rmsd = np.nan

    return rmsd


def get_rmsd(ref_mol: Mol, gen_mols: List[Mol], useFF=False, use_alignmol=False):
    num_gen = len(gen_mols)
    rmsd_vals = []
    for i in range(num_gen):
        gen_mol = gen_mols[i]
        if useFF:
            # print('Applying FF on generated molecules...')
            MMFFOptimizeMolecule(gen_mol)
        rmsd_vals.append(get_best_rmsd(gen_mol, ref_mol, use_alignmol=use_alignmol))

    return rmsd_vals


def calc_performance_stats(rmsd_array, threshold):
    coverage_recall = np.mean(
        np.nanmin(rmsd_array, axis=1, keepdims=True) < threshold, axis=0
    )
    amr_recall = np.mean(np.nanmin(rmsd_array, axis=1))
    coverage_precision = np.mean(
        np.nanmin(rmsd_array, axis=0, keepdims=True) < np.expand_dims(threshold, 1),
        axis=1,
    )
    amr_precision = np.mean(np.nanmin(rmsd_array, axis=0))

    return coverage_recall, amr_recall, coverage_precision, amr_precision


def worker_fn(job, useFF=False, use_alignmol=False):
    smi, i_true, ref_mol, gen_mols = job
    rmsd_vals = get_rmsd(ref_mol, gen_mols, useFF=useFF, use_alignmol=use_alignmol)
    return smi, i_true, rmsd_vals


_COVMAT_WORKER_TRUE_MOLS = None
_COVMAT_WORKER_GEN_MOLS = None
_COVMAT_WORKER_USE_FF = False
_COVMAT_WORKER_USE_ALIGNMOL = False

_COVMAT_ALL_TRUE_MOLS = None
_COVMAT_ALL_GEN_MOLS = None
_COVMAT_ALL_USE_FF = False
_COVMAT_ALL_USE_ALIGNMOL = False


def _init_chunk_worker(true_mols, gen_mols, useFF=False, use_alignmol=False):
    """Install one molecule's conformer lists in forked workers.

    The original per-reference jobs sent the whole ``gen_mols`` RDKit list through
    multiprocessing for every reference conformer.  For CREMP macrocycles this can
    mean thousands of repeated pickles of an 8k-conformer list.  With Linux/fork
    workers, keeping the molecule-level lists in process globals lets each task
    send only reference indices and returns chunked RMSD rows.
    """
    global _COVMAT_WORKER_TRUE_MOLS
    global _COVMAT_WORKER_GEN_MOLS
    global _COVMAT_WORKER_USE_FF
    global _COVMAT_WORKER_USE_ALIGNMOL

    _COVMAT_WORKER_TRUE_MOLS = true_mols
    _COVMAT_WORKER_GEN_MOLS = gen_mols
    _COVMAT_WORKER_USE_FF = useFF
    _COVMAT_WORKER_USE_ALIGNMOL = use_alignmol


def _chunk_worker(index_chunk):
    rows = []
    for i_true in index_chunk:
        rmsd_vals = get_rmsd(
            _COVMAT_WORKER_TRUE_MOLS[i_true],
            _COVMAT_WORKER_GEN_MOLS,
            useFF=_COVMAT_WORKER_USE_FF,
            use_alignmol=_COVMAT_WORKER_USE_ALIGNMOL,
        )
        rows.append((i_true, rmsd_vals))
    return rows


def _init_persistent_worker(all_true_mols, all_gen_mols, useFF=False, use_alignmol=False):
    """Install all molecule conformer lists once per forked worker.

    This avoids repeatedly creating and destroying a 100+ process pool for every
    CREMP molecule. Workers inherit the parent RDKit objects via Linux fork/COW;
    queued tasks send only a molecule key plus reference indices.
    """
    global _COVMAT_ALL_TRUE_MOLS
    global _COVMAT_ALL_GEN_MOLS
    global _COVMAT_ALL_USE_FF
    global _COVMAT_ALL_USE_ALIGNMOL

    _COVMAT_ALL_TRUE_MOLS = all_true_mols
    _COVMAT_ALL_GEN_MOLS = all_gen_mols
    _COVMAT_ALL_USE_FF = useFF
    _COVMAT_ALL_USE_ALIGNMOL = use_alignmol


def _persistent_chunk_worker(job):
    smiles, index_chunk = job
    assert _COVMAT_ALL_TRUE_MOLS is not None
    assert _COVMAT_ALL_GEN_MOLS is not None
    true_mols = _COVMAT_ALL_TRUE_MOLS[smiles]
    gen_mols = _COVMAT_ALL_GEN_MOLS[smiles]
    rows = []
    for i_true in index_chunk:
        rmsd_vals = get_rmsd(
            true_mols[i_true],
            gen_mols,
            useFF=_COVMAT_ALL_USE_FF,
            use_alignmol=_COVMAT_ALL_USE_ALIGNMOL,
        )
        rows.append((i_true, rmsd_vals))
    return smiles, rows


class CovMatEvaluator(object):
    """Coverage Recall Metrics Calculation for GEOM-Dataset"""

    def __init__(
        self,
        num_workers: int = 8,
        use_force_field: bool = False,
        use_alignmol: bool = False,
        thresholds: np.ndarray = np.arange(0.05, 3.05, 0.05),
        ratio: int = 2,
        filter_disconnected: bool = True,
        print_fn: Callable = print,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.use_force_field = use_force_field
        self.use_alignmol = use_alignmol
        self.thresholds = np.array(thresholds).flatten()

        self.ratio = ratio
        self.filter_disconnected = filter_disconnected

        self.print_fn = print_fn

    def __call__(self, packed_data_list, start_idx=0):
        rmsd_results = {}
        true_mols = {}
        gen_mols = {}
        for data in packed_data_list:
            if "pos_gen" not in data or "pos_ref" not in data:
                log.info("skipping due to missing pos_gen or pos_ref")
                continue
            if self.filter_disconnected and ("." in data["smiles"]):
                log.info("skipping due to disconnected molecule")
                continue

            num_atoms = data["pos_gen"].shape[1]
            if isinstance(data["pos_gen"], torch.Tensor):
                data["pos_gen"] = data["pos_gen"].cpu().numpy()

            smiles = data["smiles"]
            mol = dm.to_mol(smiles, remove_hs=False, ordered=True)
            data["pos_ref"] = data["pos_ref"].reshape(-1, num_atoms, 3)
            data["pos_gen"] = data["pos_gen"].reshape(-1, num_atoms, 3)

            num_true = data["pos_ref"].shape[0]
            num_gen = num_true * self.ratio
            if data["pos_gen"].shape[0] < num_gen:
                log.info("skipping due to insufficient number of generated conformers")
                continue
            data["pos_gen"] = data["pos_gen"][:num_gen]

            true_mols[smiles] = [
                set_rdmol_positions(mol, data["pos_ref"][i]) for i in range(num_true)
            ]
            gen_mols[smiles] = [
                set_rdmol_positions(mol, data["pos_gen"][i]) for i in range(num_gen)
            ]

            rmsd_results[smiles] = {
                "n_true": num_true,
                "n_model": num_gen,
                "rmsd": np.nan * np.ones((num_true, num_gen)),
            }
        smiles_order = list(rmsd_results.keys())

        # remove packed_data_list from memory
        del packed_data_list

        env_chunk_size = os.environ.get("CYCPEPFLOW_COVMAT_CHUNK_SIZE")
        if env_chunk_size:
            try:
                env_chunk_size = max(1, int(env_chunk_size))
            except ValueError:
                log.warning(
                    "Ignoring invalid CYCPEPFLOW_COVMAT_CHUNK_SIZE=%r", env_chunk_size
                )
                env_chunk_size = None

        pool_mode = os.environ.get("CYCPEPFLOW_COVMAT_POOL_MODE", "per_molecule")
        pool_mode = pool_mode.strip().lower().replace("-", "_")
        use_persistent_pool = pool_mode in {"persistent", "global", "all_molecules"}

        if use_persistent_pool and self.num_workers > 1:
            workers = min(
                max(1, int(self.num_workers)),
                max(1, sum(int(r["n_true"]) for r in rmsd_results.values())),
            )
            jobs = []
            for smiles in smiles_order:
                num_true = int(rmsd_results[smiles]["n_true"])
                workers_for_mol = min(workers, max(1, num_true))
                chunk_size = env_chunk_size or max(
                    1, math.ceil(num_true / max(1, workers_for_mol * 4))
                )
                for i in range(0, num_true, chunk_size):
                    jobs.append((smiles, list(range(i, min(i + chunk_size, num_true)))))

            self.print_fn(
                "CovMat persistent pool: "
                f"workers={workers} molecules={len(smiles_order)} chunks={len(jobs)}"
            )
            ctx = get_context("fork")
            with ctx.Pool(
                workers,
                initializer=_init_persistent_worker,
                initargs=(
                    true_mols,
                    gen_mols,
                    self.use_force_field,
                    self.use_alignmol,
                ),
            ) as p:
                for smiles, chunk_rows in tqdm(
                    p.imap_unordered(_persistent_chunk_worker, jobs, chunksize=1),
                    total=len(jobs),
                    desc=f"CovMat chunks persistent workers={workers}",
                ):
                    for i_true, rmsd_vals in chunk_rows:
                        rmsd_results[smiles]["rmsd"][i_true] = rmsd_vals
        else:
            for mol_idx, smiles in enumerate(
                tqdm(smiles_order, desc="CovMat molecules", total=len(smiles_order))
            ):
                num_true = rmsd_results[smiles]["n_true"]
                # Efficiency patch for CREMP-scale COV/MAT scoring on high-core CPU nodes:
                # do not spawn more workers than there are reference conformers for a molecule.
                # Several CREMP molecules have only tens of refs; starting a 192-process pool
                # for them wastes time and memory.  Large molecules still use all requested
                # workers, and chunks keep only reference indices in the multiprocessing queue.
                workers_for_mol = min(max(1, int(self.num_workers)), max(1, int(num_true)))
                if workers_for_mol > 1:
                    chunk_size = env_chunk_size or max(
                        1, math.ceil(num_true / max(1, workers_for_mol * 4))
                    )
                    chunks = [
                        list(range(i, min(i + chunk_size, num_true)))
                        for i in range(0, num_true, chunk_size)
                    ]
                    ctx = get_context("fork")
                    with ctx.Pool(
                        workers_for_mol,
                        initializer=_init_chunk_worker,
                        initargs=(
                            true_mols[smiles],
                            gen_mols[smiles],
                            self.use_force_field,
                            self.use_alignmol,
                        ),
                    ) as p:
                        for chunk_rows in tqdm(
                            p.imap_unordered(_chunk_worker, chunks, chunksize=1),
                            total=len(chunks),
                            desc=f"CovMat refs {mol_idx + 1}/{len(smiles_order)} workers={workers_for_mol}",
                            leave=False,
                        ):
                            for i_true, rmsd_vals in chunk_rows:
                                rmsd_results[smiles]["rmsd"][i_true] = rmsd_vals
                else:
                    for i_true in tqdm(
                        range(num_true),
                        desc=f"CovMat refs {mol_idx + 1}/{len(smiles_order)}",
                        leave=False,
                    ):
                        rmsd_results[smiles]["rmsd"][i_true] = get_rmsd(
                            true_mols[smiles][i_true],
                            gen_mols[smiles],
                            useFF=self.use_force_field,
                            use_alignmol=self.use_alignmol,
                        )

        stats = []
        for res in rmsd_results.values():
            stats_ = calc_performance_stats(res["rmsd"], self.thresholds)
            stats.append(stats_)
        coverage_recall, amr_recall, coverage_precision, amr_precision = zip(*stats)

        results = {
            "CoverageR": np.array(coverage_recall),  # (num_mols, num_threshold)
            "MatchingR": np.array(amr_recall),  # (num_mols)
            "thresholds": self.thresholds,
            "CoverageP": np.array(coverage_precision),  # (num_mols, num_threshold)
            "MatchingP": np.array(amr_precision),  # (num_mols)
        }
        # print_conformation_eval_results(results)
        return results, rmsd_results


def print_covmat_results(results, print_fn=print):
    df = pd.DataFrame.from_dict(
        {
            "Threshold": results["thresholds"],
            "COV-R_mean": np.mean(results["CoverageR"], 0),
            "COV-R_median": np.median(results["CoverageR"], 0),
            "COV-P_mean": np.mean(results["CoverageP"], 0),
            "COV-P_median": np.median(results["CoverageP"], 0),
        }
    )
    matching_metrics = {
        "MAT-R_mean": np.mean(results["MatchingR"]),
        "MAT-R_median": np.median(results["MatchingR"]),
        "MAT-P_mean": np.mean(results["MatchingP"]),
        "MAT-P_median": np.median(results["MatchingP"]),
    }
    return df, matching_metrics
