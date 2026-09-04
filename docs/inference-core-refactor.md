# Inference-core refactor verification

Date: 2026-09-05. Baseline: `2de6d4c0db25d46acb8cd568bcfebbd747d6ab94`.
Scope: the released inference/evaluation workflow, not a new training implementation.
This is an unreleased refactor; the published v0.1.1 tag and assets are unchanged.

## Size and dependency changes

Counts are physical lines and AST function/method definitions in `cycpepflow/**/*.py`,
excluding tests, scripts, build outputs and checkpoints.

| Measure | Baseline | Refactored |
|---|---:|---:|
| Python package files | 26 | 17 |
| Physical source lines | 4,958 | 2,583 |
| Function/method definitions | 197 | 95 |
| Direct project dependencies | 15 | 11 |

The package is 47.90% smaller by physical source lines. Nine modules were deleted:
`commons/configs.py`, `commons/sample.py`, `commons/xtb.py`, `models/base.py`,
`models/loss.py`, both `optim/` files, and both `schedulers/` files.

Removed runtime dependencies: Lightning, TorchMetrics, Pydantic and fsspec.
Other packages may still bring some of these in transitively.

## What changed

- Replaced the Lightning base hierarchy with a plain PyTorch inference module.
  Removed training hooks, losses/curricula, optimizer/scheduler code, upstream
  checkpoint download configuration, xTB helpers, redundant sampling APIs,
  unused evaluator workers/classes, unused feature utilities and unused network
  output/distance classes.
- Retained original checkpoint state names, parameter shapes, initialization/RNG
  order, feature category encodings, prior and integration mathematics, dynamic
  radius construction, configuration-selected network branches and scoring definitions.
- Build covalent hop topology and APG shortest paths once per sampling invocation,
  rather than at every network evaluation. The radius graph remains dynamic.
  The normal standalone `forward` path still computes its own topology.
- Reuse one molecule parse per dataset record. Reference conformers are built from
  that record's own coordinates, fixing stale references for repeated SMILES.
- Avoid CUDA time-window scalar comparisons on the deterministic ODE branch.
- Keep all four YAML configurations unchanged; their historical training options
  remain accepted as inert compatibility metadata. Unknown arguments are rejected.

The old `BaseFlow.predict`, `from_default`, `batched_sampling`, `CovMatEvaluator`
and training APIs are intentionally removed. These were not called by the released
workflow; this is not a promise of compatibility for unknown external callers.
See README for supported entry points and migration routes.

## Executed validation

1. **Regression suite:** 28 passed, no warnings in the final run. The baseline
   suite contained 5 tests. New tests guard the inference boundary, imports,
   safe record loading, atom/reference ordering, topology reuse, serialization
   and retained scoring semantics. Removal/refactor guards were observed failing
   before the corresponding changes.
2. **Checkpoints:** all four released checkpoints passed SHA-256 validation and
   strict state loading. Parameter counts remain 18,303,745 (B), 18,309,185
   (APG-B), 43,422,219 (L) and 43,430,379 (APG-L).
3. **GPU coordinate parity:** 24 original-versus-refactored cases: four actual
   checkpoints, three held-out molecular topologies (4-/5-/6-mer), and FP32/BF16
   network modes. Each case used two conformers and the released 50-step sampler.
   All 24 coordinate tensors were **bitwise identical**, maximum absolute
   coordinate difference **0.0**, under identical deterministic test settings:
   `CUBLAS_WORKSPACE_CONFIG=:4096:8` and
   `torch.use_deterministic_algorithms(True)`.
4. **Nondeterminism control:** ordinary seeded GPU execution was not bitwise
   reproducible even for baseline-versus-baseline runs. The largest observed
   original-repeat coordinate difference was 0.04081 Angstrom; original-versus-
   refactored ordinary-mode difference was 0.04815 Angstrom. These are separate
   diagnostic observations, not a tolerance chosen to pass the refactor.
   Production precision/determinism defaults were not changed.
5. **End-to-end CPU check:** conversion, real B-checkpoint 50-step generation,
   persistent-pool COV/MAT scoring and stereochemistry scoring completed on one
   cyclic 4-mer with two seeded RDKit ETKDGv3 reference conformers and four generated
   conformers. Generated coordinates and scoring CSVs matched the baseline exactly.
   This is an explicitly constructed regression fixture, **not CREMP reference
   data and not a reproduction of paper metrics**. No relaxation was applied.
6. **Independent review:** no blocking security or logic findings. The reviewer
   independently reported exact CPU checks for 144 hop-order cases, 10 molecular
   featurizer cases, three mixed-size APG comparisons, and eight cached-versus-
   original sampling cases spanning both priors, both samplers and post-hoc
   chirality with nonzero global/APG weights. The review was static/CPU only;
   the parent subsequently completed the GPU validation in item 3.
7. **Packaging:** a clean-source wheel contained only the 17 retained package
   Python files, excluded the four removed direct dependencies, and preserved
   the upstream MIT text. Direct import from the built wheel and a strict real
   B-checkpoint load succeeded without importing Lightning/TorchMetrics.
8. **Static gates:** `git diff --check`, the release audit, and focused Ruff
   `E4,E7,E9,F` checks over the package and tests passed. This is not a claim that
   unrelated pre-existing script/style warnings were all cleaned up.

Useful commands from the repository root:

```bash
python -m pytest tests -q
python scripts/verify_checkpoints.py
python scripts/audit_release.py
ruff check --isolated --select E4,E7,E9,F cycpepflow tests
```

## Environment and limits

Executed tests used the existing local Python 3.10.16 / PyTorch 2.4.1+cu121 /
PyG 2.6.1 / torch-cluster 1.6.3+pt24cu124 / RDKit 2024.09.2 / datamol 0.12.5
runtime. This differs from the documented paper environment and is outside the
package's declared Python 3.11/3.12 installation range; no fresh Conda solve or
full supported-environment matrix was performed. The clean wheel was built
separately using the packaging environment.

No new 1,000-molecule CREMP benchmark or production-batch throughput benchmark
was run. The runtime optimization is evidenced by eliminating repeated topology
work and preserving outputs, not by a claimed end-to-end speedup percentage.
More invasive attention packing, harmonic-prior matrix restructuring and scorer
memory redesign were deliberately deferred to avoid mixing numerical/execution
changes into this cleanup.

`LICENSE`, `NOTICE` and `LICENSES/ET-Flow-MIT.txt` are byte-for-byte unchanged.
Retained derived architecture still requires its upstream attribution; reducing
source size does not remove that obligation.
