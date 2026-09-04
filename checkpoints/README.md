# Model checkpoints

The four inference-only Lightning checkpoints are attached to the private GitHub release **`v0.1.0`** rather than committed as Git blobs.

Download and verify them from an authenticated checkout:

```bash
python scripts/download_checkpoints.py
python scripts/verify_checkpoints.py
```

Expected files:

- `cycpepflow-b.ckpt`
- `cycpepflow-apg-b.ckpt`
- `cycpepflow-l.ckpt`
- `cycpepflow-apg-l.ckpt`

`checkpoint_manifest.json` records the selected epoch/global step, parameter count, original training-checkpoint digest, release-checkpoint digest, and byte size. `SHA256SUMS` is suitable for `sha256sum -c`.

The released files retain the complete model `state_dict` and inference metadata while omitting optimizer/scheduler/trainer state. They are intended to reproduce paper inference and evaluation, not resume training.
