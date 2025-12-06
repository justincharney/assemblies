# Associate Manifold Recorder

`associate_recorder.py` reproduces the Assembly Calculus `associate` protocol
implemented in `nemo/simulations.py` while logging every population state in
areas A, B, and C. It relies on the shared `experiments.common.recording`
helpers so the same pipeline can be reused for other operations (e.g., project,
merge) by swapping in a different scheduling script. Each run emits compressed
`.npz` tensors plus a manifest so manifold analysis notebooks can ingest activity
matrices without re-running the simulations.

## Quick Start

```bash
python experiments/associate_manifold/associate_recorder.py \
  --config experiments/associate_manifold/configs/associate_small.json \
  --trials 1 --dump-records
```

Artifacts are written under `runs/associate_manifold/<operation>/<sweep_id>/seed<seed>/<timestamp>_<tag>/`
and include `config.json`, `manifest.json`, and one `trial_XXX.npz` per simulation.
Set `--no-save` for quick smoke tests, `--output-root` to customize the run
directory, and `--tag` to append a suffix (e.g., parameter sweep name).
`--trials` controls how many independent cue/target pairs you simulate per
config/seed; each trial produces its own `.npz` and overlap score.

### Batch Sweeps

Use `sweep.py` to run multiple configs/seeds in one shot:

```bash
python experiments/associate_manifold/sweep.py \
  --sweep-config experiments/associate_manifold/configs/sweep_example.json \
  --output-root runs/associate_manifold --trials 1
```

Each variant is written under `runs/associate_manifold/associate/<sweep_id>/seed<seed>/…`
so all artifacts for the same operation stay grouped. Pass `--dry-run` to print
commands without executing them.

## Output Format

Each `.npz` file stores, for every tracked area (A/B/C):

- `<area>_times`: Time index per recorded snapshot.
- `<area>_stages`: Stage labels (e.g., `joint_seed`, `post_B_recur_2`).
- `<area>_indices`: Object array of k-sized winner lists for each step.
- `<area>_dense` (optional): Binary time x neuron matrices when `record_dense`
  is enabled.

Use NumPy to load the tensors directly:

```python
import numpy as np
data = np.load("runs/associate_manifold/.../trial_000.npz", allow_pickle=True)
times = data["A_times"]
dense_c = data["C_dense"]
```

The manifest also records the overlap score computed by projecting `stimA` and
`stimB` separately into area C after training for quick sanity checks.

> **Note on scale:** `sweep_example.json` targets the paper-scale regime
> $(n \approx 10^6, p \approx 10^-3, k \approx 10^2-10^3)$. Those runs are computationally heavy, so use a
> smaller config (e.g., `associate_small.json`) while iterating on analysis code
> and switch to the large-scale sweep once you’re ready to generate final data.
