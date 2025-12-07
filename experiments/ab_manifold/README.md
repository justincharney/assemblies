# AB Manifold Recorder

`ab_recorder.py` implements a two-area Assembly Calculus protocol that explores bidirectional projections between areas `A` and `B`. It first stabilizes independent assemblies in each area, then seeds and runs **interleaved** cross-association cycles `A-->B` and `B-->A` while logging every population state. It relies on the shared `experiments.common.recording` helpers so the same pipeline can be reused for other operations by swapping in a different scheduling script. Each run emits compressed `.npz` tensors plus a manifest so manifold analysis notebooks can ingest activity matrices without re-running the simulations.

## Quick Start

```bash
python experiments/ab_manifold/ab_recorder.py \
  --config experiments/ab_manifold/configs/ab_small.json \
  --trials 1 --dump-records
```

Artifacts are written under `runs/ab_manifold/seed<seed>/<timestamp>_<tag>/` and include:
- `config.json`: effective `ABConfig` used for the run.
- `manifest.json`: trial metadata.
- `trial_XXX.npz`: per trial simulation data.
Set `--no-save` for quick smoke tests, `--output-root` to customize the run directory, and `--tag` to append a suffix (e.g., parameter sweep name). `--trials` controls how many independent simulations you run per config/seed; each trial produces its own `.npz`.

## Protocol Phases
The AB recorder runs three distinct phases:

1. **Independent Stabilization**: Both `stimA-->A` and `stimB-->B` project simultaneously with recurrent connections within each area for `stabilize_rounds` iterations. This establishes stable assemblies in `A` and `B` independently.

2. **Bidirectional Association (Interleaved Cycles)**: The protocol first seeds both directions with single steps:
    - `seed_A_to_B`: `stimA` drives `A`, and `A` projects into both `A` and `B`.
    - `seed_B_to_A`: `stimB` drives `B`, and `B` projects into both `B` and `A`.
    
    It then runs `assoc_cycles` interleaved updates. For each cycle `i`:
    - `A_to_B_recur_i`: `stimA` drives `A`; `A` projects to `A` and `B`; `B` also recurs on itself.
    - `B_to_A_recur_i`: `stimB` drives `B`; `B` projects to `B` and `A`; `A` also recurs on itself.

This interleaved schedule co-evolves the assemblies in A and B rather than training one direction to convergence before the other.

## Output Format

Each `.npz` file stores, for every tracked area (`A/B`):

- `<area>_times`: Time index per recorded snapshot.
- `<area>_stages`: Stage labels (e.g., `stabilize_5`, `A_to_B_recur_3`, `B_to_A_recur_7`).
- `<area>_indices`: Object array of k-sized winner lists for each step.
- `<area>_dense` (optional): Binary `time x neuron` matrices when `record_dense` is enabled.

Use NumPy to load the tensors directly:

```python
import numpy as np
data = np.load("runs/ab_manifold/.../trial_000.npz", allow_pickle=True)
times = data["A_times"]
stages = data["A_stages"]
dense_a = data["A_dense"]
dense_b = data["B_dense"]
```

## Configuration Parameters

The `ABConfig` dataclass supports the following parameters:

- `seed`: Random seed for reproducibility
- `n_neurons`: Number of neurons in each area (`A` and `B`)
- `k`: Assembly size (number of winners per projection)
- `p`: Connection probability
- `beta`: Plasticity parameter
- `stabilize_rounds`: Number of rounds for independent `A`/`B` stabilization
- `assoc_cycles`: Number of interleaved association cycles; each cycle consists of one `A_to_B_recur_i` step and one `B_to_A_recur_i` step.
- `record_dense`: Whether to save full binary activity matrices

Example config (`ab_small.json`):
```json
{
  "seed":             123,
  "n_neurons":        5000,
  "k":                200,
  "p":                0.02,
  "beta":             0.05,
  "stabilize_rounds": 30,
  "assoc_cycles":     30,
  "record_dense":     true
}
```

> **Note on scale:** For quick iterations and analysis prototyping, use smaller
> parameter values $(n \approx 4000, k \approx 200)$. For publication-scale experiments matching
> theoretical predictions, scale up to $n \approx 10^6$, $k \approx 10^3$, but be aware these runs
> are computationally intensive and may require significant memory and time.