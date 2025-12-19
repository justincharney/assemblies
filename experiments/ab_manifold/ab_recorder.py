"""Record bidirectional A<-->B association runs with full population activity.

This script wraps the NEMO Assembly Calculus simulator and implements a
two-area bidirectional association protocol. It explores how assemblies in
areas A and B can be cross-linked through interleaved projections in both
directions.

Crucially, A and B are allowed to have *different* hyperparameters
(k_A, k_B, beta_A, beta_B), so their independent stabilization dynamics
live on different manifolds. The subsequent A<->B association phase then
tests whether bidirectional projections align these initially distinct
population codes.

For each trial, we:
  1. Build a fresh `brain.Brain` instance (seeded by base_seed + trial_index).
  2. Define stimuli `stimA`, `stimB` and areas `A`, `B` with (k_A, beta_A)
     and (k_B, beta_B), respectively.
  3. Execute the AB protocol as a sequence of `project(...)` steps:
       - stabilize independent assemblies in A and B
       - seed bidirectional links A-->B and B-->A
       - run `assoc_cycles` interleaved A-->B and B-->A updates
  4. Record the winners in A and B at every step via `ActivityRecorder`.
  5. Store full activity timelines for downstream manifold analysis.

The `persist_trials(...)` helper then writes, per CLI invocation:

  - config.json   : effective ABConfig used for this run
  - manifest.json : trial metadata
  - trial_XXX.npz : per-trial activity tensors (times, stages, indices, dense)

This format is designed so manifold analysis code (UMAP, PHATE, GPFA, dPCA, etc.)
can run directly on the recorded assemblies without re-running the simulator.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, MutableMapping, Optional

# --------------------------------------------------------------------------- #
# Import wiring: add repo root and `nemo/` to sys.path so that, regardless of
# whether this script is invoked from the repo root or experiments directory,
# we can `import brain` and shared experiment utilities consistently.
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[2]
NEMO_DIR = REPO_ROOT / "nemo"
if str(NEMO_DIR) not in sys.path:
    sys.path.insert(0, str(NEMO_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common.recording import ActivityRecorder, persist_trials

import brain  # type: ignore  # pylint: disable=import-error


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class ABConfig:
    """Hyperparameters for a single A<-->B bidirectional association experiment.

    These fields control the two-area protocol:

      - seed      : base RNG seed; each trial uses seed + trial_index
      - n_neurons : number of neurons in each of A and B (same N, different k/beta)
      - k_A       : sparsity level (winners per timestep) in area A
      - k_B       : sparsity level (winners per timestep) in area B
      - p         : connection probability in the Brain graph
      - beta_A    : inhibition/threshold parameter in area A
      - beta_B    : inhibition/threshold parameter in area B

      - stabilize_rounds : how long to stabilize A and B independently
      - assoc_cycles     : how many interleaved A->B and B->A cycles to run
                           (each cycle consists of one A_to_B step followed
                            by one B_to_A step)

      - record_dense     : if True, ActivityRecorder stores dense (T x n)
                           binary matrices in addition to winner indices,
                           which is convenient for manifold methods but
                           heavier on disk and RAM.
    """
    seed:             int = 0
    n_neurons:        int = 5000
    k_A:              int = 200
    k_B:              int = 200
    p:              float = 0.02
    beta_A:         float = 0.05
    beta_B:         float = 0.08

    stabilize_rounds: int = 30
    assoc_cycles:     int = 100

    record_dense:    bool = True


# --------------------------------------------------------------------------- #
# Trial orchestration
# --------------------------------------------------------------------------- #
class ABTrial:
    """Orchestrate a single A<-->B bidirectional association trial with recording."""

    def __init__(self, config: ABConfig) -> None:
        self.config = config

    def run(self, *, trial_index: int) -> Dict[str, Any]:
        """Execute one AB trial and return its results.

        Parameters
        ----------
        trial_index :
            Integer index of this trial; used to offset the RNG seed so that
            each trial is independent but reproducible given (seed, trial_index).

        Returns
        -------
        dict with keys:
            - "trial_index": int
            - "recorder"   : ActivityRecorder containing the full activity
                             timeline for areas A and B.
        """
        cfg = self.config

        # Build a fresh simulator per trial so that state is not shared across
        # trials and stochasticity is controlled by (seed + trial_index).
        sim = brain.Brain(
            cfg.p,
            save_size    = False,
            save_winners = False,
            seed         = cfg.seed + trial_index,
        )

        # Stimuli: two separate k-hot sources that will drive A and B.
        # Note that k_A and k_B can differ, which already breaks symmetry.
        sim.add_stimulus("stimA", cfg.k_A)
        sim.add_stimulus("stimB", cfg.k_B)

        # Areas: two areas A and B with the same neuron count but
        # potentially different sparsity and inhibition parameters.
        sim.add_area("A", cfg.n_neurons, cfg.k_A, cfg.beta_A)
        sim.add_area("B", cfg.n_neurons, cfg.k_B, cfg.beta_B)

        # Recorder keeps a timeline of winners per area; optionally dense matrix.
        recorder = ActivityRecorder(
            {"A": cfg.n_neurons, "B": cfg.n_neurons},
            record_dense = cfg.record_dense,
        )
        step_counter = 0  # logical time index for recorded snapshots

        def project(
            stage: str,
            stim_map: MutableMapping[str, List[str]],
            area_map: MutableMapping[str, List[str]],
        ) -> None:
            """Advance the simulation by one step and log A/B winners.

            Parameters
            ----------
            stage :
                Human-readable label for this phase (e.g. "stabilize_5",
                "A_to_B_recur_3"). This string is stored alongside the activity
                and is used later to subset by protocol stage.
            stim_map :
                Mapping from stimulus name -> list of areas it drives.
                Example: {"stimA": ["A"], "stimB": ["B"]}.
            area_map :
                Mapping from source area name -> list of target areas it projects
                into. Example: {"A": ["A", "B"], "B": ["B"]}.
            """
            nonlocal step_counter

            # One Assembly Calculus update step: apply stimuli + recurrent input.
            sim.project(stim_map, area_map)

            # Snapshot of current winners in each area we track.
            winners_snapshot = {
                name: sim.area_by_name[name].winners for name in ("A", "B")
            }

            # Record this timestep for all areas under the given stage label.
            recorder.log(time=step_counter, stage=stage, winners=winners_snapshot)
            step_counter += 1

        # --------------------- Phase 1: stabilize A and B -------------------- #
        # Repeatedly drive A and B with their stimuli while allowing them to
        # project to themselves, so that each settles into a clean assembly.
        # Because (k_A, beta_A) and (k_B, beta_B) may differ, the two areas
        # generally live on different manifolds during this phase.
        project("init_pair", {"stimA": ["A"], "stimB": ["B"]}, {})
        for i in range(cfg.stabilize_rounds):
            project(
                f"stabilize_{i}",
                {"stimA": ["A"], "stimB": ["B"]},
                {"A": ["A"], "B": ["B"]},
            )

        # ------------- Phases 2+3: interleaved A<->B association ------------ #
        # First, give each direction one "seed" step so both links exist.
        project("seed_A_to_B", {"stimA": ["A"]}, {"A": ["A", "B"]})
        project("seed_B_to_A", {"stimB": ["B"]}, {"B": ["B", "A"]})

        # Then run assoc_cycles interleaved A->B and B->A steps.
        # One cycle = A_to_B_recur_i followed by B_to_A_recur_i.
        for i in range(cfg.assoc_cycles):
            project(
                f"A_to_B_recur_{i}",
                {"stimA": ["A"]},
                {"A": ["A", "B"], "B": ["B"]},
            )
            project(
                f"B_to_A_recur_{i}",
                {"stimB": ["B"]},
                {"B": ["B", "A"], "A": ["A"]},
            )

        return {
            "trial_index": trial_index,
            "recorder":    recorder,
        }


# --------------------------------------------------------------------------- #
# CLI parsing and execution
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the AB recorder CLI."""
    parser = argparse.ArgumentParser(
        description = "Record Assembly Calculus A<-->B bidirectional association runs"
    )
    parser.add_argument(
        "--config",
        type    = Path,
        default = None,
        help    = "Path to JSON config overriding default ABConfig() fields",
    )
    parser.add_argument(
        "--trials",
        type    = int,
        default = 1,
        help    = "How many independent simulations (with different seeds) to run",
    )
    parser.add_argument(
        "--output-root",
        type    = Path,
        default = Path("runs/ab_manifold"),
        help    = (
            "Directory where run folders (config.json, manifest.json, "
            "trial_XXX.npz) are saved"
        ),
    )
    parser.add_argument(
        "--tag",
        type    = str,
        default = None,
        help    = "Optional suffix for the run directory name (e.g. 'k_sweep')",
    )
    parser.add_argument(
        "--no-save",
        action = "store_true",
        help   = "Run simulations but skip writing artifacts to disk",
    )
    parser.add_argument(
        "--dump-records",
        action = "store_true",
        help   = (
            "Print a JSON preview of the recorded winners for the first trial "
            "(useful for sanity checking stage names and shapes)"
        ),
    )
    return parser.parse_args()


def load_config(path: Optional[Path]) -> ABConfig:
    """Load ABConfig from JSON, or return defaults if no path is given."""
    if path is None:
        return ABConfig()
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return ABConfig(**data)


def main() -> None:
    """Entry point: run N AB trials and optionally persist their activity."""
    args   = parse_args()
    config = load_config(args.config)

    runner = ABTrial(config)
    trials: List[Dict[str, Any]] = []

    # Run the requested number of independent trials.
    for idx in range(args.trials):
        trial = runner.run(trial_index=idx)
        trials.append(trial)
        print(f"trial={idx} done")

    # Optional quick peek at the logged winners for sanity checking.
    if args.dump_records and trials:
        preview = trials[0]["recorder"].preview()
        print(json.dumps(preview, indent=2))

    # Persist run artifacts (config, manifest, npz tensors) unless skipped.
    if not args.no_save:
        run_dir = persist_trials(
            trials      = trials,
            config      = config,
            output_root = args.output_root,
            tag         = args.tag,
        )
        if run_dir is not None:
            print(f"Saved run artifacts to {run_dir}")


if __name__ == "__main__":
    main()
