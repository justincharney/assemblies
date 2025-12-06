"""Record Assembly Calculus `associate` runs with full population activity.

This script wraps the NEMO Assembly Calculus simulator and reproduces the
`associate` protocol implemented in `nemo/simulations.py`, while logging
all time-by-neuron activity for areas A, B, and C.

For each trial, we:
  1. Build a fresh `brain.Brain` instance (seeded by base_seed + trial_index).
  2. Define stimuli `stimA`, `stimB` and areas `A`, `B`, `C`.
  3. Execute the associate protocol as a sequence of `project(...)` steps:
       - stabilize A and B
       - grow C from A
       - grow C from B
       - jointly associate A and B into C
       - re-evoke B alone and observe C
  4. Record the winners in A/B/C at every step via `ActivityRecorder`.
  5. Compute a simple overlap diagnostic between A→C and B→C projections.

The `persist_trials(...)` helper then writes, per CLI invocation:

  - config.json   : effective AssociateConfig used for this run
  - manifest.json : trial metadata + overlap scores
  - trial_XXX.npz : per-trial activity tensors (times, stages, indices, dense)

This format is designed so manifold analysis code (UMAP, PHATE, GPFA, dPCA, etc.)
can run directly on the recorded assemblies without re-running the simulator.
"""

from __future__ import annotations

import argparse
import copy
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
class AssociateConfig:
    """Hyperparameters for a single associate experiment.

    These fields mirror the parameters used in the Assembly Calculus
    `associate` protocol:

      - seed            : base RNG seed; each trial uses seed + trial_index
      - n_neurons       : number of neurons in each of A, B, and C
      - k               : sparsity level (winners per timestep per area)
      - p               : connection probability in the Brain graph
      - beta            : inhibition/threshold parameter for areas

      - stability_rounds: how long to stabilize A and B before training C
      - a_to_c_rounds   : number of recurrent A→C training steps
      - b_to_c_rounds   : number of recurrent B→C training steps
      - overlap_rounds  : number of joint A,B→C association steps
      - post_b_rounds   : number of B-only recall steps (post-training)

      - record_dense    : if True, ActivityRecorder stores dense (T × n)
                          binary matrices in addition to winner indices, which
                          is convenient for manifold methods but heavier on
                          disk and RAM.
    """
    seed:             int = 0
    n_neurons:        int = 5000
    k:                int = 200
    p:              float = 0.02
    beta:           float = 0.05
    stability_rounds: int = 9
    a_to_c_rounds:    int = 9
    b_to_c_rounds:    int = 9
    overlap_rounds:   int = 10
    post_b_rounds:    int = 9
    record_dense:    bool = True


# --------------------------------------------------------------------------- #
# Trial orchestration
# --------------------------------------------------------------------------- #
class AssociateTrial:
    """Orchestrate a single associate trial with recording and diagnostics."""

    def __init__(self, config: AssociateConfig) -> None:
        self.config = config

    def run(self, *, trial_index: int) -> Dict[str, Any]:
        """Execute one associate trial and return its results.

        Parameters
        ----------
        trial_index :
            Integer index of this trial; used to offset the RNG seed so that
            each trial is independent but reproducible given (seed, trial_index).

        Returns
        -------
        dict with keys:
            - "trial_index": int
            - "overlap"    : float in [0, 1], overlap of A→C vs B→C winners
            - "recorder"   : ActivityRecorder containing the full activity
                             timeline for areas A, B, and C.
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
        sim.add_stimulus("stimA", cfg.k)
        sim.add_stimulus("stimB", cfg.k)

        # Areas: three structurally identical areas A, B, C, each with n_neurons
        # and sparsity k; beta controls inhibition / thresholds.
        sim.add_area("A", cfg.n_neurons, cfg.k, cfg.beta)
        sim.add_area("B", cfg.n_neurons, cfg.k, cfg.beta)
        sim.add_area("C", cfg.n_neurons, cfg.k, cfg.beta)

        # Recorder keeps a timeline of winners per area; optionally dense matrix.
        recorder = ActivityRecorder(
            {"A": cfg.n_neurons, "B": cfg.n_neurons, "C": cfg.n_neurons},
            record_dense = cfg.record_dense,
        )
        step_counter = 0  # logical time index for recorded snapshots

        def project(
            stage: str,
            stim_map: MutableMapping[str, List[str]],
            area_map: MutableMapping[str, List[str]],
        ) -> None:
            """Advance the simulation by one step and log A/B/C winners.

            Parameters
            ----------
            stage :
                Human-readable label for this phase (e.g. "joint_seed",
                "A_to_C_recur_3"). This string is stored alongside the activity
                and is used later to subset by protocol stage.
            stim_map :
                Mapping from stimulus name -> list of areas it drives.
                Example: {"stimA": ["A"], "stimB": ["B"]}.
            area_map :
                Mapping from source area name -> list of target areas it projects
                into. Example: {"A": ["A", "C"], "B": ["B", "C"], "C": ["C"]}.
            """
            nonlocal step_counter

            # One Assembly Calculus update step: apply stimuli + recurrent input.
            sim.project(stim_map, area_map)

            # Snapshot of current winners in each area we track.
            winners_snapshot = {
                name: sim.area_by_name[name].winners for name in ("A", "B", "C")
            }

            # Record this timestep for all areas under the given stage label.
            recorder.log(time=step_counter, stage=stage, winners=winners_snapshot)
            step_counter += 1

        # --------------------- Phase 1: stabilize A and B -------------------- #
        # Repeatedly drive A and B with their stimuli while allowing them to
        # project to themselves, so that each settles into a clean assembly.
        project("init_pair", {"stimA": ["A"], "stimB": ["B"]}, {})
        for i in range(cfg.stability_rounds):
            project(
                f"stabilize_{i}",
                {"stimA": ["A"], "stimB": ["B"]},
                {"A": ["A"], "B": ["B"]},
            )

        # --------------------- Phase 2: grow C from A ------------------------ #
        # First seed C from A, then let A and C co-recur so C learns A's
        # footprint (an A-evoked assembly in C).
        project("seed_A_to_C", {"stimA": ["A"]}, {"A": ["A", "C"]})
        for i in range(cfg.a_to_c_rounds):
            project(
                f"A_to_C_recur_{i}",
                {"stimA": ["A"]},
                {"A": ["A", "C"], "C": ["C"]},
            )

        # --------------------- Phase 3: grow C from B ------------------------ #
        # Symmetric to Phase 2 but with B driving C, so C also learns a
        # B-evoked footprint.
        project("seed_B_to_C", {"stimB": ["B"]}, {"B": ["B", "C"]})
        for i in range(cfg.b_to_c_rounds):
            project(
                f"B_to_C_recur_{i}",
                {"stimB": ["B"]},
                {"B": ["B", "C"], "C": ["C"]},
            )

        # --------------------- Phase 4: joint association -------------------- #
        # Drive A and B together, both projecting into C (and then let C recur).
        # This encourages the C assemblies evoked by A and by B to overlap.
        project(
            "joint_seed",
            {"stimA": ["A"], "stimB": ["B"]},
            {"A": ["A", "C"], "B": ["B", "C"]},
        )
        for i in range(max(cfg.overlap_rounds - 1, 0)):
            project(
                f"joint_assoc_{i}",
                {"stimA": ["A"], "stimB": ["B"]},
                {"A": ["A", "C"], "B": ["B", "C"], "C": ["C"]},
            )

        # --------------------- Phase 5: B-only recall test ------------------- #
        # Re-evoke B alone and track whether C still co-activates in a way that
        # overlaps strongly with the A-driven footprint.
        project("post_B_seed", {"stimB": ["B"]}, {"B": ["B", "C"]})
        for i in range(cfg.post_b_rounds):
            project(
                f"post_B_recur_{i}",
                {"stimB": ["B"]},
                {"B": ["B", "C"], "C": ["C"]},
            )

        overlap_score = self._evaluate_overlap(sim)
        return {
            "trial_index": trial_index,
            "overlap":     overlap_score,
            "recorder":    recorder,
        }

    # ----------------------------- Diagnostics ------------------------------ #
    def _evaluate_overlap(self, sim: brain.Brain) -> float:
        """Quantify overlap between A→C and B→C evoked assemblies.

        We take deep copies of the trained simulator to avoid mutating the
        main simulation state. In each copy we:

          - evoke either A or B with its stimulus,
          - then project that area into C,
          - and read off the winners in C.

        The overlap score is:
            |winners_C_from_A ∩ winners_C_from_B| / k

        If either evoked C assembly is empty, we return 0.0.
        """
        cfg = self.config

        # Probe A→C on a copy of the trained brain.
        brain_from_a = copy.deepcopy(sim)
        brain_from_a.project({"stimA": ["A"]}, {})
        brain_from_a.project({}, {"A": ["C"]})
        a_into_c = set(brain_from_a.area_by_name["C"].winners)

        # Probe B→C on a separate copy.
        brain_from_b = copy.deepcopy(sim)
        brain_from_b.project({"stimB": ["B"]}, {})
        brain_from_b.project({}, {"B": ["C"]})
        b_into_c = set(brain_from_b.area_by_name["C"].winners)

        if not a_into_c or not b_into_c:
            return 0.0
        return len(a_into_c & b_into_c) / float(cfg.k)


# --------------------------------------------------------------------------- #
# CLI parsing and execution
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the associate recorder CLI."""
    parser = argparse.ArgumentParser(
        description = "Record Assembly Calculus associate runs"
    )
    parser.add_argument(
        "--config",
        type    = Path,
        default = None,
        help    = "Path to JSON config overriding default AssociateConfig() fields",
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
        default = Path("runs/associate_manifold"),
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


def load_config(path: Optional[Path]) -> AssociateConfig:
    """Load AssociateConfig from JSON, or return defaults if no path is given."""
    if path is None:
        return AssociateConfig()
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return AssociateConfig(**data)


def main() -> None:
    """Entry point: run N associate trials and optionally persist their activity."""
    args   = parse_args()
    config = load_config(args.config)

    runner = AssociateTrial(config)
    trials: List[Dict[str, Any]] = []

    # Run the requested number of independent trials.
    for idx in range(args.trials):
        trial = runner.run(trial_index=idx)
        trials.append(trial)
        print(f"trial={idx} overlap={trial['overlap']:.3f}")

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
