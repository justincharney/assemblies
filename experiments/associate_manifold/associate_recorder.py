"""Associate operation recorder for the Assembly Calculus simulator.

This script reproduces the `associate` protocol from `nemo/simulations.py`
while capturing time-by-neuron activity snapshots for areas A, B, and C. The
resulting tensors are written to compressed `.npz` bundles so downstream
manifold analyses (GPFA, dPCA, PHATE, etc.) can run directly on the recorded
population activity.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, MutableMapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
NEMO_DIR = REPO_ROOT / "nemo"
if str(NEMO_DIR) not in sys.path:
    sys.path.insert(0, str(NEMO_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common.recording import ActivityRecorder, persist_trials

import brain  # type: ignore  # pylint: disable=import-error


@dataclass
class AssociateConfig:
    seed: int = 0
    n_neurons: int = 5000
    k: int = 200
    p: float = 0.02
    beta: float = 0.05
    stability_rounds: int = 9
    a_to_c_rounds: int = 9
    b_to_c_rounds: int = 9
    overlap_rounds: int = 10
    post_b_rounds: int = 9
    record_dense: bool = True


class AssociateTrial:
    def __init__(self, config: AssociateConfig) -> None:
        self.config = config

    def run(self, *, trial_index: int) -> Dict[str, Any]:
        cfg = self.config
        sim = brain.Brain(
            cfg.p, save_size=False, save_winners=False, seed=cfg.seed + trial_index
        )
        sim.add_stimulus("stimA", cfg.k)
        sim.add_area("A", cfg.n_neurons, cfg.k, cfg.beta)
        sim.add_stimulus("stimB", cfg.k)
        sim.add_area("B", cfg.n_neurons, cfg.k, cfg.beta)
        sim.add_area("C", cfg.n_neurons, cfg.k, cfg.beta)

        recorder = ActivityRecorder(
            {"A": cfg.n_neurons, "B": cfg.n_neurons, "C": cfg.n_neurons},
            record_dense=cfg.record_dense,
        )
        step_counter = 0

        def project(
            stage: str,
            stim_map: MutableMapping[str, List[str]],
            area_map: MutableMapping[str, List[str]],
        ) -> None:
            nonlocal step_counter
            sim.project(stim_map, area_map)
            winners_snapshot = {
                name: sim.area_by_name[name].winners for name in ("A", "B", "C")
            }
            recorder.log(time=step_counter, stage=stage, winners=winners_snapshot)
            step_counter += 1

        project("init_pair", {"stimA": ["A"], "stimB": ["B"]}, {})
        for i in range(cfg.stability_rounds):
            project(
                f"stabilize_{i}",
                {"stimA": ["A"], "stimB": ["B"]},
                {"A": ["A"], "B": ["B"]},
            )

        project("seed_A_to_C", {"stimA": ["A"]}, {"A": ["A", "C"]})
        for i in range(cfg.a_to_c_rounds):
            project(
                f"A_to_C_recur_{i}",
                {"stimA": ["A"]},
                {"A": ["A", "C"], "C": ["C"]},
            )

        project("seed_B_to_C", {"stimB": ["B"]}, {"B": ["B", "C"]})
        for i in range(cfg.b_to_c_rounds):
            project(
                f"B_to_C_recur_{i}",
                {"stimB": ["B"]},
                {"B": ["B", "C"], "C": ["C"]},
            )

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
            "overlap": overlap_score,
            "recorder": recorder,
        }

    def _evaluate_overlap(self, sim: brain.Brain) -> float:
        cfg = self.config
        brain_from_a = copy.deepcopy(sim)
        brain_from_a.project({"stimA": ["A"]}, {})
        brain_from_a.project({}, {"A": ["C"]})
        a_into_c = set(brain_from_a.area_by_name["C"].winners)

        brain_from_b = copy.deepcopy(sim)
        brain_from_b.project({"stimB": ["B"]}, {})
        brain_from_b.project({}, {"B": ["C"]})
        b_into_c = set(brain_from_b.area_by_name["C"].winners)

        if not a_into_c or not b_into_c:
            return 0.0
        return len(a_into_c & b_into_c) / float(cfg.k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record Assembly Calculus associate runs"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON config overriding default parameters",
    )
    parser.add_argument(
        "--trials", type=int, default=1, help="How many independent simulations to run"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/associate_manifold"),
        help="Directory where run folders (config + manifest + npz tensors) are saved",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional suffix for the run directory name",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Skip writing artifacts to disk"
    )
    parser.add_argument(
        "--dump-records",
        action="store_true",
        help="Print recorded winners for the first trial",
    )
    return parser.parse_args()


def load_config(path: Optional[Path]) -> AssociateConfig:
    if path is None:
        return AssociateConfig()
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return AssociateConfig(**data)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runner = AssociateTrial(config)
    trials: List[Dict[str, Any]] = []
    for idx in range(args.trials):
        trial = runner.run(trial_index=idx)
        trials.append(trial)
        print(f"trial={idx} overlap={trial['overlap']:.3f}")

    if args.dump_records and trials:
        preview = trials[0]["recorder"].preview()
        print(json.dumps(preview, indent=2))

    if not args.no_save:
        run_dir = persist_trials(
            trials=trials, config=config, output_root=args.output_root, tag=args.tag
        )
        if run_dir is not None:
            print(f"Saved run artifacts to {run_dir}")


if __name__ == "__main__":
    main()
