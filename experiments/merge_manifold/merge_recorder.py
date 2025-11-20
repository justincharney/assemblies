"""Merge operation recorder for the Assembly Calculus simulator.

This script mirrors the merge protocol implemented in `nemo/simulations.py`
but logs every projection step so downstream manifold or temporal analyses
can operate on the population activity in areas A, B, and C. Each run emits
compressed `.npz` tensors (per trial) alongside a manifest describing the
merge-quality metrics that were measured.
"""

from __future__ import annotations

import argparse
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
class MergeConfig:
    seed: int = 0
    n_neurons: int = 5000
    k: int = 200
    p: float = 0.02
    beta: float = 0.05
    a_stabilize_rounds: int = 6
    b_stabilize_rounds: int = 6
    merge_rounds: int = 30
    record_dense: bool = True


class MergeTrial:
    def __init__(self, config: MergeConfig) -> None:
        self.config = config

    def run(self, *, trial_index: int) -> Dict[str, Any]:
        cfg = self.config
        sim = brain.Brain(
            cfg.p, save_size=False, save_winners=False, seed=cfg.seed + trial_index
        )
        sim.add_stimulus("stimA", cfg.k)
        sim.add_stimulus("stimB", cfg.k)
        sim.add_area("A", cfg.n_neurons, cfg.k, cfg.beta)
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

        # Create A and B assemblies separately, mirroring merge_sim setup.
        project("seed_A", {"stimA": ["A"]}, {})
        for i in range(cfg.a_stabilize_rounds):
            project(
                f"A_stabilize_{i}",
                {"stimA": ["A"]},
                {"A": ["A"]},
            )

        project("seed_B", {"stimB": ["B"]}, {})
        for i in range(cfg.b_stabilize_rounds):
            project(
                f"B_stabilize_{i}",
                {"stimB": ["B"]},
                {"B": ["B"]},
            )

        # Joint projections to C and recurrent merge iterations.
        project(
            "joint_seed",
            {"stimA": ["A"], "stimB": ["B"]},
            {"A": ["A", "C"], "B": ["B", "C"]},
        )
        project(
            "merge_start",
            {"stimA": ["A"], "stimB": ["B"]},
            {"A": ["A", "C"], "B": ["B", "C"], "C": ["C", "A", "B"]},
        )
        for i in range(max(cfg.merge_rounds - 1, 0)):
            project(
                f"merge_iter_{i}",
                {"stimA": ["A"], "stimB": ["B"]},
                {"A": ["A", "C"], "B": ["B", "C"], "C": ["C", "A", "B"]},
            )

        metrics = self._evaluate_metrics(sim)
        return {
            "trial_index": trial_index,
            **metrics,
            "recorder": recorder,
        }

    def _evaluate_metrics(self, sim: brain.Brain) -> Dict[str, float]:
        cfg = self.config
        winners_a = set(sim.area_by_name["A"].winners)
        winners_b = set(sim.area_by_name["B"].winners)
        winners_c = set(sim.area_by_name["C"].winners)
        denom = max(1, cfg.k)
        return {
            "c_vs_a": len(winners_c & winners_a) / float(denom),
            "c_vs_b": len(winners_c & winners_b) / float(denom),
            "a_vs_b": len(winners_a & winners_b) / float(denom),
            "c_density": len(winners_c) / float(denom),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record Assembly Calculus merge runs")
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
        default=Path("runs/merge_manifold"),
        help="Where run folders (config + manifest + tensors) are saved",
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


def load_config(path: Optional[Path]) -> MergeConfig:
    if path is None:
        return MergeConfig()
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return MergeConfig(**data)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runner = MergeTrial(config)
    trials: List[Dict[str, Any]] = []
    for idx in range(args.trials):
        trial = runner.run(trial_index=idx)
        trials.append(trial)
        print(
            f"trial={idx} c_vs_a={trial['c_vs_a']:.3f} "
            f"c_vs_b={trial['c_vs_b']:.3f} a_vs_b={trial['a_vs_b']:.3f}"
        )

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
