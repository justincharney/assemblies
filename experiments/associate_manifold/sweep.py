"""Batch launcher for associate recording sweeps."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import sys
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common.sweeps import write_config

ASSOCIATE_SCRIPT = (
    REPO_ROOT / "experiments" / "associate_manifold" / "associate_recorder.py"
)
OPERATION_NAME = "associate"


def load_sweep(path: Path) -> Dict[str, any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_command(config: Path, trials: int, output_root: Path, tag: str) -> List[str]:
    return [
        "python",
        str(ASSOCIATE_SCRIPT),
        "--config",
        str(config),
        "--trials",
        str(trials),
        "--output-root",
        str(output_root),
        "--tag",
        tag,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run associate sweeps")
    parser.add_argument(
        "--sweep-config", type=Path, required=True, help="JSON describing configs/seeds"
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("runs/associate_manifold")
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    spec = load_sweep(args.sweep_config)
    sweep_id = spec.get("sweep_id", args.sweep_config.stem)
    base_config = spec.get("base_config", {})
    variants = spec.get("variants", [])
    seeds = spec.get("seeds", [0])

    tmp_dir = REPO_ROOT / "experiments" / "associate_manifold" / "configs" / "generated"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for variant_idx, variant in enumerate(variants):
        variant_cfg = base_config.copy()
        variant_cfg.update(variant)
        config_name = f"{sweep_id}_var{variant_idx:02d}.json"
        config_path = write_config(tmp_dir, config_name, variant_cfg)

        for seed in seeds:
            tag = f"{OPERATION_NAME}_{sweep_id}_var{variant_idx:02d}_seed{seed}"
            cmd = build_command(
                config_path,
                args.trials,
                args.output_root / OPERATION_NAME / sweep_id / f"seed{seed}",
                tag,
            )
            if args.dry_run:
                print(" ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
