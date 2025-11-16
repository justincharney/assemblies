"""Shared helpers for batching experiment runs (parameter sweeps)."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass
class RunSpec:
    command: List[str]
    config_path: Path
    output_root: Path
    tag: str
    env: Optional[Mapping[str, str]] = None


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_config(config_dir: Path, name: str, payload: Mapping[str, Any]) -> Path:
    ensure_parent_dir(config_dir)
    path = config_dir / name
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def run_command(cmd: List[str], *, env: Optional[Mapping[str, str]] = None) -> None:
    subprocess.run(cmd, check=True, env=None if env is None else dict(env))


def format_tag(operation: str, sweep_id: str, seed: int) -> str:
    return f"{operation}_{sweep_id}_seed{seed}"
