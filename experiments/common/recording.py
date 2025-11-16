"""Reusable recording utilities for Assembly Calculus experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np


class ActivityRecorder:
    """Collects per-step winners and optional dense population activity."""

    def __init__(self, area_sizes: Mapping[str, int], *, record_dense: bool) -> None:
        self.area_sizes = dict(area_sizes)
        self.record_dense = record_dense
        self._area_buffers: Dict[str, Dict[str, List[Any]]] = {
            name: {"time": [], "stage": [], "indices": [], "dense": []}
            for name in self.area_sizes
        }
        self._preview: List[Dict[str, Any]] = []

    def log(self, *, time: int, stage: str, winners: Mapping[str, Iterable[int]]) -> None:
        """Append a snapshot for each tracked area."""

        for area, idx_iter in winners.items():
            idx_arr = np.asarray(list(idx_iter), dtype=np.int32)
            buf = self._area_buffers[area]
            buf["time"].append(time)
            buf["stage"].append(stage)
            buf["indices"].append(idx_arr)
            if self.record_dense:
                dense_vec = np.zeros(self.area_sizes[area], dtype=np.uint8)
                if idx_arr.size:
                    dense_vec[idx_arr] = 1
                buf["dense"].append(dense_vec)
            self._preview.append(
                {
                    "time": time,
                    "stage": stage,
                    "area": area,
                    "indices": idx_arr.tolist(),
                }
            )

    def to_npz_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for area, buf in self._area_buffers.items():
            payload[f"{area}_times"] = np.asarray(buf["time"], dtype=np.int32)
            payload[f"{area}_stages"] = np.asarray(buf["stage"], dtype=object)
            payload[f"{area}_indices"] = np.asarray(buf["indices"], dtype=object)
            if buf["dense"]:
                payload[f"{area}_dense"] = np.stack(buf["dense"], axis=0)
        return payload

    def preview(self) -> List[Dict[str, Any]]:
        return self._preview


def _ensure_mapping(config: Any) -> Dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    raise TypeError("config must be a dataclass or mapping")


def persist_trials(
    *,
    trials: List[Dict[str, Any]],
    config: Any,
    output_root: Path,
    tag: Optional[str] = None,
) -> Optional[Path]:
    """Write config, manifest, and per-trial tensors to disk."""

    if not trials:
        return None

    output_root.mkdir(parents=True, exist_ok=True)
    base_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    if tag:
        base_name = f"{base_name}_{tag}"
    run_dir = output_root / base_name
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{base_name}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=False, exist_ok=False)

    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(_ensure_mapping(config), handle, indent=2)

    manifest: List[Dict[str, Any]] = []
    for entry in trials:
        trial_idx = int(entry.get("trial_index", len(manifest)))
        recorder = entry["recorder"]
        if not isinstance(recorder, ActivityRecorder):
            raise TypeError("trial['recorder'] must be an ActivityRecorder")
        payload = recorder.to_npz_payload()
        record_file = run_dir / f"trial_{trial_idx:03d}.npz"
        np.savez_compressed(record_file, **payload)
        manifest_entry = {k: v for k, v in entry.items() if k != "recorder"}
        manifest_entry["record_file"] = record_file.name
        manifest.append(manifest_entry)

    with (run_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return run_dir
