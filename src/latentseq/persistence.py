"""Persistence primitives support exact reusable-object restoration without owning RNG state.

LatentSeq pretrained directories use human-readable JSON manifests plus NumPy NPZ arrays.  The
helpers here only move static component state across the filesystem; random streams and in-progress
sample trajectories are deliberately absent from this persistence layer.
"""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


# helpers


def ensure_directory(path: str | os.PathLike[str]) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def numpy_to_tensor(
    array: np.ndarray,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    tensor = torch.from_numpy(array)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.to(device=device)
