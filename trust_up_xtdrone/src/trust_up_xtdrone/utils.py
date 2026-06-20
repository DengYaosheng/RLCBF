"""Numerical helpers shared by the TRUST-UP core modules."""

from dataclasses import fields
from typing import Any, Dict, Iterable, Mapping

import numpy as np


EPS = 1.0e-9


def as_vector3(value: Iterable[float]) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(3)


def clamp_norm(vec: Iterable[float], limit: float) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if 0.0 < limit < norm:
        return arr * (float(limit) / max(norm, EPS))
    return arr


def dataclass_subset(data_cls: Any, data: Mapping[str, Any]) -> Dict[str, Any]:
    names = {item.name for item in fields(data_cls)}
    return {key: data[key] for key in names if key in data}
