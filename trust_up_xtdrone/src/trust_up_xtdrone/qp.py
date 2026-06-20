"""Small deterministic QP solver for TRUST-UP safety filtering.

The QP solved here is the projection

    min_v 0.5 ||v - v_nom||^2
    s.t.  A v >= b

for v in R^2 or R^3.  Enumerating active sets is faster and easier to audit
than pulling in a large optimization dependency on flight hardware.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Optional

import numpy as np


@dataclass
class ProjectionResult:
    value: np.ndarray
    feasible: bool
    active: tuple
    max_violation: float
    objective: float


def _as_2d(a: np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if arr.size == 0:
        return np.zeros((0, dim), dtype=float)
    return arr.reshape((-1, dim))


def project_to_halfspaces(
    nominal: Iterable[float],
    a_mat: Optional[Iterable[Iterable[float]]] = None,
    b_vec: Optional[Iterable[float]] = None,
    *,
    tol: float = 1.0e-8,
    max_iter_fallback: int = 50,
) -> ProjectionResult:
    """Project ``nominal`` onto linear halfspaces ``A v >= b``.

    The feasible active-set optimum has at most ``dim`` independent active
    constraints.  If numerical degeneracy prevents an exact active-set answer,
    a sequential projection fallback is used and marked infeasible when any
    residual violation remains.
    """

    v0 = np.asarray(nominal, dtype=float).reshape(-1)
    dim = int(v0.size)
    if dim == 0:
        raise ValueError("nominal vector must be non-empty")

    a = _as_2d(np.asarray(a_mat if a_mat is not None else [], dtype=float), dim)
    if b_vec is None:
        b = np.zeros((a.shape[0],), dtype=float)
    else:
        b = np.asarray(b_vec, dtype=float).reshape(-1)
    if a.shape[0] != b.shape[0]:
        raise ValueError("A and b have incompatible row counts")

    if a.shape[0] == 0:
        return ProjectionResult(v0.copy(), True, tuple(), 0.0, 0.0)

    residual = b - a.dot(v0)
    if float(np.max(residual)) <= tol:
        return ProjectionResult(v0.copy(), True, tuple(), float(np.max(residual)), 0.0)

    best = None
    row_ids = range(a.shape[0])
    for active_size in range(1, min(dim, a.shape[0]) + 1):
        for active in combinations(row_ids, active_size):
            a_act = a[list(active), :]
            b_act = b[list(active)]
            gram = a_act.dot(a_act.T)
            correction = a_act.T.dot(np.linalg.pinv(gram).dot(b_act - a_act.dot(v0)))
            candidate = v0 + correction
            violation = b - a.dot(candidate)
            max_violation = float(np.max(violation))
            if max_violation <= 5.0 * tol:
                obj = 0.5 * float(np.dot(candidate - v0, candidate - v0))
                if best is None or obj < best.objective:
                    best = ProjectionResult(candidate, True, tuple(active), max_violation, obj)

    if best is not None:
        return best

    candidate = v0.copy()
    for _ in range(max_iter_fallback):
        improved = False
        for row, bound in zip(a, b):
            norm_sq = float(np.dot(row, row))
            if norm_sq <= tol:
                continue
            gap = float(bound - np.dot(row, candidate))
            if gap > tol:
                candidate = candidate + (gap / norm_sq) * row
                improved = True
        if not improved:
            break

    violation = b - a.dot(candidate)
    max_violation = float(np.max(violation))
    obj = 0.5 * float(np.dot(candidate - v0, candidate - v0))
    return ProjectionResult(candidate, max_violation <= 1.0e-5, tuple(), max_violation, obj)
