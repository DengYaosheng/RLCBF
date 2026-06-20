"""Structured CBF-QP safety-filter interface.

This module keeps the dependency-light active-set solver used on XTDrone, but
wraps it in the same engineering shape used by mature safety-control libraries:
named constraints, projection reports, and solver-swappable filter objects.
"""

from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .qp import ProjectionResult, project_to_halfspaces


@dataclass
class LinearSafetyConstraint:
    name: str
    row: np.ndarray
    bound: float
    kind: str = "cbf"
    metadata: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def make(
        cls,
        name: str,
        row: Iterable[float],
        bound: float,
        *,
        kind: str = "cbf",
        metadata: Optional[Mapping[str, object]] = None,
    ) -> "LinearSafetyConstraint":
        return cls(
            name=name,
            row=np.asarray(row, dtype=float).reshape(-1),
            bound=float(bound),
            kind=kind,
            metadata=metadata or {},
        )


@dataclass
class SafetyFilterResult(ProjectionResult):
    solver: str
    constraints: List[LinearSafetyConstraint]

    @classmethod
    def from_projection(
        cls,
        projection: ProjectionResult,
        constraints: Sequence[LinearSafetyConstraint],
        solver: str,
    ) -> "SafetyFilterResult":
        return cls(
            value=projection.value,
            feasible=projection.feasible,
            active=projection.active,
            max_violation=projection.max_violation,
            objective=projection.objective,
            solver=solver,
            constraints=list(constraints),
        )

    def active_names(self) -> List[str]:
        names = []
        for idx in self.active:
            if 0 <= int(idx) < len(self.constraints):
                names.append(self.constraints[int(idx)].name)
        return names

    def max_violation_by_kind(self) -> Mapping[str, float]:
        if not self.constraints:
            return {}
        rows = np.vstack([c.row for c in self.constraints])
        bounds = np.asarray([c.bound for c in self.constraints], dtype=float)
        residual = bounds - rows.dot(self.value)
        out = {}
        for constraint, value in zip(self.constraints, residual):
            out[constraint.kind] = max(float(out.get(constraint.kind, 0.0)), float(value))
        return out


class CbfQpSafetyFilter:
    """Projection-based CBF-QP filter with a swappable solver boundary."""

    def __init__(self, name: str, *, solver: str = "active_set_projection", tolerance: float = 1.0e-8):
        self.name = str(name)
        self.solver = str(solver)
        self.tolerance = float(tolerance)
        self.last_result: Optional[SafetyFilterResult] = None

    def project(
        self,
        nominal: Iterable[float],
        constraints: Sequence[LinearSafetyConstraint],
    ) -> SafetyFilterResult:
        nominal_arr = np.asarray(nominal, dtype=float).reshape(-1)
        clean_constraints = []
        for constraint in constraints:
            row = np.asarray(constraint.row, dtype=float).reshape(-1)
            if row.shape[0] != nominal_arr.shape[0]:
                raise ValueError("constraint %s has dimension %d, expected %d" % (constraint.name, row.shape[0], nominal_arr.shape[0]))
            if float(np.linalg.norm(row)) <= self.tolerance:
                continue
            clean_constraints.append(
                LinearSafetyConstraint(
                    constraint.name,
                    row,
                    float(constraint.bound),
                    kind=constraint.kind,
                    metadata=constraint.metadata,
                )
            )
        rows = [c.row for c in clean_constraints]
        bounds = [c.bound for c in clean_constraints]
        projection = project_to_halfspaces(nominal_arr, rows, bounds, tol=self.tolerance)
        self.last_result = SafetyFilterResult.from_projection(projection, clean_constraints, self.solver)
        return self.last_result
