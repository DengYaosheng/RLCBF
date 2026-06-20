"""Fast-Planner-style B-spline planning utilities.

The implementation is intentionally dependency-light for ROS Noetic/XTDrone
deployments, but follows the engineering structure used by mature quadrotor
planners: reference fitting, smoothness, obstacle clearance, dynamic
feasibility, and iterative control-point projection are separated from the
controller.
"""

from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Sequence, Tuple

import numpy as np


EPS = 1.0e-9


def clamp_norm(vec: Iterable[float], limit: float) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if limit > 0.0 and norm > limit:
        return arr * (limit / max(norm, EPS))
    return arr


@dataclass
class BsplineOptimizerConfig:
    iterations: int = 8
    step_size: float = 0.08
    fit_weight: float = 1.0
    endpoint_weight: float = 3.0
    smooth_weight: float = 0.18
    obstacle_weight: float = 0.10
    dynamic_weight: float = 0.08
    obstacle_clearance: float = 0.65
    max_ref_speed: float = 1.6
    max_ref_accel: float = 1.2
    max_ref_jerk: float = 0.0
    dynamic_projection_iterations: int = 3

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "BsplineOptimizerConfig":
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


@dataclass
class BsplineOptimizationReport:
    speed_max: float = 0.0
    accel_max: float = 0.0
    jerk_max: float = 0.0
    obstacle_clearance_min: float = float("inf")
    projection_passes: int = 0
    cost_history: List[float] = field(default_factory=list)


class UniformCubicBspline:
    """Uniform cubic B-spline basis and derivative helpers."""

    @staticmethod
    def basis(u: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        u = float(np.clip(u, 0.0, 1.0))
        u2 = u * u
        u3 = u2 * u
        basis = np.array(
            [
                (1.0 - 3.0 * u + 3.0 * u2 - u3) / 6.0,
                (4.0 - 6.0 * u2 + 3.0 * u3) / 6.0,
                (1.0 + 3.0 * u + 3.0 * u2 - 3.0 * u3) / 6.0,
                u3 / 6.0,
            ],
            dtype=float,
        )
        basis_d = np.array(
            [
                -0.5 * (1.0 - u) ** 2,
                1.5 * u2 - 2.0 * u,
                -1.5 * u2 + u + 0.5,
                0.5 * u2,
            ],
            dtype=float,
        )
        basis_dd = np.array([1.0 - u, 3.0 * u - 2.0, -3.0 * u + 1.0, u], dtype=float)
        basis_ddd = np.array([-1.0, 3.0, -3.0, 1.0], dtype=float)
        return basis, basis_d, basis_dd, basis_ddd

    @staticmethod
    def evaluate(control_points: np.ndarray, u: float, knot_dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        basis, basis_d, basis_dd, _ = UniformCubicBspline.basis(u)
        active = np.asarray(control_points, dtype=float).reshape((-1, 3))[:4]
        dt = max(float(knot_dt), 1.0e-3)
        return basis.dot(active), basis_d.dot(active) / dt, basis_dd.dot(active) / (dt * dt)


class FastPlannerBsplineOptimizer:
    """Gradient and projection backend inspired by Fast-Planner/EGO-style stacks."""

    def __init__(self, config: BsplineOptimizerConfig):
        self.config = config
        self.last_report = BsplineOptimizationReport()

    @staticmethod
    def _obstacle_position(obstacle) -> np.ndarray:
        return np.asarray(getattr(obstacle, "position"), dtype=float).reshape(3)

    @staticmethod
    def _obstacle_radius(obstacle) -> float:
        return float(getattr(obstacle, "radius", 0.0))

    def optimize(self, raw_control_points: np.ndarray, knot_dt: float, obstacles: Sequence[object]) -> np.ndarray:
        cfg = self.config
        ctrl = np.asarray(raw_control_points, dtype=float).reshape((-1, 3)).copy()
        raw = ctrl.copy()
        n = ctrl.shape[0]
        if n < 4:
            raise ValueError("cubic B-spline optimization requires at least 4 control points")
        report = BsplineOptimizationReport()
        knot_dt = max(float(knot_dt), 1.0e-3)

        for _ in range(max(int(cfg.iterations), 0)):
            cost, grad = self._cost_and_gradient(ctrl, raw, knot_dt, obstacles)
            report.cost_history.append(cost)
            ctrl -= float(cfg.step_size) * grad
            ctrl = self.project_dynamic_feasibility(ctrl, knot_dt)
            report.projection_passes += max(int(cfg.dynamic_projection_iterations), 0)

        self._fill_report(report, ctrl, knot_dt, obstacles)
        self.last_report = report
        return ctrl

    def _cost_and_gradient(
        self,
        ctrl: np.ndarray,
        raw: np.ndarray,
        knot_dt: float,
        obstacles: Sequence[object],
    ) -> Tuple[float, np.ndarray]:
        cfg = self.config
        n = ctrl.shape[0]
        grad = float(cfg.fit_weight) * (ctrl - raw)
        cost = 0.5 * float(cfg.fit_weight) * float(np.sum((ctrl - raw) ** 2))
        grad[0] += float(cfg.endpoint_weight) * (ctrl[0] - raw[0])
        grad[-1] += float(cfg.endpoint_weight) * (ctrl[-1] - raw[-1])
        cost += 0.5 * float(cfg.endpoint_weight) * (
            float(np.sum((ctrl[0] - raw[0]) ** 2)) + float(np.sum((ctrl[-1] - raw[-1]) ** 2))
        )

        for j in range(1, n - 1):
            second = ctrl[j - 1] - 2.0 * ctrl[j] + ctrl[j + 1]
            cost += 0.5 * float(cfg.smooth_weight) * float(np.dot(second, second))
            smooth_grad = float(cfg.smooth_weight) * second
            grad[j - 1] += smooth_grad
            grad[j] -= 2.0 * smooth_grad
            grad[j + 1] += smooth_grad

        for j in range(n):
            for obstacle in obstacles:
                diff = ctrl[j] - self._obstacle_position(obstacle)
                dist = float(np.linalg.norm(diff))
                safe = self._obstacle_radius(obstacle) + float(cfg.obstacle_clearance)
                if 1.0e-6 < dist < safe:
                    gap = safe - dist
                    cost += 0.5 * float(cfg.obstacle_weight) * gap * gap
                    grad[j] -= float(cfg.obstacle_weight) * gap * diff / max(dist, EPS)

        max_speed = max(float(cfg.max_ref_speed), 1.0e-3)
        max_accel = max(float(cfg.max_ref_accel), 1.0e-3)
        max_jerk = max(float(cfg.max_ref_jerk), 0.0)
        for j in range(n - 1):
            diff = ctrl[j + 1] - ctrl[j]
            norm = float(np.linalg.norm(diff))
            speed = norm / knot_dt
            if speed > max_speed and norm > 1.0e-6:
                excess = speed - max_speed
                cost += 0.5 * float(cfg.dynamic_weight) * excess * excess
                g = float(cfg.dynamic_weight) * excess * diff / (knot_dt * norm)
                grad[j] -= g
                grad[j + 1] += g

        for j in range(1, n - 1):
            second = ctrl[j - 1] - 2.0 * ctrl[j] + ctrl[j + 1]
            norm = float(np.linalg.norm(second))
            accel = norm / (knot_dt * knot_dt)
            if accel > max_accel and norm > 1.0e-6:
                excess = accel - max_accel
                cost += 0.5 * float(cfg.dynamic_weight) * excess * excess
                g = float(cfg.dynamic_weight) * excess * second / (knot_dt * knot_dt * norm)
                grad[j - 1] += g
                grad[j] -= 2.0 * g
                grad[j + 1] += g

        if max_jerk > 0.0:
            for j in range(1, n - 2):
                third = ctrl[j - 1] - 3.0 * ctrl[j] + 3.0 * ctrl[j + 1] - ctrl[j + 2]
                norm = float(np.linalg.norm(third))
                jerk = norm / (knot_dt ** 3)
                if jerk > max_jerk and norm > 1.0e-6:
                    excess = jerk - max_jerk
                    cost += 0.5 * float(cfg.dynamic_weight) * excess * excess
                    g = float(cfg.dynamic_weight) * excess * third / (knot_dt ** 3 * norm)
                    grad[j - 1] += g
                    grad[j] -= 3.0 * g
                    grad[j + 1] += 3.0 * g
                    grad[j + 2] -= g
        return float(cost), grad

    def project_dynamic_feasibility(self, ctrl: np.ndarray, knot_dt: float) -> np.ndarray:
        cfg = self.config
        out = ctrl.copy()
        dt = max(float(knot_dt), 1.0e-3)
        max_step = max(float(cfg.max_ref_speed), 1.0e-3) * dt
        max_second = max(float(cfg.max_ref_accel), 1.0e-3) * dt * dt
        max_third = max(float(cfg.max_ref_jerk), 0.0) * dt ** 3
        n = out.shape[0]
        for _ in range(max(int(cfg.dynamic_projection_iterations), 0)):
            for j in range(n - 1):
                diff = out[j + 1] - out[j]
                norm = float(np.linalg.norm(diff))
                if norm > max_step and norm > 1.0e-6:
                    midpoint = 0.5 * (out[j] + out[j + 1])
                    half = 0.5 * max_step * diff / norm
                    out[j] = midpoint - half
                    out[j + 1] = midpoint + half

            for j in range(1, n - 1):
                second = out[j - 1] - 2.0 * out[j] + out[j + 1]
                norm = float(np.linalg.norm(second))
                if norm > max_second and norm > 1.0e-6:
                    desired = second * (max_second / norm)
                    out[j] = 0.5 * (out[j - 1] + out[j + 1] - desired)

            if max_third > 0.0:
                for j in range(1, n - 2):
                    third = out[j - 1] - 3.0 * out[j] + 3.0 * out[j + 1] - out[j + 2]
                    norm = float(np.linalg.norm(third))
                    if norm > max_third and norm > 1.0e-6:
                        correction = 0.25 * (third - third * (max_third / norm))
                        out[j] += correction
                        out[j + 1] -= correction
        return out

    def _fill_report(self, report: BsplineOptimizationReport, ctrl: np.ndarray, knot_dt: float, obstacles: Sequence[object]) -> None:
        dt = max(float(knot_dt), 1.0e-3)
        if ctrl.shape[0] >= 2:
            speeds = np.linalg.norm(np.diff(ctrl, axis=0), axis=1) / dt
            report.speed_max = float(np.max(speeds)) if speeds.size else 0.0
        if ctrl.shape[0] >= 3:
            accels = np.linalg.norm(ctrl[:-2] - 2.0 * ctrl[1:-1] + ctrl[2:], axis=1) / (dt * dt)
            report.accel_max = float(np.max(accels)) if accels.size else 0.0
        if ctrl.shape[0] >= 4:
            jerks = np.linalg.norm(ctrl[:-3] - 3.0 * ctrl[1:-2] + 3.0 * ctrl[2:-1] - ctrl[3:], axis=1) / (dt ** 3)
            report.jerk_max = float(np.max(jerks)) if jerks.size else 0.0
        clearances = []
        for point in ctrl:
            for obstacle in obstacles:
                clearances.append(float(np.linalg.norm(point - self._obstacle_position(obstacle)) - self._obstacle_radius(obstacle)))
        report.obstacle_clearance_min = min(clearances) if clearances else float("inf")
