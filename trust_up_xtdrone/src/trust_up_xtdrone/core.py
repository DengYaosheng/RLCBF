"""TRUST-UP core dynamics, constraints, and target trajectories."""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .adaptive import AdaptiveRobustnessEstimator
from .planning import BsplineOptimizerConfig, FastPlannerBsplineOptimizer, UniformCubicBspline
from .safety_filter import CbfQpSafetyFilter, LinearSafetyConstraint
from .utils import EPS, clamp_norm, dataclass_subset


@dataclass
class AgentState:
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radius: float = 0.0
    stamp: float = 0.0
    name: str = ""

    @classmethod
    def from_xyz(
        cls,
        xyz: Sequence[float],
        vel: Optional[Sequence[float]] = None,
        acc: Optional[Sequence[float]] = None,
        radius: float = 0.0,
        stamp: float = 0.0,
        name: str = "",
    ) -> "AgentState":
        return cls(
            position=np.asarray(xyz, dtype=float).reshape(3),
            velocity=np.asarray(vel if vel is not None else [0.0, 0.0, 0.0], dtype=float).reshape(3),
            acceleration=np.asarray(acc if acc is not None else [0.0, 0.0, 0.0], dtype=float).reshape(3),
            radius=float(radius),
            stamp=float(stamp),
            name=name,
        )


def yaw_rate_to_target(pos: np.ndarray, target: np.ndarray, current_yaw: float = 0.0, max_yaw_rate: float = 0.8) -> float:
    delta = target[:2] - pos[:2]
    if float(np.linalg.norm(delta)) < 1.0e-4:
        return 0.0
    desired = float(np.arctan2(delta[1], delta[0]))
    err = (desired - current_yaw + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.clip(1.5 * err, -max_yaw_rate, max_yaw_rate))


@dataclass
class SafetyParameters:
    collision_radius: float = 0.5
    sensing_radius: float = 1.0
    desired_range: float = 0.75
    use_object_radius: bool = False
    hocbf_k0: float = 2.0
    hocbf_k1: float = 2.4
    sensing_k0: float = 1.6
    sensing_k1: float = 2.0
    robustness_margin: float = 0.05
    input_alpha: float = 2.0
    kappa_c: float = 0.45
    kappa_ell: float = 0.75
    kappa_epsilon: float = 0.04
    velocity_barrier_gamma: float = 4.0
    velocity_barrier_margin: float = 0.02
    tracking_inner_margin: float = 0.0
    tracking_outer_margin: float = 0.0
    obstacle_enforcement_margin: float = 0.0

    def radius_sum(self, pursuer: AgentState, other: AgentState) -> float:
        if not self.use_object_radius:
            return 0.0
        return max(float(pursuer.radius), 0.0) + max(float(other.radius), 0.0)

    def collision_bound(self, pursuer: AgentState, other: AgentState) -> float:
        return float(self.collision_radius) + self.radius_sum(pursuer, other)

    def sensing_bound(self, pursuer: AgentState, target: AgentState) -> float:
        return float(self.sensing_radius) + self.radius_sum(pursuer, target)

    def desired_bound(self, pursuer: AgentState, target: AgentState) -> float:
        return float(self.desired_range) + self.radius_sum(pursuer, target)

    def enforced_collision_bound(self, pursuer: AgentState, other: AgentState, *, obstacle: bool = False) -> float:
        margin = float(self.obstacle_enforcement_margin) if obstacle else float(self.tracking_inner_margin)
        return self.collision_bound(pursuer, other) + max(margin, 0.0)

    def enforced_sensing_bound(self, pursuer: AgentState, target: AgentState) -> float:
        paper_outer = self.sensing_bound(pursuer, target)
        paper_inner = self.collision_bound(pursuer, target)
        inner_guard = self.enforced_collision_bound(pursuer, target)
        outer_guard = paper_outer - max(float(self.tracking_outer_margin), 0.0)
        return max(outer_guard, inner_guard + 0.05, paper_inner + 0.05)


@dataclass
class VehicleLimits:
    max_speed: float = 1.2
    max_vertical_speed: float = 0.6
    max_accel: float = 0.9
    max_jerk: float = 0.0
    max_yaw_rate: float = 0.8
    takeoff_altitude: float = 2.0


@dataclass
class PolicyParameters:
    kp_range: float = 1.4
    kd_relative: float = 0.8
    target_velocity_feedforward: float = 0.35
    max_nominal_accel: float = 0.9


@dataclass
class TrajectorySmoothingParameters(BsplineOptimizerConfig):
    enabled: bool = False
    knot_dt: float = 0.8
    horizon_control_points: int = 8
    limit_integrated_target: bool = True
    target_pair_clearance: float = 1.0
    target_pair_activation_margin: float = 1.0
    target_pair_gain: float = 2.5
    target_pair_hard_gain: float = 6.0
    target_pair_damping: float = 1.4


class NominalPolicy:
    """Deterministic SAC-compatible nominal policy.

    A TorchScript SAC policy can be wrapped outside this class.  The default
    policy preserves the paper reward intent: keep the target distance inside
    [r_i, R_i] before the CBF-QP filter enforces hard constraints.
    """

    def __init__(self, safety: SafetyParameters, limits: VehicleLimits, params: PolicyParameters):
        self.safety = safety
        self.limits = limits
        self.params = params

    def __call__(self, pursuer: AgentState, target: AgentState, command_velocity: np.ndarray) -> np.ndarray:
        zeta = pursuer.position - target.position
        distance = float(np.linalg.norm(zeta))
        direction = zeta / max(distance, EPS)
        desired = self.safety.desired_bound(pursuer, target)
        radial_accel = -self.params.kp_range * (distance - desired) * direction
        rel_vel = pursuer.velocity - target.velocity
        damping = -self.params.kd_relative * rel_vel
        feedforward = self.params.target_velocity_feedforward * target.acceleration
        accel = radial_accel + damping + feedforward
        return clamp_norm(accel, min(self.params.max_nominal_accel, self.limits.max_accel))


class TrustUpController:
    """CBF-QP safety filter with paper-style target pursuit constraints."""

    def __init__(
        self,
        safety: Optional[SafetyParameters] = None,
        limits: Optional[VehicleLimits] = None,
        policy: Optional[NominalPolicy] = None,
        adaptive: Optional[AdaptiveRobustnessEstimator] = None,
    ):
        self.safety = safety or SafetyParameters()
        self.limits = limits or VehicleLimits()
        self.policy = policy or NominalPolicy(self.safety, self.limits, PolicyParameters())
        self.adaptive = adaptive or AdaptiveRobustnessEstimator()
        self.command_velocity = np.zeros(3)
        self.previous_safe_accel = np.zeros(3)
        self.accel_filter = CbfQpSafetyFilter("hocbf_accel")
        self.velocity_filter = CbfQpSafetyFilter("velocity_guard")

    def reset(self, initial_velocity: Optional[Sequence[float]] = None) -> None:
        self.command_velocity = np.asarray(initial_velocity if initial_velocity is not None else [0.0, 0.0, 0.0], dtype=float)
        self.previous_safe_accel = np.zeros(3)
        self.adaptive.reset(initial_velocity)

    def _kappa(self, zeta: np.ndarray, zeta_dot: np.ndarray) -> Tuple[float, float]:
        s = float(np.dot(zeta, zeta))
        ell2 = self.safety.kappa_ell ** 2
        denom = (s - ell2) ** 2 + self.safety.kappa_epsilon
        kappa = self.safety.kappa_c + 1.0 / max(denom, EPS)
        s_dot = 2.0 * float(np.dot(zeta, zeta_dot))
        kappa_dot = -(2.0 * (s - ell2) * s_dot) / max(denom * denom, EPS)
        return float(kappa), float(kappa_dot)

    def _append_collision_constraint(
        self,
        rows: List[np.ndarray],
        bounds: List[float],
        pursuer: AgentState,
        obstacle: AgentState,
        robustness_margin: Optional[float] = None,
    ) -> Dict[str, float]:
        is_target = obstacle.name.startswith("target") or obstacle.name.startswith("trust_target")
        radius = self.safety.enforced_collision_bound(pursuer, obstacle, obstacle=not is_target)
        paper_radius = self.safety.collision_bound(pursuer, obstacle)
        r = pursuer.position - obstacle.position
        r_dot = pursuer.velocity - obstacle.velocity
        h = float(np.dot(r, r) - radius * radius)
        h_dot = 2.0 * float(np.dot(r, r_dot))
        a = 2.0 * r
        b = (
            2.0 * float(np.dot(r, obstacle.acceleration))
            - 2.0 * float(np.dot(r_dot, r_dot))
            - self.safety.hocbf_k1 * h_dot
            - self.safety.hocbf_k0 * h
            + float(self.safety.robustness_margin if robustness_margin is None else robustness_margin)
        )
        if float(np.linalg.norm(a)) > 1.0e-6:
            rows.append(a)
            bounds.append(float(b))
        return {"h": h, "h_dot": h_dot, "distance": float(np.linalg.norm(r)), "radius": radius, "paper_radius": paper_radius}

    def _append_sensing_constraint(
        self,
        rows: List[np.ndarray],
        bounds: List[float],
        pursuer: AgentState,
        target: AgentState,
        robustness_margin: Optional[float] = None,
    ) -> Dict[str, float]:
        radius = self.safety.enforced_sensing_bound(pursuer, target)
        paper_radius = self.safety.sensing_bound(pursuer, target)
        zeta = pursuer.position - target.position
        zeta_dot = pursuer.velocity - target.velocity
        h = float(radius * radius - np.dot(zeta, zeta))
        h_dot = -2.0 * float(np.dot(zeta, zeta_dot))
        a = -2.0 * zeta
        b = (
            float(self.safety.robustness_margin if robustness_margin is None else robustness_margin)
            - 2.0 * float(np.dot(zeta, target.acceleration))
            + 2.0 * float(np.dot(zeta_dot, zeta_dot))
            - self.safety.sensing_k1 * h_dot
            - self.safety.sensing_k0 * h
        )
        if float(np.linalg.norm(a)) > 1.0e-6:
            rows.append(a)
            bounds.append(float(b))
        return {"h": h, "h_dot": h_dot, "distance": float(np.linalg.norm(zeta)), "radius": radius, "paper_radius": paper_radius}

    def _append_input_constraint(
        self,
        rows: List[np.ndarray],
        bounds: List[float],
        pursuer: AgentState,
        target: AgentState,
        robustness_margin: Optional[float] = None,
    ) -> Dict[str, float]:
        zeta = pursuer.position - target.position
        zeta_dot = self.command_velocity - target.velocity
        kappa, kappa_dot = self._kappa(zeta, zeta_dot)
        speed_bound = min(kappa, self.limits.max_speed)
        h = float(speed_bound * speed_bound - np.dot(self.command_velocity, self.command_velocity))
        a = -2.0 * self.command_velocity
        b = (
            float(self.safety.robustness_margin if robustness_margin is None else robustness_margin)
            - 2.0 * speed_bound * kappa_dot
            - self.safety.input_alpha * h
        )
        if float(np.linalg.norm(a)) > 1.0e-6:
            rows.append(a)
            bounds.append(float(b))
        return {"h": h, "kappa": kappa, "speed_bound": speed_bound}

    def _velocity_guard(
        self,
        command: np.ndarray,
        pursuer: AgentState,
        target: AgentState,
        obstacles: Sequence[AgentState],
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        rows: List[np.ndarray] = []
        bounds: List[float] = []
        gamma = float(self.safety.velocity_barrier_gamma)
        margin = float(self.safety.velocity_barrier_margin)

        zeta = pursuer.position - target.position
        sensing_radius = self.safety.enforced_sensing_bound(pursuer, target)
        h_s = float(sensing_radius ** 2 - np.dot(zeta, zeta))
        rows.append(-2.0 * zeta)
        bounds.append(float(-2.0 * np.dot(zeta, target.velocity) - gamma * h_s + margin))

        collision_distances = []
        for obstacle in obstacles:
            is_target = obstacle.name.startswith("target") or obstacle.name.startswith("trust_target")
            radius = self.safety.enforced_collision_bound(pursuer, obstacle, obstacle=not is_target)
            r = pursuer.position - obstacle.position
            h_c = float(np.dot(r, r) - radius * radius)
            if float(np.linalg.norm(r)) > 1.0e-6:
                rows.append(2.0 * r)
                bounds.append(float(2.0 * np.dot(r, obstacle.velocity) - gamma * h_c + margin))
            collision_distances.append(float(np.linalg.norm(r)))

        constraints = [
            LinearSafetyConstraint.make("velocity_guard_%02d" % idx, row, bound, kind="velocity_guard")
            for idx, (row, bound) in enumerate(zip(rows, bounds))
        ]
        guarded = self.velocity_filter.project(command, constraints)
        return guarded.value, {
            "velocity_guard_feasible": guarded.feasible,
            "velocity_guard_active": guarded.active_names(),
            "velocity_guard_max_violation": guarded.max_violation,
            "velocity_guard_violation_by_kind": dict(guarded.max_violation_by_kind()),
            "min_collision_distance": min(collision_distances) if collision_distances else None,
        }

    def guard_velocity_command(
        self,
        command: Sequence[float],
        pursuer: AgentState,
        target: AgentState,
        obstacles: Sequence[AgentState],
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        guarded_velocity, guard_diag = self._velocity_guard(np.asarray(command, dtype=float).reshape(3), pursuer, target, obstacles)
        guarded_velocity = clamp_norm(guarded_velocity, self.limits.max_speed)
        if abs(guarded_velocity[2]) > self.limits.max_vertical_speed:
            guarded_velocity[2] = float(np.sign(guarded_velocity[2]) * self.limits.max_vertical_speed)
        return guarded_velocity, guard_diag

    def step(
        self,
        pursuer: AgentState,
        target: AgentState,
        obstacles: Sequence[AgentState],
        dt: float,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        dt = max(float(dt), 1.0e-3)
        nominal = self.policy(pursuer, target, self.command_velocity)
        adaptive_info = self.adaptive.update(pursuer, target, obstacles, self.previous_safe_accel, dt, self.safety)
        nominal = self.adaptive.compensate(nominal)
        if self.limits.max_jerk > 0.0:
            nominal = self.previous_safe_accel + clamp_norm(nominal - self.previous_safe_accel, self.limits.max_jerk * dt)
        adaptive_margin = float(self.safety.robustness_margin) + float(adaptive_info.margin_scale)
        rows: List[np.ndarray] = []
        bounds: List[float] = []
        barrier_info: Dict[str, object] = {"collisions": []}

        barrier_info["sensing"] = self._append_sensing_constraint(rows, bounds, pursuer, target, adaptive_margin)
        barrier_info["target_collision"] = self._append_collision_constraint(rows, bounds, pursuer, target, adaptive_margin)
        for obstacle in obstacles:
            info = self._append_collision_constraint(rows, bounds, pursuer, obstacle, adaptive_margin)
            info["name"] = obstacle.name
            barrier_info["collisions"].append(info)
        barrier_info["input"] = self._append_input_constraint(rows, bounds, pursuer, target, adaptive_margin)

        constraints = [
            LinearSafetyConstraint.make("hocbf_%02d" % idx, row, bound, kind="hocbf")
            for idx, (row, bound) in enumerate(zip(rows, bounds))
        ]
        result = self.accel_filter.project(nominal, constraints)
        safe_accel = clamp_norm(result.value, self.limits.max_accel)
        self.previous_safe_accel = safe_accel.copy()
        self.command_velocity = self.command_velocity + safe_accel * dt
        self.command_velocity, guard_diag = self.guard_velocity_command(self.command_velocity, pursuer, target, [target] + list(obstacles))

        diag: Dict[str, object] = {
            "nominal_accel": nominal,
            "safe_accel": safe_accel,
            "command_velocity": self.command_velocity.copy(),
            "adaptive": adaptive_info.as_dict(),
            "adaptive_robustness_margin": adaptive_margin,
            "qp_feasible": result.feasible,
            "qp_active": result.active,
            "qp_active_names": result.active_names(),
            "qp_max_violation": result.max_violation,
            "qp_violation_by_kind": dict(result.max_violation_by_kind()),
            **guard_diag,
            "barriers": barrier_info,
        }
        return self.command_velocity.copy(), diag


class TargetTrajectory:
    """Paper target dynamics with circular and figure-8 references."""

    def __init__(
        self,
        scenario: str,
        obstacles: Sequence[AgentState],
        *,
        offset: Sequence[float] = (0.0, 0.0, 0.0),
        potential_gain: float = 1.0,
        potential_clip: float = 1.5,
        use_dimension_consistent_reference_velocity: bool = True,
        reference_scale: float = 1.0,
        angular_rate: float = 0.1,
        smoothing: Optional[Mapping[str, object]] = None,
    ):
        self.scenario = scenario
        self.obstacles = list(obstacles)
        self.offset = np.asarray(offset, dtype=float).reshape(3)
        self.potential_gain = float(potential_gain)
        self.potential_clip = float(potential_clip)
        self.use_dimension_consistent_reference_velocity = bool(use_dimension_consistent_reference_velocity)
        self.reference_scale = float(reference_scale)
        self.angular_rate = float(angular_rate)
        smoothing = smoothing or {}
        self.smoothing = TrajectorySmoothingParameters(**dataclass_subset(TrajectorySmoothingParameters, smoothing))
        self._bspline_optimizer = FastPlannerBsplineOptimizer(
            BsplineOptimizerConfig.from_mapping(self.smoothing.__dict__)
        )

    def _raw_reference(self, target_index: int, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        w = self.angular_rate
        scale = self.reference_scale
        if self.scenario == "figure8":
            if target_index == 0:
                pos = scale * np.array([5.0 * np.sin(w * t), 5.0 * np.sin(2.0 * w * t), 3.0])
                vel = scale * np.array([5.0 * w * np.cos(w * t), 10.0 * w * np.cos(2.0 * w * t), 0.0])
                acc = scale * np.array([-5.0 * w * w * np.sin(w * t), -20.0 * w * w * np.sin(2.0 * w * t), 0.0])
            else:
                pos = scale * np.array([5.0 * np.sin(w * t), 3.0, 5.0 * np.sin(2.0 * w * t)])
                vel = scale * np.array([5.0 * w * np.cos(w * t), 0.0, 10.0 * w * np.cos(2.0 * w * t)])
                acc = scale * np.array([-5.0 * w * w * np.sin(w * t), 0.0, -20.0 * w * w * np.sin(2.0 * w * t)])
        else:
            if target_index == 0:
                pos = scale * np.array([5.0 * np.sin(w * t), 5.0 * np.cos(w * t), 0.0])
                vel = scale * np.array([5.0 * w * np.cos(w * t), -5.0 * w * np.sin(w * t), 0.0])
                acc = scale * np.array([-5.0 * w * w * np.sin(w * t), -5.0 * w * w * np.cos(w * t), 0.0])
            else:
                pos = scale * np.array([5.0 * np.sin(w * t), 0.0, 5.0 * np.cos(w * t)])
                vel = scale * np.array([5.0 * w * np.cos(w * t), 0.0, -5.0 * w * np.sin(w * t)])
                acc = scale * np.array([-5.0 * w * w * np.sin(w * t), 0.0, -5.0 * w * w * np.cos(w * t)])
        return pos + self.offset, vel, acc

    @staticmethod
    def _bspline_basis(u: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        basis, basis_d, basis_dd, _ = UniformCubicBspline.basis(u)
        return basis, basis_d, basis_dd

    def _bspline_control_points(self, target_index: int, t: float) -> Tuple[np.ndarray, float]:
        params = self.smoothing
        knot_dt = max(float(params.knot_dt), 1.0e-3)
        n = max(int(params.horizon_control_points), 4)
        segment = math.floor(float(t) / knot_dt)
        control_times = [(segment - 1 + i) * knot_dt for i in range(n)]
        raw = np.vstack([self._raw_reference(target_index, ti)[0] for ti in control_times])
        return self._optimize_control_points(raw, knot_dt), float(t) / knot_dt - segment

    def _optimize_control_points(self, raw: np.ndarray, knot_dt: float) -> np.ndarray:
        return self._bspline_optimizer.optimize(raw, knot_dt, self.obstacles)

    def reference(self, target_index: int, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.smoothing.enabled:
            return self._raw_reference(target_index, t)
        ctrl, u = self._bspline_control_points(target_index, t)
        knot_dt = max(float(self.smoothing.knot_dt), 1.0e-3)
        basis, basis_d, basis_dd = self._bspline_basis(u)
        active = ctrl[:4]
        pos = basis.dot(active)
        vel = basis_d.dot(active) / knot_dt
        acc = basis_dd.dot(active) / (knot_dt * knot_dt)
        return pos, vel, acc

    def potential_field(self, position: np.ndarray) -> np.ndarray:
        total = np.zeros(3)
        for obstacle in self.obstacles:
            diff = position - obstacle.position
            dist = float(np.linalg.norm(diff))
            if dist < 1.0e-4:
                continue
            term = (1.0 / dist - 0.1) * diff / max(dist ** 3, EPS)
            total += term
        return clamp_norm(self.potential_gain * total, self.potential_clip)

    def desired_acceleration(self, target_index: int, state: AgentState, t: float) -> np.ndarray:
        ref_p, ref_v, ref_a = self.reference(target_index, t)
        velocity_term = ref_v - state.velocity
        if not self.use_dimension_consistent_reference_velocity:
            velocity_term = ref_p - state.velocity
        return ref_a + (ref_p - state.position) + velocity_term + self.potential_field(state.position)

    def target_pair_accelerations(self, states: Sequence[AgentState]) -> List[np.ndarray]:
        params = self.smoothing
        accels = [np.zeros(3) for _ in states]
        clearance = max(float(params.target_pair_clearance), 0.0)
        activation_margin = max(float(params.target_pair_activation_margin), 1.0e-3)
        for i, first in enumerate(states):
            for j, second in enumerate(states[i + 1 :], start=i + 1):
                diff = first.position - second.position
                dist = float(np.linalg.norm(diff))
                if dist < 1.0e-6:
                    direction = np.array([1.0, 0.0, 0.0])
                    dist = 1.0e-6
                else:
                    direction = diff / dist
                min_center_distance = float(first.radius) + float(second.radius) + clearance
                activation_distance = min_center_distance + activation_margin
                if dist >= activation_distance:
                    continue
                gap = activation_distance - dist
                hard_gap = max(min_center_distance - dist, 0.0)
                rel_speed = float(np.dot(first.velocity - second.velocity, direction))
                closing_damping = max(-rel_speed, 0.0)
                magnitude = (
                    float(params.target_pair_gain) * gap * gap
                    + float(params.target_pair_hard_gain) * hard_gap * hard_gap
                    + float(params.target_pair_damping) * closing_damping
                )
                accel = magnitude * direction
                accels[i] += accel
                accels[j] -= accel
        return accels

    def _limit_target_acceleration(self, raw_accel: np.ndarray, previous_accel: np.ndarray, dt: float) -> np.ndarray:
        acc = np.asarray(raw_accel, dtype=float).reshape(3)
        if self.smoothing.enabled and self.smoothing.limit_integrated_target:
            if self.smoothing.max_ref_jerk > 0.0:
                acc = previous_accel + clamp_norm(acc - previous_accel, self.smoothing.max_ref_jerk * max(float(dt), 1.0e-3))
            acc = clamp_norm(acc, self.smoothing.max_ref_accel)
        return acc

    def integrate(self, target_index: int, state: AgentState, t: float, dt: float) -> AgentState:
        acc = self.desired_acceleration(target_index, state, t)
        acc = self._limit_target_acceleration(acc, state.acceleration, dt)
        vel = state.velocity + acc * dt
        if self.smoothing.enabled and self.smoothing.limit_integrated_target:
            vel = clamp_norm(vel, self.smoothing.max_ref_speed)
        pos = state.position + vel * dt
        return AgentState(pos, vel, acc, radius=state.radius, stamp=t + dt, name=state.name)

    def integrate_all(self, states: Sequence[AgentState], t: float, dt: float) -> List[AgentState]:
        pair_accels = self.target_pair_accelerations(states)
        integrated = []
        for idx, state in enumerate(states):
            raw_acc = self.desired_acceleration(idx, state, t) + pair_accels[idx]
            acc = self._limit_target_acceleration(raw_acc, state.acceleration, dt)
            vel = state.velocity + acc * dt
            if self.smoothing.enabled and self.smoothing.limit_integrated_target:
                vel = clamp_norm(vel, self.smoothing.max_ref_speed)
            pos = state.position + vel * dt
            integrated.append(AgentState(pos, vel, acc, radius=state.radius, stamp=t + dt, name=state.name))
        return integrated
