"""Adaptive robustness estimation for TRUST-UP target pursuit."""

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np

from .utils import EPS, clamp_norm, dataclass_subset


@dataclass
class AdaptiveParameters:
    enabled: bool = True
    adaptation_gain: float = 0.55
    leakage: float = 0.08
    residual_filter_gain: float = 0.35
    disturbance_limit: float = 0.9
    max_margin: float = 0.28
    residual_margin_gain: float = 0.08
    pressure_margin_gain: float = 0.045
    energy_margin_gain: float = 0.025
    pressure_activation_margin: float = 0.35

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, object]]) -> "AdaptiveParameters":
        if data is None:
            return cls()
        return cls(**dataclass_subset(cls, data))


@dataclass
class AdaptiveEstimate:
    disturbance: np.ndarray = field(default_factory=lambda: np.zeros(3))
    residual: np.ndarray = field(default_factory=lambda: np.zeros(3))
    disturbance_norm: float = 0.0
    residual_norm: float = 0.0
    margin_scale: float = 0.0
    tracking_energy: float = 0.0
    clearance_pressure: float = 0.0

    def as_dict(self) -> Mapping[str, object]:
        return {
            "disturbance": self.disturbance.copy(),
            "residual": self.residual.copy(),
            "disturbance_norm": self.disturbance_norm,
            "residual_norm": self.residual_norm,
            "margin_scale": self.margin_scale,
            "tracking_energy": self.tracking_energy,
            "clearance_pressure": self.clearance_pressure,
        }


class AdaptiveRobustnessEstimator:
    """Online residual estimator used to tighten CBF constraints near stress."""

    def __init__(self, params: Optional[AdaptiveParameters] = None):
        self.params = params or AdaptiveParameters()
        self.estimate = AdaptiveEstimate()
        self._previous_velocity: Optional[np.ndarray] = None
        self._filtered_residual = np.zeros(3)

    def reset(self, velocity: Optional[Sequence[float]] = None) -> None:
        self.estimate = AdaptiveEstimate()
        self._filtered_residual = np.zeros(3)
        self._previous_velocity = None if velocity is None else np.asarray(velocity, dtype=float).reshape(3)

    def update(
        self,
        pursuer,
        target,
        obstacles: Sequence[object],
        previous_commanded_accel: Sequence[float],
        dt: float,
        safety,
    ) -> AdaptiveEstimate:
        if not self.params.enabled:
            self._previous_velocity = np.asarray(pursuer.velocity, dtype=float).reshape(3)
            self.estimate = AdaptiveEstimate()
            return self.estimate

        dt = max(float(dt), 1.0e-3)
        velocity = np.asarray(pursuer.velocity, dtype=float).reshape(3)
        commanded = np.asarray(previous_commanded_accel, dtype=float).reshape(3)
        if self._previous_velocity is None:
            residual = np.zeros(3)
        else:
            observed_accel = (velocity - self._previous_velocity) / dt
            residual = observed_accel - commanded
        self._previous_velocity = velocity.copy()

        gain = float(np.clip(self.params.residual_filter_gain, 0.0, 1.0))
        self._filtered_residual = (1.0 - gain) * self._filtered_residual + gain * residual

        pressure = self._clearance_pressure(pursuer, target, obstacles, safety)
        energy = self._tracking_energy(pursuer, target, safety)
        feature_scale = 1.0 + 0.35 * pressure + 0.15 * np.tanh(energy)
        next_disturbance = (
            (1.0 - float(self.params.leakage) * dt) * self.estimate.disturbance
            + float(self.params.adaptation_gain) * dt * feature_scale * self._filtered_residual
        )
        disturbance = clamp_norm(next_disturbance, float(self.params.disturbance_limit))

        residual_norm = float(np.linalg.norm(self._filtered_residual))
        disturbance_norm = float(np.linalg.norm(disturbance))
        margin = (
            float(self.params.residual_margin_gain) * disturbance_norm
            + float(self.params.pressure_margin_gain) * pressure
            + float(self.params.energy_margin_gain) * np.sqrt(max(energy, 0.0))
        )
        margin = float(np.clip(margin, 0.0, float(self.params.max_margin)))

        self.estimate = AdaptiveEstimate(
            disturbance=disturbance,
            residual=self._filtered_residual.copy(),
            disturbance_norm=disturbance_norm,
            residual_norm=residual_norm,
            margin_scale=margin,
            tracking_energy=float(energy),
            clearance_pressure=float(pressure),
        )
        return self.estimate

    def compensate(self, nominal_accel: Sequence[float]) -> np.ndarray:
        if not self.params.enabled:
            return np.asarray(nominal_accel, dtype=float).reshape(3)
        return np.asarray(nominal_accel, dtype=float).reshape(3) - self.estimate.disturbance

    def _tracking_energy(self, pursuer, target, safety) -> float:
        relative_position = np.asarray(pursuer.position, dtype=float).reshape(3) - np.asarray(target.position, dtype=float).reshape(3)
        relative_velocity = np.asarray(pursuer.velocity, dtype=float).reshape(3) - np.asarray(target.velocity, dtype=float).reshape(3)
        distance = float(np.linalg.norm(relative_position))
        desired = float(safety.desired_bound(pursuer, target))
        range_error = distance - desired
        return 0.5 * range_error * range_error + 0.125 * float(np.dot(relative_velocity, relative_velocity))

    def _clearance_pressure(self, pursuer, target, obstacles: Sequence[object], safety) -> float:
        activation = max(float(self.params.pressure_activation_margin), 1.0e-3)
        pressure = self._shell_pressure(
            pursuer,
            target,
            float(safety.enforced_collision_bound(pursuer, target)),
            float(safety.enforced_sensing_bound(pursuer, target)),
            activation,
        )
        for obstacle in obstacles:
            lower = float(safety.enforced_collision_bound(pursuer, obstacle, obstacle=True))
            pressure += self._lower_bound_pressure(pursuer, obstacle, lower, activation)
        return float(pressure)

    @staticmethod
    def _center_distance(first, second) -> float:
        first_position = np.asarray(first.position, dtype=float).reshape(3)
        second_position = np.asarray(second.position, dtype=float).reshape(3)
        return float(np.linalg.norm(first_position - second_position))

    def _lower_bound_pressure(self, first, second, lower_bound: float, activation: float) -> float:
        clearance = self._center_distance(first, second) - lower_bound
        if clearance >= activation:
            return 0.0
        return float(((activation - clearance) / activation) ** 2)

    def _shell_pressure(self, first, second, lower_bound: float, upper_bound: float, activation: float) -> float:
        distance = self._center_distance(first, second)
        lower_gap = distance - lower_bound
        upper_gap = upper_bound - distance
        inner = max((activation - lower_gap) / activation, 0.0)
        outer = max((activation - upper_gap) / activation, 0.0)
        return float(inner * inner + outer * outer + EPS * 0.0)
