"""Adaptive robustness estimation for TRUST-UP target pursuit.

The local names in this module intentionally follow compact paper notation:
``z``/``dz`` for relative position and velocity, ``eps`` for the sampled
residual, ``nu``/``eta`` for bounded uncertainty surrogates, and ``rho`` for
constraint pressure.  The comments map these names to the TRUST-UP equations.
"""

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

        tau = max(float(dt), 1.0e-3)
        v = np.asarray(pursuer.velocity, dtype=float).reshape(3)
        u0 = np.asarray(previous_commanded_accel, dtype=float).reshape(3)
        if self._previous_velocity is None:
            eps = np.zeros(3)
        else:
            # TRUST-UP Eq. (19): dot(u_i) = v_i + Z_i xi.  The simulator does not
            # identify theta/xi explicitly, so eps is the sampled model mismatch:
            # eps ~= dot(u_i)_meas - v_i, a discrete proxy for Z_i tilde(xi).
            du = (v - self._previous_velocity) / tau
            eps = du - u0
        self._previous_velocity = v.copy()

        a = float(np.clip(self.params.residual_filter_gain, 0.0, 1.0))
        self._filtered_residual = (1.0 - a) * self._filtered_residual + a * eps

        # rho corresponds to how close C_{u,i}, C_{c,i}, and C_{s,i} are to their
        # boundaries.  e is the pursuit tracking energy around the desired shell.
        rho = self._clearance_pressure(pursuer, target, obstacles, safety)
        e = self._tracking_energy(pursuer, target, safety)
        chi = 1.0 + 0.35 * rho + 0.15 * np.tanh(e)

        # TRUST-UP Lemma 1 bounds ||tilde(theta)|| <= nu_bar and
        # ||tilde(xi)|| <= eta_bar.  d_hat is an implementation-level lumped
        # disturbance surrogate for Y_i tilde(theta) + Z_i tilde(xi).
        d_hat = (
            (1.0 - float(self.params.leakage) * tau) * self.estimate.disturbance
            + float(self.params.adaptation_gain) * tau * chi * self._filtered_residual
        )
        d = clamp_norm(d_hat, float(self.params.disturbance_limit))

        nu = float(np.linalg.norm(self._filtered_residual))
        eta = float(np.linalg.norm(d))
        delta = (
            float(self.params.residual_margin_gain) * eta
            + float(self.params.pressure_margin_gain) * rho
            + float(self.params.energy_margin_gain) * np.sqrt(max(e, 0.0))
        )
        delta = float(np.clip(delta, 0.0, float(self.params.max_margin)))

        self.estimate = AdaptiveEstimate(
            disturbance=d,
            residual=self._filtered_residual.copy(),
            disturbance_norm=eta,
            residual_norm=nu,
            margin_scale=delta,
            tracking_energy=float(e),
            clearance_pressure=float(rho),
        )
        return self.estimate

    def compensate(self, nominal_accel: Sequence[float]) -> np.ndarray:
        if not self.params.enabled:
            return np.asarray(nominal_accel, dtype=float).reshape(3)
        return np.asarray(nominal_accel, dtype=float).reshape(3) - self.estimate.disturbance

    def _tracking_energy(self, pursuer, target, safety) -> float:
        # zeta_i = x_i - q_i appears in TRUST-UP Eqs. (18), (26), (27), (33), (34).
        z = np.asarray(pursuer.position, dtype=float).reshape(3) - np.asarray(target.position, dtype=float).reshape(3)
        dz = np.asarray(pursuer.velocity, dtype=float).reshape(3) - np.asarray(target.velocity, dtype=float).reshape(3)
        rr = float(np.linalg.norm(z))
        ell = float(safety.desired_bound(pursuer, target))
        return 0.5 * (rr - ell) ** 2 + 0.125 * float(np.dot(dz, dz))

    def _clearance_pressure(self, pursuer, target, obstacles: Sequence[object], safety) -> float:
        eps = max(float(self.params.pressure_activation_margin), 1.0e-3)
        psi = self._shell_pressure(
            pursuer,
            target,
            float(safety.enforced_collision_bound(pursuer, target)),
            float(safety.enforced_sensing_bound(pursuer, target)),
            eps,
        )
        for obstacle in obstacles:
            r = float(safety.enforced_collision_bound(pursuer, obstacle, obstacle=True))
            psi += self._lower_bound_pressure(pursuer, obstacle, r, eps)
        return float(psi)

    @staticmethod
    def _center_distance(first, second) -> float:
        x = np.asarray(first.position, dtype=float).reshape(3)
        p = np.asarray(second.position, dtype=float).reshape(3)
        return float(np.linalg.norm(x - p))

    @staticmethod
    def _hc_from_rr(rr: float, r: float) -> float:
        return float(rr * rr - r * r)

    @staticmethod
    def _hs_from_rr(rr: float, r_big: float) -> float:
        return float(r_big * r_big - rr * rr)

    def _lower_bound_pressure(self, first, second, r: float, eps: float) -> float:
        # Eq. (17) pressure uses the same squared barrier h_{c,i,k}, not a
        # separate linear distance heuristic.
        rr = self._center_distance(first, second)
        hc = self._hc_from_rr(rr, r)
        hc_eps = self._hc_from_rr(r + eps, r)
        pc = max((hc_eps - hc) / max(hc_eps, EPS), 0.0)
        return float(pc * pc)

    def _shell_pressure(self, first, second, r: float, r_big: float, eps: float) -> float:
        # Eq. (17) inner barrier and Eq. (18) outer sensing barrier are both
        # evaluated in their native squared-distance units.
        rr = self._center_distance(first, second)
        hc = self._hc_from_rr(rr, r)
        hs = self._hs_from_rr(rr, r_big)
        hc_eps = self._hc_from_rr(r + eps, r)
        hs_eps = self._hs_from_rr(max(r_big - eps, 0.0), r_big)
        pc = max((hc_eps - hc) / max(hc_eps, EPS), 0.0)
        ps = max((hs_eps - hs) / max(hs_eps, EPS), 0.0)
        return float(pc * pc + ps * ps + EPS * 0.0)
