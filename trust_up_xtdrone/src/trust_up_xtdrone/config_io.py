"""Configuration helpers shared by ROS and non-ROS entrypoints."""

import os
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import yaml

from .core import AgentState, PolicyParameters, SafetyParameters, VehicleLimits


def package_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_path(path: str) -> str:
    if not path:
        return path
    text = os.path.expanduser(os.path.expandvars(path))
    marker = "$(find trust_up_xtdrone)"
    if marker in text:
        text = text.replace(marker, package_root())
    return text


def load_yaml(path: str) -> Dict[str, Any]:
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def default_config_path() -> str:
    return os.path.join(package_root(), "config", "paper_target_pursuit.yaml")


def vector3(value: Sequence[float]) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(3)


def scenario_offset(config: Mapping[str, Any]) -> np.ndarray:
    scenario = config.get("scenario", {})
    if scenario.get("use_gazebo_offset", False):
        return vector3(scenario.get("gazebo_frame_offset", [0.0, 0.0, 0.0]))
    return np.zeros(3)


def safety_from_config(config: Mapping[str, Any]) -> SafetyParameters:
    data = config.get("safety", {})
    return SafetyParameters(**{k: data[k] for k in SafetyParameters.__dataclass_fields__ if k in data})


def limits_from_config(config: Mapping[str, Any]) -> VehicleLimits:
    data = config.get("vehicle_profile", {})
    return VehicleLimits(**{k: data[k] for k in VehicleLimits.__dataclass_fields__ if k in data})


def policy_from_config(config: Mapping[str, Any]) -> PolicyParameters:
    data = config.get("nominal_policy", {})
    return PolicyParameters(**{k: data[k] for k in PolicyParameters.__dataclass_fields__ if k in data})


def obstacles_from_config(config: Mapping[str, Any], *, apply_offset: bool = True) -> List[AgentState]:
    offset = scenario_offset(config) if apply_offset else np.zeros(3)
    states = []
    for item in config.get("obstacles", []):
        states.append(
            AgentState.from_xyz(
                vector3(item["center"]) + offset,
                radius=float(item.get("radius", 0.0)),
                name=str(item.get("name", "obstacle")),
            )
        )
    return states


def target_initial_states(config: Mapping[str, Any], *, apply_offset: bool = True) -> List[AgentState]:
    offset = scenario_offset(config) if apply_offset else np.zeros(3)
    targets = []
    for idx, item in enumerate(config.get("targets", [])):
        targets.append(
            AgentState.from_xyz(
                vector3(item["initial_position"]) + offset,
                radius=float(item.get("radius", 0.0)),
                name=str(item.get("name", "target_%d" % idx)),
            )
        )
    return targets


def pursuer_initial_states(config: Mapping[str, Any], *, apply_offset: bool = True) -> List[AgentState]:
    offset = scenario_offset(config) if apply_offset else np.zeros(3)
    pursuers = []
    for idx, item in enumerate(config.get("pursuers", [])):
        pursuers.append(
            AgentState.from_xyz(
                vector3(item["initial_position"]) + offset,
                radius=float(item.get("radius", 0.0)),
                name=str(item.get("name", "pursuer_%d" % idx)),
            )
        )
    return pursuers


def as_float_list(values: Iterable[float]) -> List[float]:
    return [float(x) for x in values]
