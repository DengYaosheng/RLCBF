#!/usr/bin/env python3
"""Run the paper target-pursuit experiment without ROS/Gazebo."""

import argparse
import csv
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PKG_DIR, "src"))

from trust_up_xtdrone.config_io import (  # noqa: E402
    default_config_path,
    load_yaml,
    obstacles_from_config,
    policy_from_config,
    safety_from_config,
    scenario_offset,
    target_initial_states,
    limits_from_config,
)
from trust_up_xtdrone.core import AgentState, NominalPolicy, TrustUpController, TargetTrajectory, clamp_norm  # noqa: E402


def smooth_command(raw_cmd, previous, dt, limits, cfg):
    smooth_cfg = cfg.get("command_smoothing", {})
    raw_cmd = np.asarray(raw_cmd, dtype=float).reshape(3)
    if not bool(smooth_cfg.get("enabled", True)):
        return raw_cmd
    if previous is None:
        previous = np.zeros(3)
    tau = float(smooth_cfg.get("tau_s", 0.35))
    max_accel = float(smooth_cfg.get("max_command_accel", limits.max_accel))
    max_vertical_accel = float(smooth_cfg.get("max_vertical_command_accel", 0.8 * max_accel))
    alpha = dt / max(tau + dt, 1.0e-3)
    desired = previous + alpha * (raw_cmd - previous)
    delta = desired - previous
    delta_xy = clamp_norm(delta[:2], max(max_accel, 0.0) * dt)
    delta_z = float(np.clip(delta[2], -max(max_vertical_accel, 0.0) * dt, max(max_vertical_accel, 0.0) * dt))
    cmd = previous + np.array([delta_xy[0], delta_xy[1], delta_z], dtype=float)
    cmd = clamp_norm(cmd, limits.max_speed)
    if abs(cmd[2]) > limits.max_vertical_speed:
        cmd[2] = float(np.sign(cmd[2]) * limits.max_vertical_speed)
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--scenario", choices=["circle", "figure8"], default=None)
    parser.add_argument("--output", default=os.path.join(PKG_DIR, "headless_result.csv"))
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.scenario:
        cfg.setdefault("scenario", {})["name"] = args.scenario
    dt = float(cfg.get("paper", {}).get("dt", 0.1))
    steps = int(cfg.get("paper", {}).get("steps", 600))

    safety = safety_from_config(cfg)
    limits = limits_from_config(cfg)
    policy_params = policy_from_config(cfg)
    obstacles = obstacles_from_config(cfg)
    targets = target_initial_states(cfg)
    trajectory = TargetTrajectory(
        cfg.get("scenario", {}).get("name", "circle"),
        obstacles,
        offset=scenario_offset(cfg),
        potential_gain=float(cfg.get("scenario", {}).get("target_potential_gain", 1.0)),
        potential_clip=float(cfg.get("scenario", {}).get("target_potential_clip", 1.5)),
        use_dimension_consistent_reference_velocity=bool(
            cfg.get("scenario", {}).get("use_dimension_consistent_reference_velocity", True)
        ),
        reference_scale=float(cfg.get("scenario", {}).get("reference_scale", 1.0)),
        angular_rate=float(cfg.get("scenario", {}).get("angular_rate", 0.1)),
        smoothing=cfg.get("trajectory_smoothing", {}),
    )
    if bool(cfg.get("scenario", {}).get("initialize_targets_from_reference", True)):
        targets = [
            target.__class__.from_xyz(
                trajectory.reference(i, 0.0)[0],
                trajectory.reference(i, 0.0)[1],
                trajectory.reference(i, 0.0)[2],
                radius=target.radius,
                name=target.name,
            )
            for i, target in enumerate(targets)
        ]

    controllers = [
        TrustUpController(safety, limits, NominalPolicy(safety, limits, policy_params))
        for _ in targets
    ]
    filtered_commands = [None for _ in targets]
    default_uav_radius = float(cfg.get("physical_envelope", {}).get("uav_radius", 0.0))
    pursuer_cfg = cfg.get("pursuers", [])
    pursuers = []
    for i, target in enumerate(targets):
        # Start inside the allowed shell to satisfy the paper initial condition.
        direction = target.velocity.copy()
        if float((direction ** 2).sum() ** 0.5) < 1.0e-6:
            direction = target.position - obstacles[0].position
        direction = direction / max(float((direction ** 2).sum() ** 0.5), 1.0e-6)
        pursuer_radius = float(pursuer_cfg[i].get("radius", default_uav_radius)) if i < len(pursuer_cfg) else default_uav_radius
        bootstrap = AgentState.from_xyz([0.0, 0.0, 0.0], radius=pursuer_radius)
        initial_range = safety.desired_bound(bootstrap, target)
        pursuers.append(target.__class__.from_xyz(target.position - initial_range * direction, radius=pursuer_radius, name="pursuer_%d" % i))

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "t",
                "agent",
                "px",
                "py",
                "pz",
                "tx",
                "ty",
                "tz",
                "distance",
                "cmd_vx",
                "cmd_vy",
                "cmd_vz",
                "qp_feasible",
                "qp_max_violation",
            ]
        )
        for step in range(steps):
            t = step * dt
            targets = trajectory.integrate_all(targets, t, dt)
            for i, controller in enumerate(controllers):
                other_obstacles = obstacles[:]
                for j, p in enumerate(pursuers):
                    if i != j:
                        other_obstacles.append(p)
                for j, target in enumerate(targets):
                    if i != j:
                        other_obstacles.append(target)
                cmd, diag = controller.step(pursuers[i], targets[i], other_obstacles, dt)
                cmd_smoothed = smooth_command(cmd, filtered_commands[i], dt, limits, cfg)
                cmd_guarded, _ = controller.guard_velocity_command(cmd_smoothed, pursuers[i], targets[i], [targets[i]] + other_obstacles)
                filtered_commands[i] = cmd_guarded.copy()
                controller.command_velocity = cmd_guarded.copy()
                pursuers[i].velocity = cmd_guarded
                pursuers[i].position = pursuers[i].position + cmd_guarded * dt
                dist = float(((pursuers[i].position - targets[i].position) ** 2).sum() ** 0.5)
                writer.writerow(
                    [
                        "%.3f" % t,
                        i,
                        *["%.6f" % v for v in pursuers[i].position],
                        *["%.6f" % v for v in targets[i].position],
                        "%.6f" % dist,
                        *["%.6f" % v for v in cmd_guarded],
                        bool(diag["qp_feasible"]),
                        "%.6g" % float(diag["qp_max_violation"]),
                    ]
                )

    print("wrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
