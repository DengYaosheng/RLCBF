#!/usr/bin/env python3
"""Run headless TRUST-UP scenarios and emit constraint/smoothness metrics."""

import argparse
import collections
import csv
import json
import os
import subprocess
import sys
from typing import Dict, List

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PKG_DIR, "src"))

from trust_up_xtdrone.config_io import default_config_path, load_yaml, safety_from_config  # noqa: E402


def percentile(values: List[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def evaluate_csv(path: str, cfg: Dict) -> Dict:
    safety = safety_from_config(cfg)
    targets = cfg.get("targets", [])
    pursuers = cfg.get("pursuers", [])
    obstacles = cfg.get("obstacles", [])
    radius_sum = 0.0
    if safety.use_object_radius:
        radius_sum = float(pursuers[0].get("radius", 0.0)) + float(targets[0].get("radius", 0.0))
    paper = (float(safety.collision_radius), float(safety.sensing_radius))
    guard = (paper[0] + float(safety.tracking_inner_margin), paper[1] - float(safety.tracking_outer_margin))
    target_pair_min = float(cfg.get("trajectory_smoothing", {}).get("target_pair_clearance", 0.0))
    target_radius_sum = 0.0
    if len(targets) >= 2:
        target_radius_sum = float(targets[0].get("radius", 0.0)) + float(targets[1].get("radius", 0.0))

    rows = list(csv.DictReader(open(path, "r", encoding="utf-8")))
    clearances = collections.defaultdict(list)
    commands = collections.defaultdict(list)
    target_by_t = collections.defaultdict(dict)
    pursuer_by_t = collections.defaultdict(dict)
    paper_violations = 0
    guard_violations = 0
    pursuer_obstacle_violations = 0
    target_obstacle_violations = 0
    qp_positive = 0
    max_qp = 0.0
    pursuer_obstacle_clearances = []
    target_obstacle_clearances = []
    for row in rows:
        agent = int(row["agent"])
        t = float(row["t"])
        pursuer_position = np.array([float(row["px"]), float(row["py"]), float(row["pz"])], dtype=float)
        target_position = np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])], dtype=float)
        clearance = float(row["distance"]) - radius_sum
        clearances[agent].append(clearance)
        commands[agent].append(np.array([float(row["cmd_vx"]), float(row["cmd_vy"]), float(row["cmd_vz"])], dtype=float))
        target_by_t[t][agent] = target_position
        pursuer_by_t[t][agent] = pursuer_position
        if clearance < paper[0] - 1.0e-6 or clearance > paper[1] + 1.0e-6:
            paper_violations += 1
        if clearance < guard[0] - 1.0e-6 or clearance > guard[1] + 1.0e-6:
            guard_violations += 1
        for obstacle in obstacles:
            obstacle_center = np.asarray(obstacle["center"], dtype=float)
            if bool(cfg.get("scenario", {}).get("use_gazebo_offset", False)):
                obstacle_center = obstacle_center + np.asarray(cfg.get("scenario", {}).get("gazebo_frame_offset", [0.0, 0.0, 0.0]), dtype=float)
            obstacle_radius = float(obstacle.get("radius", 0.0)) if safety.use_object_radius else 0.0
            pursuer_radius = float(pursuers[agent].get("radius", 0.0)) if safety.use_object_radius and agent < len(pursuers) else 0.0
            target_radius = float(targets[agent].get("radius", 0.0)) if safety.use_object_radius and agent < len(targets) else 0.0
            pursuer_obstacle_clearance = float(np.linalg.norm(pursuer_position - obstacle_center) - pursuer_radius - obstacle_radius)
            target_obstacle_clearance = float(np.linalg.norm(target_position - obstacle_center) - target_radius - obstacle_radius)
            pursuer_obstacle_clearances.append(pursuer_obstacle_clearance)
            target_obstacle_clearances.append(target_obstacle_clearance)
            if pursuer_obstacle_clearance < paper[0] - 1.0e-6:
                pursuer_obstacle_violations += 1
            if target_obstacle_clearance < 0.0 - 1.0e-6:
                target_obstacle_violations += 1
        qp = float(row["qp_max_violation"])
        qp_positive += int(qp > 1.0e-8)
        max_qp = max(max_qp, qp)

    target_pair_clearances = []
    for data in target_by_t.values():
        if 0 in data and 1 in data:
            target_pair_clearances.append(float(np.linalg.norm(data[0] - data[1]) - target_radius_sum))
    target_pair_violations = sum(1 for value in target_pair_clearances if value < target_pair_min - 1.0e-6)
    pursuer_pair_clearances = []
    pursuer_pair_violations = 0
    if len(pursuers) >= 2:
        pursuer_radius_sum = (
            float(pursuers[0].get("radius", 0.0)) + float(pursuers[1].get("radius", 0.0))
            if safety.use_object_radius
            else 0.0
        )
        for data in pursuer_by_t.values():
            if 0 in data and 1 in data:
                clearance = float(np.linalg.norm(data[0] - data[1]) - pursuer_radius_sum)
                pursuer_pair_clearances.append(clearance)
                if clearance < paper[0] - 1.0e-6:
                    pursuer_pair_violations += 1

    agents = {}
    for agent, values in sorted(clearances.items()):
        acc = []
        jerk = []
        previous_acc = None
        for current, nxt in zip(commands[agent], commands[agent][1:]):
            command_acc = (nxt - current) / 0.1
            acc.append(float(np.linalg.norm(command_acc)))
            if previous_acc is not None:
                jerk.append(float(np.linalg.norm((command_acc - previous_acc) / 0.1)))
            previous_acc = command_acc
        agents[str(agent)] = {
            "clearance_min": min(values),
            "clearance_max": max(values),
            "command_accel_p95": percentile(acc, 95),
            "command_accel_max": max(acc) if acc else 0.0,
            "command_jerk_p95": percentile(jerk, 95),
            "command_jerk_max": max(jerk) if jerk else 0.0,
        }

    paper_total_violations = (
        paper_violations
        + pursuer_pair_violations
        + pursuer_obstacle_violations
        + target_pair_violations
        + target_obstacle_violations
    )

    return {
        "rows": len(rows),
        "paper_total_violations": paper_total_violations,
        "pursuer_target_violations": paper_violations,
        "paper_violations": paper_violations,
        "guard_violations": guard_violations,
        "pursuer_pair_violations": pursuer_pair_violations,
        "pursuer_obstacle_violations": pursuer_obstacle_violations,
        "target_obstacle_violations": target_obstacle_violations,
        "target_pair_violations": target_pair_violations,
        "pursuer_pair_clearance_min": min(pursuer_pair_clearances) if pursuer_pair_clearances else None,
        "pursuer_pair_clearance_max": max(pursuer_pair_clearances) if pursuer_pair_clearances else None,
        "pursuer_obstacle_clearance_min": min(pursuer_obstacle_clearances) if pursuer_obstacle_clearances else None,
        "target_obstacle_clearance_min": min(target_obstacle_clearances) if target_obstacle_clearances else None,
        "target_pair_clearance_min": min(target_pair_clearances) if target_pair_clearances else None,
        "target_pair_clearance_max": max(target_pair_clearances) if target_pair_clearances else None,
        "qp_positive": qp_positive,
        "qp_max_violation": max_qp,
        "agents": agents,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--output-dir", default="/tmp/trust_up_metrics")
    parser.add_argument("--scenarios", nargs="+", default=["circle", "figure8"])
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    summary = {}
    runner = os.path.join(PKG_DIR, "scripts", "run_headless_paper_experiment.py")
    for scenario in args.scenarios:
        csv_path = os.path.join(args.output_dir, "%s.csv" % scenario)
        subprocess.check_call([sys.executable, runner, "--config", args.config, "--scenario", scenario, "--output", csv_path])
        summary[scenario] = evaluate_csv(csv_path, cfg)

    json_path = os.path.join(args.output_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("wrote %s" % json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
