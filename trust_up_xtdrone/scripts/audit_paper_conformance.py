#!/usr/bin/env python3
"""Audit TRUST-UP target-pursuit settings against arXiv 2411.17552v3."""

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PKG_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PKG_DIR / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from spawn_paper_world import xtdrone_visual_uav_sdf  # noqa: E402
from trust_up_xtdrone.config_io import default_config_path, load_yaml  # noqa: E402


PAPER = {
    "dt": 0.1,
    "steps": 600,
    "collision_radius": 0.5,
    "sensing_radius": 1.0,
    "reference_scale": 1.0,
    "angular_rate": 0.1,
    "obstacles": [
        ([4.70, 3.25, 3.00], 0.3),
        ([-4.20, 3.00, 4.75], 0.3),
    ],
}


def close(value: float, expected: float, tol: float = 1.0e-9) -> bool:
    return abs(float(value) - float(expected)) <= tol


def vector_close(value: Sequence[float], expected: Sequence[float], tol: float = 1.0e-9) -> bool:
    return bool(np.allclose(np.asarray(value, dtype=float), np.asarray(expected, dtype=float), atol=tol, rtol=0.0))


def add_check(checks: List[Dict], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def check_config(cfg: Dict, checks: List[Dict]) -> None:
    paper = cfg.get("paper", {})
    scenario = cfg.get("scenario", {})
    safety = cfg.get("safety", {})
    add_check(checks, "paper.dt", close(paper.get("dt"), PAPER["dt"]), "expected %.3f" % PAPER["dt"])
    add_check(checks, "paper.steps", int(paper.get("steps", -1)) == PAPER["steps"], "expected %d" % PAPER["steps"])
    add_check(
        checks,
        "scenario.reference_scale",
        close(scenario.get("reference_scale"), PAPER["reference_scale"]),
        "paper uses unscaled 5 m references",
    )
    add_check(
        checks,
        "scenario.angular_rate",
        close(scenario.get("angular_rate"), PAPER["angular_rate"]),
        "paper uses 0.1 rad/s and 0.2 rad/s components",
    )
    add_check(
        checks,
        "safety.collision_radius",
        close(safety.get("collision_radius"), PAPER["collision_radius"]),
        "paper r_i = 0.5 m",
    )
    add_check(
        checks,
        "safety.sensing_radius",
        close(safety.get("sensing_radius"), PAPER["sensing_radius"]),
        "paper R_i = 1.0 m before physical hull padding",
    )
    add_check(
        checks,
        "agents.count",
        len(cfg.get("pursuers", [])) == 2 and len(cfg.get("targets", [])) == 2,
        "paper experiment has two pursuers and two target UAVs",
    )
    add_check(
        checks,
        "safety.use_object_radius",
        bool(safety.get("use_object_radius", False)),
        "physical XTDrone hull radii are included for center-to-center deployment constraints",
    )
    obstacles = cfg.get("obstacles", [])
    add_check(checks, "obstacles.count", len(obstacles) == 2, "paper experiment has two spherical obstacles")
    for idx, (center, radius) in enumerate(PAPER["obstacles"]):
        if idx >= len(obstacles):
            add_check(checks, "obstacle_%d" % idx, False, "missing")
            continue
        obs = obstacles[idx]
        add_check(
            checks,
            "obstacle_%d.center" % idx,
            vector_close(obs.get("center", []), center),
            "expected %s" % center,
        )
        add_check(
            checks,
            "obstacle_%d.radius" % idx,
            close(obs.get("radius"), radius),
            "expected %.2f m" % radius,
        )


def check_xtdrone_visual(cfg: Dict, checks: List[Dict]) -> None:
    source_sdf = os.path.expanduser(os.path.expandvars(cfg.get("target_visual", {}).get("xtdrone_sdf", "")))
    add_check(checks, "xtdrone.source_sdf_exists", bool(source_sdf and os.path.exists(source_sdf)), source_sdf)
    if not source_sdf or not os.path.exists(source_sdf):
        return
    try:
        sdf = xtdrone_visual_uav_sdf("audit_uav", source_sdf)
        root = ET.fromstring(sdf)
        visual_count = len(root.findall(".//visual"))
        add_check(checks, "xtdrone.visual_count", visual_count > 0, "%d visuals" % visual_count)
        add_check(checks, "xtdrone.no_fallback_visual", "fallback_body" not in sdf, "must use resolved XTDrone/PX4 iris meshes")
        add_check(checks, "xtdrone.no_unresolved_model_uri", "model://" not in sdf, "all mesh URIs must resolve to real files")
        add_check(checks, "xtdrone.no_plugins", len(root.findall(".//plugin")) == 0, "replay visuals are kinematic")
        add_check(checks, "xtdrone.no_collisions", len(root.findall(".//collision")) == 0, "visual UAVs do not perturb Gazebo physics")
        add_check(
            checks,
            "xtdrone.same_visual_for_targets",
            "iris.stl" in sdf and "iris_prop" in sdf,
            "targets are spawned from the same XTDrone UAV visual source",
        )
    except Exception as exc:
        add_check(checks, "xtdrone.visual_sdf_parse", False, str(exc))


def run_metrics(config_path: str, output_dir: Path) -> Dict:
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    runner = SCRIPT_DIR / "evaluate_headless_metrics.py"
    subprocess.check_call(
        [
            sys.executable,
            str(runner),
            "--config",
            config_path,
            "--output-dir",
            str(metrics_dir),
        ]
    )
    with open(metrics_dir / "metrics.json", "r", encoding="utf-8") as f:
        return json.load(f)


def check_metrics(metrics: Dict, checks: List[Dict]) -> None:
    for scenario in ("circle", "figure8"):
        data = metrics.get(scenario, {})
        add_check(
            checks,
            "%s.paper_total_violations" % scenario,
            int(data.get("paper_total_violations", -1)) == 0,
            json.dumps(
                {
                    "pursuer_target": data.get("pursuer_target_violations"),
                    "pursuer_pair": data.get("pursuer_pair_violations"),
                    "pursuer_obstacle": data.get("pursuer_obstacle_violations"),
                    "target_pair": data.get("target_pair_violations"),
                    "target_obstacle": data.get("target_obstacle_violations"),
                },
                sort_keys=True,
            ),
        )
        add_check(
            checks,
            "%s.rows" % scenario,
            int(data.get("rows", 0)) == 1200,
            "2 agents x 600 control steps",
        )


def run_gazebo_smoke(config_path: str, output_dir: Path) -> Dict:
    log_path = output_dir / "gazebo_smoke.log"
    cmd = [
        "roslaunch",
        "trust_up_xtdrone",
        "gazebo_paper_replay.launch",
        "scenario:=circle",
        "gui:=false",
        "loop:=false",
        "max_replay_steps:=80",
        "shutdown_when_done:=true",
        "enable_trails:=false",
        "spawn_visual_volumes:=false",
        "verbose:=false",
        "config:=%s" % config_path,
    ]
    env = os.environ.copy()
    env.setdefault("ROS_HOME", str(output_dir / "ros_home"))
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, timeout=90, env=env)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    bad_patterns = [
        "Missing model.config",
        "g/gui-plugin is really loading a SystemPlugin",
        "Traceback",
        "RLException",
        "Gazebo TRUST-UP replay complete",
    ]
    return {
        "returncode": proc.returncode,
        "log": str(log_path),
        "missing_model_config": "Missing model.config" in text,
        "bad_gui_plugin_warning": "g/gui-plugin is really loading a SystemPlugin" in text,
        "traceback": "Traceback" in text,
        "reset_seen": "Gazebo TRUST-UP replay reset" in text,
        "clean_exit": "process has finished cleanly" in text or proc.returncode == 0,
        "completed": "Gazebo TRUST-UP replay complete" in text
        or ("Gazebo TRUST-UP replay reset" in text and "process has finished cleanly" in text and proc.returncode == 0),
    }


def write_markdown(path: Path, checks: List[Dict], metrics: Dict, gazebo: Dict) -> None:
    lines = [
        "# TRUST-UP Paper Conformance Audit",
        "",
        "## Checks",
        "",
    ]
    for check in checks:
        mark = "PASS" if check["passed"] else "FAIL"
        lines.append("- %s `%s`: %s" % (mark, check["name"], check["detail"]))
    if metrics:
        lines.extend(["", "## Metrics", "", "```json", json.dumps(metrics, indent=2, sort_keys=True), "```"])
    if gazebo:
        lines.extend(["", "## Gazebo Smoke", "", "```json", json.dumps(gazebo, indent=2, sort_keys=True), "```"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--output-dir", default="/tmp/trust_up_paper_audit")
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--gazebo-smoke", action="store_true")
    args = parser.parse_args()

    config_path = os.path.abspath(os.path.expanduser(os.path.expandvars(args.config)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checks: List[Dict] = []
    cfg = load_yaml(config_path)
    check_config(cfg, checks)
    check_xtdrone_visual(cfg, checks)

    metrics = {}
    if not args.skip_metrics:
        metrics = run_metrics(config_path, output_dir)
        check_metrics(metrics, checks)

    gazebo = {}
    if args.gazebo_smoke:
        gazebo = run_gazebo_smoke(config_path, output_dir)
        add_check(checks, "gazebo_smoke.returncode", gazebo["returncode"] == 0, "returncode=%s" % gazebo["returncode"])
        add_check(checks, "gazebo_smoke.reset_seen", bool(gazebo["reset_seen"]), "log=%s" % gazebo["log"])
        add_check(checks, "gazebo_smoke.clean_exit", bool(gazebo["clean_exit"]), "log=%s" % gazebo["log"])
        add_check(checks, "gazebo_smoke.completed", bool(gazebo["completed"]), "log=%s" % gazebo["log"])
        add_check(checks, "gazebo_smoke.no_missing_model_config", not gazebo["missing_model_config"], "log=%s" % gazebo["log"])
        add_check(checks, "gazebo_smoke.no_bad_gui_plugin_warning", not gazebo["bad_gui_plugin_warning"], "log=%s" % gazebo["log"])
        add_check(checks, "gazebo_smoke.no_traceback", not gazebo["traceback"], "log=%s" % gazebo["log"])

    passed = all(check["passed"] for check in checks)
    result = {
        "passed": passed,
        "config": config_path,
        "checks": checks,
        "metrics": metrics,
        "gazebo_smoke": gazebo,
    }
    json_path = output_dir / "audit.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(output_dir / "audit.md", checks, metrics, gazebo)
    print(json.dumps(result, indent=2, sort_keys=True))
    print("wrote %s" % json_path)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
