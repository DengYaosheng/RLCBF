#!/usr/bin/env python3
"""Create an Isaac Sim TRUST-UP pursuit scene.

Run with Isaac Sim's Python, for example:

  ./python.sh /path/to/trust_up_xtdrone/isaac/trust_up_isaac_scene.py \
    --config /path/to/trust_up_xtdrone/config/paper_target_pursuit.yaml \
    --isaac-config /path/to/trust_up_xtdrone/config/isaac_sim.yaml

The script creates a metric world with two pursuer UAV placeholders, two target
UAV placeholders, paper obstacles, and ROS-bridge-friendly prim names.  If you
have real USD quadrotor assets, set `scene.pursuer_usd` and `scene.target_usd`
in `config/isaac_sim.yaml`.
"""

import argparse
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PKG_DIR, "src"))

from trust_up_xtdrone.config_io import load_yaml, obstacles_from_config, pursuer_initial_states, resolve_path, target_initial_states


def add_quadrotor_prims(stage, path, position, color):
    from pxr import Gf, UsdGeom

    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))

    body = UsdGeom.Cube.Define(stage, path + "/body")
    body.CreateSizeAttr(1.0)
    body.AddScaleOp().Set(Gf.Vec3f(0.42, 0.24, 0.10))
    body.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])

    arm_x = UsdGeom.Cube.Define(stage, path + "/arm_x")
    arm_x.CreateSizeAttr(1.0)
    arm_x.AddScaleOp().Set(Gf.Vec3f(0.85, 0.04, 0.035))
    arm_x.GetDisplayColorAttr().Set([Gf.Vec3f(0.03, 0.035, 0.04)])

    arm_y = UsdGeom.Cube.Define(stage, path + "/arm_y")
    arm_y.CreateSizeAttr(1.0)
    arm_y.AddScaleOp().Set(Gf.Vec3f(0.04, 0.85, 0.035))
    arm_y.GetDisplayColorAttr().Set([Gf.Vec3f(0.03, 0.035, 0.04)])

    for idx, (x, y) in enumerate(((0.34, 0.34), (0.34, -0.34), (-0.34, 0.34), (-0.34, -0.34))):
        rotor = UsdGeom.Cylinder.Define(stage, "%s/rotor_%d" % (path, idx))
        rotor.CreateRadiusAttr(0.14)
        rotor.CreateHeightAttr(0.025)
        rotor.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.06))
        rotor.GetDisplayColorAttr().Set([Gf.Vec3f(0.02, 0.02, 0.025)])
    return root


def add_usd_reference(stage, path, usd_path, position):
    from pxr import Gf, UsdGeom

    prim = UsdGeom.Xform.Define(stage, path)
    prim.GetPrim().GetReferences().AddReference(resolve_path(usd_path))
    prim.AddTranslateOp().Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    return prim


def add_sphere(stage, path, position, radius, color):
    from pxr import Gf, UsdGeom

    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(float(radius))
    sphere.AddTranslateOp().Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    sphere.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    return sphere


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(PKG_DIR, "config", "paper_target_pursuit.yaml"))
    parser.add_argument("--isaac-config", default=os.path.join(PKG_DIR, "config", "isaac_sim.yaml"))
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    try:
        from omni.isaac.kit import SimulationApp
    except Exception as exc:
        raise SystemExit("This script must be run inside Isaac Sim's Python: %s" % exc)

    simulation_app = SimulationApp({"headless": bool(args.headless)})
    from omni.isaac.core import World
    from pxr import Gf, UsdGeom

    cfg = load_yaml(args.config)
    isaac_cfg = load_yaml(args.isaac_config)
    scene_cfg = isaac_cfg.get("scene", {})
    stage_path = resolve_path(scene_cfg.get("stage_path", "/tmp/trust_up_target_pursuit.usd"))

    world = World(stage_units_in_meters=float(scene_cfg.get("meters_per_unit", 1.0)))
    world.scene.add_default_ground_plane()
    stage = world.stage
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    pursuer_usd = scene_cfg.get("pursuer_usd", "")
    target_usd = scene_cfg.get("target_usd", "")
    for idx, state in enumerate(pursuer_initial_states(cfg)):
        path = "/World/iris_%d" % idx
        if pursuer_usd:
            add_usd_reference(stage, path, pursuer_usd, state.position)
        else:
            add_quadrotor_prims(stage, path, state.position, (1.0, 0.18, 0.14))

    for idx, state in enumerate(target_initial_states(cfg)):
        path = "/World/trust_target_%d" % idx
        if target_usd:
            add_usd_reference(stage, path, target_usd, state.position)
        else:
            add_quadrotor_prims(stage, path, state.position, (0.12, 0.42, 1.0))

    for obstacle in obstacles_from_config(cfg):
        add_sphere(stage, "/World/%s" % obstacle.name, obstacle.position, obstacle.radius, (0.12, 0.12, 0.14))

    light = UsdGeom.Sphere.Define(stage, "/World/lighting_anchor")
    light.CreateRadiusAttr(0.05)
    light.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 12.0))

    stage.GetRootLayer().Export(stage_path)
    print("wrote Isaac USD scene: %s" % stage_path)
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
