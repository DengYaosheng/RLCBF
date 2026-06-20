#!/usr/bin/env python3
"""Spawn paper obstacles and kinematic target visuals into Gazebo."""

import os
import sys
import math
import copy
import glob
import xml.etree.ElementTree as ET

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PKG_DIR, "src"))

import rospy  # noqa: E402
from geometry_msgs.msg import Point, Pose, Quaternion  # noqa: E402

from trust_up_xtdrone.config_io import (  # noqa: E402
    default_config_path,
    load_yaml,
    obstacles_from_config,
    scenario_offset,
    target_initial_states,
)
from trust_up_xtdrone.core import AgentState, TargetTrajectory  # noqa: E402


def sphere_sdf(name: str, radius: float, rgba: str, static: bool, *, visual_only: bool = False, transparency: float = 0.0) -> str:
    static_text = "true" if static else "false"
    collision = "" if visual_only else """
      <collision name="collision">
        <geometry><sphere><radius>{radius}</radius></sphere></geometry>
      </collision>""".format(radius=float(radius))
    return """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{name}">
    <static>{static}</static>
    <link name="link">
      <gravity>false</gravity>
      <kinematic>true</kinematic>
      <inertial><mass>0.05</mass><inertia><ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz></inertia></inertial>
      <visual name="visual">
        <geometry><sphere><radius>{radius}</radius></sphere></geometry>
        <transparency>{transparency}</transparency>
        <material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse></material>
      </visual>
      {collision}
    </link>
  </model>
</sdf>""".format(name=name, radius=float(radius), rgba=rgba, static=static_text, transparency=float(transparency), collision=collision)


def xtdrone_visual_uav_sdf(name: str, source_sdf: str) -> str:
    """Load XTDrone UAV visuals and collapse them into one kinematic link."""
    source_sdf = os.path.expanduser(os.path.expandvars(source_sdf))
    tree = ET.parse(source_sdf)
    root = tree.getroot()
    source_model = root.find("model")
    if source_model is None:
        raise RuntimeError("XTDrone SDF has no model element: %s" % source_sdf)
    model_root = os.path.dirname(os.path.dirname(source_sdf))
    model_search_roots = [
        model_root,
        "/home/ysdeng/work/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models",
        "/home/ysdeng/work/PX4-Autopilot/Tools/simulation/gz/models",
        "/home/ysdeng/projects/XTDrone/sitl_config/robotic_arm/le_arm/meshes",
    ]
    for entry in os.environ.get("GAZEBO_MODEL_PATH", "").split(":"):
        if entry and entry not in model_search_roots:
            model_search_roots.append(entry)
    model_search_roots = [root_path for root_path in model_search_roots if os.path.isdir(root_path)]

    def parse_pose(text: str):
        values = [float(v) for v in (text or "0 0 0 0 0 0").split()]
        values = (values + [0.0] * 6)[:6]
        return values

    def pose_text(values):
        return " ".join("%.9g" % float(v) for v in values)

    def resolve_model_uri(text: str):
        if not text.startswith("model://"):
            return text
        rest = text[len("model://") :]
        parts = rest.split("/", 1)
        if len(parts) != 2:
            return None
        model_name, subpath = parts
        basename = os.path.basename(subpath)
        candidates = []
        for root_path in model_search_roots:
            candidates.append(os.path.join(root_path, model_name, subpath))
            candidates.append(os.path.join(root_path, "iris", subpath))
            candidates.extend(glob.glob(os.path.join(root_path, "*", subpath)))
            if basename:
                candidates.extend(glob.glob(os.path.join(root_path, "*", "meshes", basename)))
                candidates.extend(glob.glob(os.path.join(root_path, "**", "meshes", basename), recursive=True))
        for candidate in candidates:
            if os.path.exists(candidate):
                return "file://%s" % os.path.abspath(candidate)
        return None

    def rewrite_model_uris(visual) -> bool:
        for uri in visual.findall(".//uri"):
            text = (uri.text or "").strip()
            if not text.startswith("model://"):
                continue
            resolved = resolve_model_uri(text)
            if not resolved:
                return False
            uri.text = resolved
        return True

    def add_material(visual):
        material = ET.SubElement(visual, "material")
        ET.SubElement(material, "ambient").text = "0.75 0.75 0.75 1"
        ET.SubElement(material, "diffuse").text = "0.75 0.75 0.75 1"

    def add_box_visual(name_text, pose_text_value, size_text):
        visual = ET.SubElement(link, "visual", {"name": name_text})
        ET.SubElement(visual, "pose").text = pose_text_value
        geometry = ET.SubElement(visual, "geometry")
        box = ET.SubElement(geometry, "box")
        ET.SubElement(box, "size").text = size_text
        add_material(visual)

    def add_cylinder_visual(name_text, pose_text_value, radius_text, length_text):
        visual = ET.SubElement(link, "visual", {"name": name_text})
        ET.SubElement(visual, "pose").text = pose_text_value
        geometry = ET.SubElement(visual, "geometry")
        cylinder = ET.SubElement(geometry, "cylinder")
        ET.SubElement(cylinder, "radius").text = radius_text
        ET.SubElement(cylinder, "length").text = length_text
        add_material(visual)

    def add_fallback_quadrotor_visuals():
        add_box_visual("fallback_body", "0 0 0 0 0 0", "0.42 0.24 0.11")
        add_box_visual("fallback_nose", "0.25 0 0.02 0 0 0", "0.18 0.09 0.06")
        add_box_visual("fallback_arm_x", "0 0 0 0 0 0", "0.82 0.045 0.04")
        add_box_visual("fallback_arm_y", "0 0 0 0 0 1.57079632679", "0.82 0.045 0.04")
        for idx, (x, y) in enumerate(((0.36, 0.36), (-0.36, 0.36), (-0.36, -0.36), (0.36, -0.36))):
            add_cylinder_visual("fallback_rotor_%d" % idx, "%.3f %.3f 0.025 0 0 0" % (x, y), "0.145", "0.018")

    new_root = ET.Element("sdf", {"version": "1.6"})
    model = ET.SubElement(new_root, "model", {"name": name})
    ET.SubElement(model, "static").text = "false"
    link = ET.SubElement(model, "link", {"name": "base_link"})
    ET.SubElement(link, "gravity").text = "false"
    ET.SubElement(link, "self_collide").text = "false"
    ET.SubElement(link, "kinematic").text = "true"
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass").text = "0.05"
    inertia = ET.SubElement(inertial, "inertia")
    ET.SubElement(inertia, "ixx").text = "0.001"
    ET.SubElement(inertia, "iyy").text = "0.001"
    ET.SubElement(inertia, "izz").text = "0.001"
    ET.SubElement(inertia, "ixy").text = "0.0"
    ET.SubElement(inertia, "ixz").text = "0.0"
    ET.SubElement(inertia, "iyz").text = "0.0"

    visual_id = 0
    for source_link in source_model.findall("link"):
        link_name = source_link.get("name", "link")
        link_pose = parse_pose(source_link.findtext("pose"))
        for source_visual in source_link.findall("visual"):
            visual = copy.deepcopy(source_visual)
            visual.set("name", "%s_%s_%03d" % (link_name, source_visual.get("name", "visual"), visual_id))
            visual_id += 1
            visual_pose = parse_pose(visual.findtext("pose"))
            combined_pose = [link_pose[i] + visual_pose[i] for i in range(6)]
            pose = visual.find("pose")
            if pose is None:
                pose = ET.Element("pose")
                visual.insert(0, pose)
            pose.text = pose_text(combined_pose)
            if rewrite_model_uris(visual):
                link.append(visual)
            else:
                visual_id -= 1

    if visual_id == 0:
        add_fallback_quadrotor_visuals()
    return ET.tostring(new_root, encoding="unicode")


def wire_shell_sdf(name: str, radius: float, rgba: str, static: bool, *, tube_radius: float = 0.012, segments: int = 48) -> str:
    static_text = "true" if static else "false"
    seg_count = max(int(segments), 12)
    seg_len = 2.0 * float(radius) * math.sin(math.pi / seg_count)
    links = []

    def add_segment(link_name: str, x: float, y: float, z: float, roll: float, pitch: float, yaw: float):
        links.append(
            """
    <link name="{link_name}">
      <pose>{x:.6f} {y:.6f} {z:.6f} {roll:.6f} {pitch:.6f} {yaw:.6f}</pose>
      <gravity>false</gravity>
      <inertial><mass>0.001</mass><inertia><ixx>0.000001</ixx><iyy>0.000001</iyy><izz>0.000001</izz></inertia></inertial>
      <visual name="visual">
        <geometry><cylinder><radius>{tube_radius:.6f}</radius><length>{seg_len:.6f}</length></cylinder></geometry>
        <material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse></material>
      </visual>
    </link>""".format(
                link_name=link_name,
                x=x,
                y=y,
                z=z,
                roll=roll,
                pitch=pitch,
                yaw=yaw,
                tube_radius=float(tube_radius),
                seg_len=seg_len,
                rgba=rgba,
            )
        )

    for idx in range(seg_count):
        theta = 2.0 * math.pi * (idx + 0.5) / seg_count
        x = float(radius) * math.cos(theta)
        y = float(radius) * math.sin(theta)
        z = float(radius) * math.sin(theta)

        tangent_xy = theta + math.pi / 2.0
        add_segment("xy_%02d" % idx, x, y, 0.0, 0.0, math.pi / 2.0, tangent_xy)
        add_segment("xz_%02d" % idx, x, 0.0, z, 0.0, -theta, 0.0)
        add_segment("yz_%02d" % idx, 0.0, x, z, theta, 0.0, 0.0)

    return """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{name}">
    <static>{static}</static>{links}
  </model>
</sdf>""".format(name=name, static=static_text, links="".join(links))


def spawn_model(spawn_srv, delete_srv, name: str, sdf: str, xyz, existing_names=None):
    if existing_names is None or name in existing_names:
        try:
            delete_srv(name)
        except Exception:
            pass
        if existing_names is not None:
            existing_names.discard(name)
    pose = Pose()
    pose.position = Point(float(xyz[0]), float(xyz[1]), float(xyz[2]))
    pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
    spawn_srv(name, sdf, "", pose, "world")
    if existing_names is not None:
        existing_names.add(name)


def main():
    rospy.init_node("spawn_paper_world")
    cfg_path = rospy.get_param("~config_file", default_config_path())
    cfg = load_yaml(cfg_path)
    if rospy.has_param("~scenario"):
        cfg.setdefault("scenario", {})["name"] = rospy.get_param("~scenario")

    from gazebo_msgs.srv import DeleteModel, GetWorldProperties, SpawnModel

    rospy.wait_for_service("/gazebo/spawn_sdf_model")
    spawn_srv = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
    delete_srv = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
    existing_names = set()
    try:
        rospy.wait_for_service("/gazebo/get_world_properties", timeout=2.0)
        world_srv = rospy.ServiceProxy("/gazebo/get_world_properties", GetWorldProperties)
        existing_names = set(world_srv().model_names)
    except Exception:
        existing_names = set()

    for obs in obstacles_from_config(cfg):
        spawn_model(spawn_srv, delete_srv, obs.name, sphere_sdf(obs.name, obs.radius, "0.02 0.02 0.02 1.0", True), obs.position, existing_names)

    obstacles = obstacles_from_config(cfg)
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
    targets = target_initial_states(cfg)
    if bool(cfg.get("scenario", {}).get("initialize_targets_from_reference", True)):
        targets = [
            AgentState.from_xyz(
                trajectory.reference(i, 0.0)[0],
                trajectory.reference(i, 0.0)[1],
                trajectory.reference(i, 0.0)[2],
                radius=target.radius,
                name=target.name,
            )
            for i, target in enumerate(targets)
        ]

    target_visual = cfg.get("target_visual", {})
    source_sdf = target_visual.get(
        "xtdrone_sdf",
        "/home/ysdeng/projects/XTDrone/sitl_config/models/iris_zhihang/iris_zhihang.sdf",
    )
    use_object_radius = bool(cfg.get("safety", {}).get("use_object_radius", False))
    pursuer_radius = float(cfg.get("physical_envelope", {}).get("uav_radius", 0.0)) if use_object_radius else 0.0
    collision_radius = float(cfg.get("safety", {}).get("collision_radius", 0.5))
    sensing_radius = float(cfg.get("safety", {}).get("sensing_radius", 1.0))
    for idx, target in enumerate(targets):
        shell_padding = pursuer_radius + (float(target.radius) if use_object_radius else 0.0)
        target_model_sdf = xtdrone_visual_uav_sdf(target.name, source_sdf)
        spawn_model(
            spawn_srv,
            delete_srv,
            target.name,
            target_model_sdf,
            target.position,
            existing_names,
        )
        spawn_model(
            spawn_srv,
            delete_srv,
            "%s_collision_shell" % target.name,
            wire_shell_sdf("%s_collision_shell" % target.name, collision_radius + shell_padding, "1.0 0.62 0.04 1.0", False, tube_radius=0.014),
            target.position,
            existing_names,
        )
        spawn_model(
            spawn_srv,
            delete_srv,
            "%s_sensing_shell" % target.name,
            wire_shell_sdf("%s_sensing_shell" % target.name, sensing_radius + shell_padding, "1.0 0.95 0.15 1.0", False, tube_radius=0.01),
            target.position,
            existing_names,
        )
    rospy.loginfo("Spawned TRUST-UP paper obstacles and targets")


if __name__ == "__main__":
    main()
