#!/usr/bin/env python3
"""Publish and optionally move the two paper target UAVs in Gazebo."""

import json
import math
import os
import sys
from typing import List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PKG_DIR, "src"))

import numpy as np  # noqa: E402
import rospy  # noqa: E402
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Vector3  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from std_msgs.msg import ColorRGBA, Header, String  # noqa: E402
from visualization_msgs.msg import Marker, MarkerArray  # noqa: E402

from trust_up_xtdrone.config_io import (  # noqa: E402
    default_config_path,
    load_yaml,
    obstacles_from_config,
    pursuer_initial_states,
    scenario_offset,
    target_initial_states,
)
from trust_up_xtdrone.core import AgentState, TargetTrajectory  # noqa: E402


def yaw_from_velocity(velocity) -> float:
    vx, vy = float(velocity[0]), float(velocity[1])
    if math.hypot(vx, vy) < 1.0e-4:
        return 0.0
    return math.atan2(vy, vx)


def quaternion_from_yaw(yaw: float) -> Quaternion:
    return Quaternion(0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


def rotate_xy(offset, yaw: float):
    c = math.cos(yaw)
    s = math.sin(yaw)
    x, y, z = [float(v) for v in offset]
    return [c * x - s * y, s * x + c * y, z]


def vec_from_point(point) -> np.ndarray:
    return np.array([point.x, point.y, point.z], dtype=float)


def make_odom(state: AgentState, frame_id: str) -> Odometry:
    msg = Odometry()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.child_frame_id = state.name
    msg.pose.pose.position = Point(*[float(x) for x in state.position])
    msg.pose.pose.orientation = quaternion_from_yaw(yaw_from_velocity(state.velocity))
    msg.twist.twist.linear.x = float(state.velocity[0])
    msg.twist.twist.linear.y = float(state.velocity[1])
    msg.twist.twist.linear.z = float(state.velocity[2])
    return msg


def sphere_marker(
    marker_id: int,
    name: str,
    position,
    radius: float,
    color,
    frame_id: str,
    *,
    alpha: float = 0.35,
) -> Marker:
    marker = Marker()
    marker.header = Header(stamp=rospy.Time.now(), frame_id=frame_id)
    marker.ns = name
    marker.id = marker_id
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position = Point(*[float(x) for x in position])
    marker.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
    diameter = 2.0 * float(radius)
    marker.scale = Vector3(diameter, diameter, diameter)
    marker.color = ColorRGBA(float(color[0]), float(color[1]), float(color[2]), alpha)
    marker.lifetime = rospy.Duration(0.3)
    return marker


def delete_all_marker(frame_id: str) -> Marker:
    marker = Marker()
    marker.header = Header(stamp=rospy.Time.now(), frame_id=frame_id)
    marker.action = Marker.DELETEALL
    return marker


def cube_marker(
    marker_id: int,
    name: str,
    position,
    size,
    color,
    frame_id: str,
    yaw: float,
    *,
    offset=(0.0, 0.0, 0.0),
    alpha: float = 0.95,
) -> Marker:
    marker = Marker()
    marker.header = Header(stamp=rospy.Time.now(), frame_id=frame_id)
    marker.ns = name
    marker.id = marker_id
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    shifted = rotate_xy(offset, yaw)
    marker.pose.position = Point(
        float(position[0]) + shifted[0],
        float(position[1]) + shifted[1],
        float(position[2]) + shifted[2],
    )
    marker.pose.orientation = quaternion_from_yaw(yaw)
    marker.scale = Vector3(float(size[0]), float(size[1]), float(size[2]))
    marker.color = ColorRGBA(float(color[0]), float(color[1]), float(color[2]), alpha)
    marker.lifetime = rospy.Duration(0.3)
    return marker


def rotor_marker(
    marker_id: int,
    name: str,
    position,
    offset,
    color,
    frame_id: str,
    yaw: float,
) -> Marker:
    marker = Marker()
    marker.header = Header(stamp=rospy.Time.now(), frame_id=frame_id)
    marker.ns = name
    marker.id = marker_id
    marker.type = Marker.CYLINDER
    marker.action = Marker.ADD
    shifted = rotate_xy(offset, yaw)
    marker.pose.position = Point(
        float(position[0]) + shifted[0],
        float(position[1]) + shifted[1],
        float(position[2]) + shifted[2],
    )
    marker.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
    marker.scale = Vector3(0.20, 0.20, 0.025)
    marker.color = ColorRGBA(float(color[0]), float(color[1]), float(color[2]), 0.88)
    marker.lifetime = rospy.Duration(0.3)
    return marker


def arrow_marker(
    marker_id: int,
    name: str,
    position,
    yaw: float,
    color,
    frame_id: str,
) -> Marker:
    marker = Marker()
    marker.header = Header(stamp=rospy.Time.now(), frame_id=frame_id)
    marker.ns = name
    marker.id = marker_id
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    marker.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
    start = Point(float(position[0]), float(position[1]), float(position[2]) + 0.11)
    end = Point(
        float(position[0]) + 0.62 * math.cos(yaw),
        float(position[1]) + 0.62 * math.sin(yaw),
        float(position[2]) + 0.11,
    )
    marker.points = [start, end]
    marker.scale = Vector3(0.035, 0.085, 0.12)
    marker.color = ColorRGBA(float(color[0]), float(color[1]), float(color[2]), 0.88)
    marker.lifetime = rospy.Duration(0.3)
    return marker


def append_uav_markers(
    arr: MarkerArray,
    marker_id: int,
    name: str,
    state: AgentState,
    color,
    frame_id: str,
    *,
    alpha: float = 0.96,
) -> int:
    yaw = yaw_from_velocity(state.velocity)
    dark = [0.035, 0.04, 0.05]
    accent = [min(color[0] + 0.35, 1.0), min(color[1] + 0.25, 1.0), min(color[2] + 0.25, 1.0)]
    arr.markers.append(cube_marker(marker_id, "%s_shadow" % name, state.position, [0.72, 0.50, 0.012], [0.0, 0.0, 0.0], frame_id, yaw, offset=[0.0, 0.0, -0.11], alpha=0.18))
    marker_id += 1
    arr.markers.append(cube_marker(marker_id, "%s_body" % name, state.position, [0.38, 0.22, 0.09], color, frame_id, yaw, alpha=alpha))
    marker_id += 1
    arr.markers.append(cube_marker(marker_id, "%s_nose" % name, state.position, [0.16, 0.08, 0.05], accent, frame_id, yaw, offset=[0.24, 0.0, 0.01], alpha=alpha))
    marker_id += 1
    arr.markers.append(cube_marker(marker_id, "%s_arm_x" % name, state.position, [0.72, 0.035, 0.035], dark, frame_id, yaw))
    marker_id += 1
    arr.markers.append(cube_marker(marker_id, "%s_arm_y" % name, state.position, [0.72, 0.035, 0.035], dark, frame_id, yaw + math.pi / 2.0))
    marker_id += 1
    for rotor_id, offset in enumerate(([0.30, 0.30, 0.04], [0.30, -0.30, 0.04], [-0.30, 0.30, 0.04], [-0.30, -0.30, 0.04])):
        arr.markers.append(rotor_marker(marker_id, "%s_rotor_%d" % (name, rotor_id), state.position, offset, dark, frame_id, yaw))
        marker_id += 1
    arr.markers.append(arrow_marker(marker_id, "%s_heading" % name, state.position, yaw, accent, frame_id))
    marker_id += 1
    return marker_id


def ring_marker(
    marker_id: int,
    name: str,
    position,
    radius: float,
    color,
    frame_id: str,
    plane: str,
    *,
    alpha: float = 0.85,
    width: float = 0.018,
    segments: int = 96,
) -> Marker:
    marker = Marker()
    marker.header = Header(stamp=rospy.Time.now(), frame_id=frame_id)
    marker.ns = name
    marker.id = marker_id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
    marker.scale.x = float(width)
    marker.color = ColorRGBA(float(color[0]), float(color[1]), float(color[2]), alpha)
    cx, cy, cz = [float(x) for x in position]
    for idx in range(int(segments) + 1):
        theta = 2.0 * math.pi * idx / float(segments)
        c = float(radius) * math.cos(theta)
        s = float(radius) * math.sin(theta)
        if plane == "xy":
            marker.points.append(Point(cx + c, cy + s, cz))
        elif plane == "xz":
            marker.points.append(Point(cx + c, cy, cz + s))
        else:
            marker.points.append(Point(cx, cy + c, cz + s))
    marker.lifetime = rospy.Duration(0.3)
    return marker


def line_marker(
    marker_id: int,
    name: str,
    points,
    color,
    frame_id: str,
    *,
    alpha: float = 0.86,
    width: float = 0.028,
) -> Marker:
    marker = Marker()
    marker.header = Header(stamp=rospy.Time.now(), frame_id=frame_id)
    marker.ns = name
    marker.id = marker_id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
    marker.scale.x = float(width)
    marker.color = ColorRGBA(float(color[0]), float(color[1]), float(color[2]), alpha)
    marker.points = [Point(*[float(v) for v in p]) for p in points]
    marker.lifetime = rospy.Duration(0.3)
    return marker


def trail_marker(marker_id: int, name: str, points, color, frame_id: str, width: float = 0.035) -> Marker:
    marker = line_marker(marker_id, name, points, color, frame_id, alpha=0.76, width=width)
    marker.lifetime = rospy.Duration(0.7)
    return marker


def text_marker(
    marker_id: int,
    name: str,
    text: str,
    position,
    color,
    frame_id: str,
    *,
    scale: float = 0.22,
) -> Marker:
    marker = Marker()
    marker.header = Header(stamp=rospy.Time.now(), frame_id=frame_id)
    marker.ns = name
    marker.id = marker_id
    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD
    marker.pose.position = Point(*[float(v) for v in position])
    marker.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
    marker.scale.z = float(scale)
    marker.color = ColorRGBA(float(color[0]), float(color[1]), float(color[2]), 0.96)
    marker.text = text
    marker.lifetime = rospy.Duration(0.3)
    return marker


class PaperTargetServer:
    def __init__(self):
        cfg_path = rospy.get_param("~config_file", default_config_path())
        self.cfg = load_yaml(cfg_path)
        if rospy.has_param("~scenario"):
            self.cfg.setdefault("scenario", {})["name"] = rospy.get_param("~scenario")

        self.frame_id = rospy.get_param("~frame_id", "map")
        self.dt = float(self.cfg.get("paper", {}).get("dt", 0.1))
        self.obstacles = obstacles_from_config(self.cfg)
        self.targets: List[AgentState] = target_initial_states(self.cfg)
        self.pursuers: List[AgentState] = pursuer_initial_states(self.cfg)
        self.trajectory = TargetTrajectory(
            self.cfg.get("scenario", {}).get("name", "circle"),
            self.obstacles,
            offset=scenario_offset(self.cfg),
            potential_gain=float(self.cfg.get("scenario", {}).get("target_potential_gain", 1.0)),
            potential_clip=float(self.cfg.get("scenario", {}).get("target_potential_clip", 1.5)),
            use_dimension_consistent_reference_velocity=bool(
                self.cfg.get("scenario", {}).get("use_dimension_consistent_reference_velocity", True)
            ),
            reference_scale=float(self.cfg.get("scenario", {}).get("reference_scale", 1.0)),
            angular_rate=float(self.cfg.get("scenario", {}).get("angular_rate", 0.1)),
            smoothing=self.cfg.get("trajectory_smoothing", {}),
        )
        if bool(self.cfg.get("scenario", {}).get("initialize_targets_from_reference", True)):
            self.targets = [
                AgentState.from_xyz(
                    self.trajectory.reference(i, 0.0)[0],
                    self.trajectory.reference(i, 0.0)[1],
                    self.trajectory.reference(i, 0.0)[2],
                    radius=target.radius,
                    name=target.name,
                )
                for i, target in enumerate(self.targets)
            ]
        self.trails = [[] for _ in self.targets]
        self.pursuer_trails = [[] for _ in self.pursuers]
        self.max_trail_points = int(rospy.get_param("~max_trail_points", 450))
        self.stale_timeout = float(self.cfg.get("xtdrone", {}).get("stale_timeout_s", 0.7))
        xt = self.cfg.get("xtdrone", {})
        self.uav_type = str(rospy.get_param("~uav_type", xt.get("uav_type", "iris")))
        self.pursuer_ids = list(rospy.get_param("~pursuer_ids", xt.get("pursuer_ids", [0, 1])))
        for idx, state in enumerate(self.pursuers):
            if not state.name:
                vehicle_id = int(self.pursuer_ids[idx]) if idx < len(self.pursuer_ids) else idx
                state.name = "%s_%d" % (self.uav_type, vehicle_id)
            rospy.Subscriber("/%s/mavros/local_position/odom" % state.name, Odometry, self.make_pursuer_odom_callback(idx), queue_size=1)
            rospy.Subscriber("/%s/mavros/local_position/pose" % state.name, PoseStamped, self.make_pursuer_pose_callback(idx), queue_size=1)
        topic_prefix = self.cfg.get("topics", {}).get("target_odom_prefix", "/trust_up")
        self.odom_pubs = [
            rospy.Publisher("%s/target_%d/odom" % (topic_prefix, i), Odometry, queue_size=1)
            for i in range(len(self.targets))
        ]
        self.marker_pub = rospy.Publisher(self.cfg.get("topics", {}).get("markers", "/trust_up/markers"), MarkerArray, queue_size=1)
        self.diag_pub = rospy.Publisher("/trust_up/target_server/status", String, queue_size=1)
        self.use_gazebo = bool(rospy.get_param("~set_gazebo_model_state", True))
        self.gazebo_move_start_delay_s = float(rospy.get_param("~gazebo_move_start_delay_s", 5.0))
        self.trajectory_start_delay_s = float(rospy.get_param("~trajectory_start_delay_s", 0.0))
        self.set_model_state = None
        if self.use_gazebo:
            try:
                from gazebo_msgs.msg import ModelState
                from gazebo_msgs.srv import SetModelState

                self.ModelState = ModelState
                rospy.wait_for_service("/gazebo/set_model_state", timeout=2.0)
                self.set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
            except Exception as exc:
                rospy.logwarn("Gazebo set_model_state unavailable: %s", exc)
                self.set_model_state = None

        self.start_time = rospy.Time.now().to_sec()

    def make_pursuer_odom_callback(self, idx: int):
        def callback(msg: Odometry):
            now = msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec()
            previous = self.pursuers[idx]
            self.pursuers[idx] = AgentState(
                vec_from_point(msg.pose.pose.position),
                vec_from_point(msg.twist.twist.linear),
                radius=previous.radius,
                stamp=now,
                name=previous.name,
            )

        return callback

    def make_pursuer_pose_callback(self, idx: int):
        def callback(msg: PoseStamped):
            now = msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec()
            previous = self.pursuers[idx]
            pos = vec_from_point(msg.pose.position)
            vel = np.zeros(3)
            if previous.stamp > 0.0 and now > previous.stamp:
                vel = (pos - previous.position) / max(now - previous.stamp, 1.0e-3)
            self.pursuers[idx] = AgentState(pos, vel, radius=previous.radius, stamp=now, name=previous.name)

        return callback

    def state_is_fresh(self, state: AgentState) -> bool:
        return state.stamp > 0.0 and rospy.Time.now().to_sec() - float(state.stamp) <= self.stale_timeout

    def append_trail_point(self, trail, position):
        pos = np.asarray(position, dtype=float).reshape(3)
        if not trail or float(np.linalg.norm(pos - np.asarray(trail[-1], dtype=float))) > 0.025:
            trail.append(pos.copy())
        if len(trail) > self.max_trail_points:
            del trail[:-self.max_trail_points]

    def publish_markers(self):
        arr = MarkerArray()
        arr.markers.append(delete_all_marker(self.frame_id))
        marker_id = 0
        for obs in self.obstacles:
            arr.markers.append(sphere_marker(marker_id, obs.name, obs.position, obs.radius, [0.12, 0.12, 0.14], self.frame_id, alpha=0.72))
            marker_id += 1
            arr.markers.append(text_marker(marker_id, "%s_label" % obs.name, "obs r=%.2f" % obs.radius, obs.position + np.array([0.0, 0.0, obs.radius + 0.22]), [0.86, 0.86, 0.86], self.frame_id, scale=0.16))
            marker_id += 1
        use_object_radius = bool(self.cfg.get("safety", {}).get("use_object_radius", False))
        pursuer_radius = float(self.cfg.get("physical_envelope", {}).get("uav_radius", 0.0)) if use_object_radius else 0.0
        base_safe_r = float(self.cfg.get("safety", {}).get("collision_radius", 0.5))
        base_sense_r = float(self.cfg.get("safety", {}).get("sensing_radius", 1.0))
        inner_margin = max(float(self.cfg.get("safety", {}).get("tracking_inner_margin", 0.0)), 0.0)
        outer_margin = max(float(self.cfg.get("safety", {}).get("tracking_outer_margin", 0.0)), 0.0)
        target_colors = [[0.14, 0.44, 1.0], [0.0, 0.74, 0.60]]
        pursuer_colors = [[1.0, 0.22, 0.18], [0.92, 0.22, 0.88]]
        inner_color = [1.0, 0.36, 0.16]
        outer_color = [0.0, 0.78, 1.0]
        guard_inner_color = [0.98, 0.92, 0.20]
        guard_outer_color = [0.15, 1.0, 0.56]
        target_pair_clearance = float(self.cfg.get("trajectory_smoothing", {}).get("target_pair_clearance", 0.0))
        target_pair_activation = target_pair_clearance + float(self.cfg.get("trajectory_smoothing", {}).get("target_pair_activation_margin", 0.0))
        for i, target in enumerate(self.targets):
            shell_padding = pursuer_radius + (float(target.radius) if use_object_radius else 0.0)
            safe_r = base_safe_r + shell_padding
            sense_r = base_sense_r + shell_padding
            guard_safe_r = safe_r + inner_margin
            guard_sense_r = max(sense_r - outer_margin, guard_safe_r + 0.05)
            color = target_colors[i % len(target_colors)]
            marker_id = append_uav_markers(arr, marker_id, "target_%d" % i, target, color, self.frame_id)
            arr.markers.append(sphere_marker(marker_id, "target_%d_collision_fill" % i, target.position, safe_r, inner_color, self.frame_id, alpha=0.035))
            marker_id += 1
            arr.markers.append(sphere_marker(marker_id, "target_%d_sensing_fill" % i, target.position, sense_r, outer_color, self.frame_id, alpha=0.025))
            marker_id += 1
            for plane in ("xy", "xz", "yz"):
                arr.markers.append(ring_marker(marker_id, "target_%d_collision_%s" % (i, plane), target.position, safe_r, inner_color, self.frame_id, plane, alpha=0.95, width=0.024))
                marker_id += 1
                arr.markers.append(ring_marker(marker_id, "target_%d_sensing_%s" % (i, plane), target.position, sense_r, outer_color, self.frame_id, plane, alpha=0.74, width=0.018))
                marker_id += 1
                arr.markers.append(ring_marker(marker_id, "target_%d_guard_collision_%s" % (i, plane), target.position, guard_safe_r, guard_inner_color, self.frame_id, plane, alpha=0.62, width=0.012))
                marker_id += 1
                arr.markers.append(ring_marker(marker_id, "target_%d_guard_sensing_%s" % (i, plane), target.position, guard_sense_r, guard_outer_color, self.frame_id, plane, alpha=0.55, width=0.011))
                marker_id += 1
            if len(self.trails[i]) >= 2:
                arr.markers.append(trail_marker(marker_id, "target_%d_trail" % i, self.trails[i], color, self.frame_id, width=0.038))
                marker_id += 1
            label_pos = target.position + np.array([0.0, 0.0, sense_r + 0.28])
            arr.markers.append(
                text_marker(
                    marker_id,
                    "target_%d_label" % i,
                    "T%d target\npaper clear [%.2f, %.2f]\nguard clear [%.2f, %.2f]" % (
                        i,
                        base_safe_r,
                        base_sense_r,
                        base_safe_r + inner_margin,
                        base_sense_r - outer_margin,
                    ),
                    label_pos,
                    color,
                    self.frame_id,
                    scale=0.20,
                )
            )
            marker_id += 1

        for i, pursuer in enumerate(self.pursuers):
            fresh = self.state_is_fresh(pursuer)
            if fresh:
                self.append_trail_point(self.pursuer_trails[i], pursuer.position)
            color = pursuer_colors[i % len(pursuer_colors)] if fresh else [0.55, 0.55, 0.58]
            marker_id = append_uav_markers(arr, marker_id, "pursuer_%d" % i, pursuer, color, self.frame_id, alpha=0.94 if fresh else 0.45)
            if len(self.pursuer_trails[i]) >= 2:
                arr.markers.append(trail_marker(marker_id, "pursuer_%d_trail" % i, self.pursuer_trails[i], color, self.frame_id, width=0.042))
                marker_id += 1
            label = "P%d %s" % (i, pursuer.name)
            if not fresh:
                label += "\nwaiting for MAVROS odom"
            arr.markers.append(text_marker(marker_id, "pursuer_%d_label" % i, label, pursuer.position + np.array([0.0, 0.0, 0.55]), color, self.frame_id, scale=0.20))
            marker_id += 1

        for i in range(len(self.targets)):
            for j in range(i + 1, len(self.targets)):
                first = self.targets[i]
                second = self.targets[j]
                center_distance = float(np.linalg.norm(first.position - second.position))
                hull_clearance = center_distance - float(first.radius) - float(second.radius)
                if hull_clearance < target_pair_clearance:
                    state = "TARGET COLLISION RISK"
                    color = [1.0, 0.10, 0.06]
                elif hull_clearance < target_pair_activation:
                    state = "TARGET GUARD"
                    color = [1.0, 0.72, 0.08]
                else:
                    state = "TARGET OK"
                    color = [0.18, 0.80, 1.0]
                arr.markers.append(line_marker(marker_id, "target_pair_%d_%d" % (i, j), [first.position, second.position], color, self.frame_id, alpha=0.72, width=0.026))
                marker_id += 1
                mid = 0.5 * (first.position + second.position) + np.array([0.0, 0.0, 0.36])
                arr.markers.append(
                    text_marker(
                        marker_id,
                        "target_pair_%d_%d_label" % (i, j),
                        "T%d-T%d %s\nhull clear=%.2f m, min %.2f" % (i, j, state, hull_clearance, target_pair_clearance),
                        mid,
                        color,
                        self.frame_id,
                        scale=0.17,
                    )
                )
                marker_id += 1

        for i, pursuer in enumerate(self.pursuers):
            if i >= len(self.targets):
                continue
            target = self.targets[i]
            radius_sum = (float(pursuer.radius) + float(target.radius)) if use_object_radius else 0.0
            center_distance = float(np.linalg.norm(pursuer.position - target.position))
            paper_clearance = center_distance - radius_sum
            if paper_clearance < base_safe_r:
                state = "TOO CLOSE"
                color = [1.0, 0.12, 0.08]
            elif paper_clearance > base_sense_r:
                state = "TOO FAR"
                color = [1.0, 0.66, 0.05]
            elif paper_clearance < base_safe_r + inner_margin or paper_clearance > base_sense_r - outer_margin:
                state = "GUARD"
                color = [1.0, 0.76, 0.10]
            else:
                state = "OK"
                color = [0.12, 0.88, 0.35]
            arr.markers.append(line_marker(marker_id, "pair_%d_range" % i, [pursuer.position, target.position], color, self.frame_id, alpha=0.88, width=0.032))
            marker_id += 1
            mid = 0.5 * (pursuer.position + target.position) + np.array([0.0, 0.0, 0.24])
            arr.markers.append(
                text_marker(
                    marker_id,
                    "pair_%d_range_label" % i,
                    "P%d-T%d %s\nclear=%.2f m\npaper [%.2f, %.2f], guard [%.2f, %.2f]" % (
                        i,
                        i,
                        state,
                        paper_clearance,
                        base_safe_r,
                        base_sense_r,
                        base_safe_r + inner_margin,
                        base_sense_r - outer_margin,
                    ),
                    mid,
                    color,
                    self.frame_id,
                    scale=0.18,
                )
            )
            marker_id += 1
        self.marker_pub.publish(arr)

    def set_gazebo_model_pose(self, model_name: str, state: AgentState):
        if self.set_model_state is None:
            return
        if rospy.Time.now().to_sec() - self.start_time < self.gazebo_move_start_delay_s:
            return
        req = self.ModelState()
        req.model_name = model_name
        req.reference_frame = "world"
        req.pose.position = Point(*[float(x) for x in state.position])
        req.pose.orientation = quaternion_from_yaw(yaw_from_velocity(state.velocity))
        req.twist.linear.x = float(state.velocity[0])
        req.twist.linear.y = float(state.velocity[1])
        req.twist.linear.z = float(state.velocity[2])
        try:
            self.set_model_state(req)
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "failed to move model %s: %s", model_name, exc)

    def set_gazebo_target(self, state: AgentState):
        self.set_gazebo_model_pose(state.name, state)
        self.set_gazebo_model_pose("%s_collision_shell" % state.name, state)
        self.set_gazebo_model_pose("%s_sensing_shell" % state.name, state)

    def spin(self):
        rate = rospy.Rate(1.0 / self.dt)
        while not rospy.is_shutdown():
            elapsed = rospy.Time.now().to_sec() - self.start_time
            t = max(0.0, elapsed - self.trajectory_start_delay_s)
            if elapsed >= self.trajectory_start_delay_s:
                self.targets = self.trajectory.integrate_all(self.targets, t, self.dt)
            for i, state in enumerate(self.targets):
                self.trails[i].append(state.position.copy())
                if len(self.trails[i]) > self.max_trail_points:
                    self.trails[i] = self.trails[i][-self.max_trail_points :]
                self.odom_pubs[i].publish(make_odom(state, self.frame_id))
                self.set_gazebo_target(state)
            self.publish_markers()
            self.diag_pub.publish(
                String(
                    json.dumps(
                        {
                            "scenario": self.trajectory.scenario,
                            "t": t,
                            "trajectory_waiting": elapsed < self.trajectory_start_delay_s,
                        }
                    )
                )
            )
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("paper_target_server")
    try:
        PaperTargetServer().spin()
    except rospy.ROSInterruptException:
        pass
