#!/usr/bin/env python3
"""XTDrone-facing TRUST-UP pursuit controller."""

import json
import os
import sys
from typing import Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PKG_DIR, "src"))

import numpy as np  # noqa: E402
import rospy  # noqa: E402
from geometry_msgs.msg import PoseArray, PoseStamped, Twist  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from trust_up_xtdrone.config_io import (  # noqa: E402
    default_config_path,
    load_yaml,
    limits_from_config,
    obstacles_from_config,
    policy_from_config,
    safety_from_config,
)
from trust_up_xtdrone.core import AgentState, NominalPolicy, TrustUpController, clamp_norm, yaw_rate_to_target  # noqa: E402


def vec_from_point(point) -> np.ndarray:
    return np.array([point.x, point.y, point.z], dtype=float)


class StateSlot:
    def __init__(self, name: str, radius: float = 0.0):
        self.name = name
        self.radius = float(radius)
        self.state = None
        self.last_pose = None
        self.last_stamp = 0.0

    def update_pose(self, msg: PoseStamped):
        now = msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec()
        pos = vec_from_point(msg.pose.position)
        vel = np.zeros(3)
        if self.last_pose is not None and now > self.last_stamp:
            vel = (pos - self.last_pose) / max(now - self.last_stamp, 1.0e-3)
        self.state = AgentState(pos, vel, radius=self.radius, stamp=now, name=self.name)
        self.last_pose = pos
        self.last_stamp = now

    def update_odom(self, msg: Odometry):
        now = msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec()
        pos = vec_from_point(msg.pose.pose.position)
        vel = vec_from_point(msg.twist.twist.linear)
        self.state = AgentState(pos, vel, radius=self.radius, stamp=now, name=self.name)
        self.last_pose = pos
        self.last_stamp = now

    def fresh(self, timeout: float) -> bool:
        return self.state is not None and (rospy.Time.now().to_sec() - self.state.stamp) <= timeout


class TrustUpPursuitNode:
    def __init__(self):
        cfg_path = rospy.get_param("~config_file", default_config_path())
        self.cfg = load_yaml(cfg_path)
        if rospy.has_param("~scenario"):
            self.cfg.setdefault("scenario", {})["name"] = rospy.get_param("~scenario")

        xt = self.cfg.get("xtdrone", {})
        self.uav_type = rospy.get_param("~uav_type", xt.get("uav_type", "iris"))
        self.pursuer_ids = list(rospy.get_param("~pursuer_ids", xt.get("pursuer_ids", [0, 1])))
        self.stale_timeout = float(xt.get("stale_timeout_s", 0.7))
        self.auto_arm = bool(rospy.get_param("~auto_arm", xt.get("auto_arm", False)))
        self.auto_offboard = bool(rospy.get_param("~auto_offboard", xt.get("auto_offboard", False)))
        self.auto_start_delay_s = float(rospy.get_param("~auto_start_delay_s", xt.get("auto_start_delay_s", 8.0)))
        self.pursuit_start_delay_s = float(rospy.get_param("~pursuit_start_delay_s", xt.get("pursuit_start_delay_s", 0.0)))
        self.publish_rate_hz = float(xt.get("publish_rate_hz", 30.0))
        self.command_namespace = str(xt.get("command_namespace", "/xtdrone")).rstrip("/")
        self.start_time = rospy.Time.now().to_sec()
        smooth_cfg = self.cfg.get("command_smoothing", {})
        self.command_smoothing_enabled = bool(smooth_cfg.get("enabled", True))
        self.command_tau = float(smooth_cfg.get("tau_s", 0.35))
        self.max_command_accel = float(smooth_cfg.get("max_command_accel", self.cfg.get("vehicle_profile", {}).get("max_accel", 1.0)))
        self.max_vertical_command_accel = float(smooth_cfg.get("max_vertical_command_accel", 0.8 * self.max_command_accel))
        self.filtered_commands: Dict[int, np.ndarray] = {}

        safety = safety_from_config(self.cfg)
        limits = limits_from_config(self.cfg)
        policy_params = policy_from_config(self.cfg)
        self.safety = safety
        self.limits = limits
        self.controllers = [
            TrustUpController(safety, limits, NominalPolicy(safety, limits, policy_params))
            for _ in self.pursuer_ids
        ]
        self.static_obstacles = obstacles_from_config(self.cfg)
        self.dynamic_obstacles: List[AgentState] = []
        default_uav_radius = float(self.cfg.get("physical_envelope", {}).get("uav_radius", 0.0))
        pursuer_cfg = self.cfg.get("pursuers", [])
        target_cfg = self.cfg.get("targets", [])

        self.pursuers: Dict[int, StateSlot] = {}
        self.targets: Dict[int, StateSlot] = {}
        self.cmd_vel_pubs = {}
        self.cmd_pubs = {}
        for local_idx, vehicle_id in enumerate(self.pursuer_ids):
            name = "%s_%d" % (self.uav_type, int(vehicle_id))
            pursuer_radius = float(pursuer_cfg[local_idx].get("radius", default_uav_radius)) if local_idx < len(pursuer_cfg) else default_uav_radius
            slot = StateSlot(name, pursuer_radius)
            self.pursuers[local_idx] = slot
            rospy.Subscriber("/%s/mavros/local_position/pose" % name, PoseStamped, slot.update_pose, queue_size=1)
            rospy.Subscriber("/%s/mavros/local_position/odom" % name, Odometry, slot.update_odom, queue_size=1)
            self.cmd_vel_pubs[local_idx] = rospy.Publisher(
                "%s/%s/cmd_vel_enu" % (self.command_namespace, name), Twist, queue_size=1
            )
            self.cmd_pubs[local_idx] = rospy.Publisher(
                "%s/%s/cmd" % (self.command_namespace, name), String, queue_size=3
            )

            target_radius = float(target_cfg[local_idx].get("radius", default_uav_radius)) if local_idx < len(target_cfg) else default_uav_radius
            target_slot = StateSlot("target_%d" % local_idx, target_radius)
            self.targets[local_idx] = target_slot
            rospy.Subscriber("/trust_up/target_%d/odom" % local_idx, Odometry, target_slot.update_odom, queue_size=1)

        rospy.Subscriber(self.cfg.get("topics", {}).get("dynamic_obstacles", "/trust_up/dynamic_obstacles"), PoseArray, self.dynamic_obstacle_callback, queue_size=1)
        self.diag_pub = rospy.Publisher(self.cfg.get("topics", {}).get("diagnostics", "/trust_up/diagnostics"), String, queue_size=10)
        self.last_mode_cmd = 0.0

    def dynamic_obstacle_callback(self, msg: PoseArray):
        now = msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec()
        obstacles = []
        for idx, pose in enumerate(msg.poses):
            radius = pose.orientation.w if pose.orientation.w > 0.0 else 0.25
            obstacles.append(AgentState.from_xyz(vec_from_point(pose.position), radius=radius, stamp=now, name="dynamic_%d" % idx))
        self.dynamic_obstacles = obstacles

    def send_mode_commands(self):
        if not (self.auto_arm or self.auto_offboard):
            return
        now = rospy.Time.now().to_sec()
        if now - self.start_time < self.auto_start_delay_s:
            return
        if now - self.last_mode_cmd < 1.0:
            return
        self.last_mode_cmd = now
        for pub in self.cmd_pubs.values():
            if self.auto_offboard:
                pub.publish(String("OFFBOARD"))
            if self.auto_arm:
                pub.publish(String("ARM"))

    def make_obstacles_for(self, local_idx: int) -> List[AgentState]:
        obstacles = list(self.static_obstacles) + list(self.dynamic_obstacles)
        for idx, slot in self.pursuers.items():
            if idx != local_idx and slot.fresh(self.stale_timeout):
                obstacles.append(slot.state)
        for idx, slot in self.targets.items():
            if idx != local_idx and slot.fresh(self.stale_timeout):
                obstacles.append(slot.state)
        return obstacles

    def publish_hover(self, local_idx: int):
        self.cmd_vel_pubs[local_idx].publish(Twist())

    def bootstrap_command(self, pursuer: AgentState, target: AgentState) -> np.ndarray:
        delta = target.position - pursuer.position
        distance = float(np.linalg.norm(delta))
        if distance < 1.0e-6:
            return np.zeros(3)
        direction = delta / distance
        range_error = distance - self.safety.desired_bound(pursuer, target)
        cmd = target.velocity + 1.0 * range_error * direction
        cmd = clamp_norm(cmd, 0.85 * float(self.limits.max_speed))
        if abs(cmd[2]) > self.limits.max_vertical_speed:
            cmd[2] = float(np.sign(cmd[2]) * self.limits.max_vertical_speed)
        return cmd

    def publish_velocity(self, local_idx: int, pursuer: AgentState, target: AgentState, cmd: np.ndarray):
        twist = Twist()
        twist.linear.x = float(cmd[0])
        twist.linear.y = float(cmd[1])
        twist.linear.z = float(cmd[2])
        twist.angular.z = yaw_rate_to_target(pursuer.position, target.position)
        self.cmd_vel_pubs[local_idx].publish(twist)

    def smooth_command(self, local_idx: int, raw_cmd: np.ndarray, dt: float) -> np.ndarray:
        raw_cmd = np.asarray(raw_cmd, dtype=float).reshape(3)
        if not self.command_smoothing_enabled:
            self.filtered_commands[local_idx] = raw_cmd.copy()
            return raw_cmd
        previous = self.filtered_commands.get(local_idx)
        if previous is None:
            previous = np.zeros(3)
        alpha = dt / max(self.command_tau + dt, 1.0e-3)
        desired = previous + alpha * (raw_cmd - previous)
        delta = desired - previous
        delta_xy = clamp_norm(delta[:2], max(self.max_command_accel, 0.0) * dt)
        delta_z_limit = max(self.max_vertical_command_accel, 0.0) * dt
        delta_z = float(np.clip(delta[2], -delta_z_limit, delta_z_limit))
        smoothed = previous + np.array([delta_xy[0], delta_xy[1], delta_z], dtype=float)
        smoothed = clamp_norm(smoothed, self.limits.max_speed)
        if abs(smoothed[2]) > self.limits.max_vertical_speed:
            smoothed[2] = float(np.sign(smoothed[2]) * self.limits.max_vertical_speed)
        self.filtered_commands[local_idx] = smoothed.copy()
        return smoothed

    def spin(self):
        rate = rospy.Rate(self.publish_rate_hz)
        last_t = rospy.Time.now().to_sec()
        rospy.loginfo(
            "TRUST-UP pursuit node started: uav_type=%s ids=%s auto_arm=%s auto_offboard=%s auto_start_delay_s=%.1f pursuit_start_delay_s=%.1f",
            self.uav_type,
            self.pursuer_ids,
            self.auto_arm,
            self.auto_offboard,
            self.auto_start_delay_s,
            self.pursuit_start_delay_s,
        )
        while not rospy.is_shutdown():
            self.send_mode_commands()
            now = rospy.Time.now().to_sec()
            dt = min(max(now - last_t, 1.0 / max(self.publish_rate_hz, 1.0)), 0.15)
            last_t = now
            diagnostics = {"agents": []}
            for idx in range(len(self.pursuer_ids)):
                if not self.pursuers[idx].fresh(self.stale_timeout) or not self.targets[idx].fresh(self.stale_timeout):
                    self.publish_hover(idx)
                    diagnostics["agents"].append(
                        {
                            "id": idx,
                            "ready": False,
                            "pursuer_fresh": self.pursuers[idx].fresh(self.stale_timeout),
                            "target_fresh": self.targets[idx].fresh(self.stale_timeout),
                        }
                    )
                    continue
                pursuer = self.pursuers[idx].state
                target = self.targets[idx].state
                obstacles = self.make_obstacles_for(idx)
                if now - self.start_time < self.pursuit_start_delay_s:
                    cmd = self.bootstrap_command(pursuer, target)
                    cmd_smoothed = self.smooth_command(idx, cmd, dt)
                    cmd_smoothed, post_guard = self.controllers[idx].guard_velocity_command(cmd_smoothed, pursuer, target, [target] + obstacles)
                    self.filtered_commands[idx] = cmd_smoothed.copy()
                    self.controllers[idx].reset(cmd_smoothed)
                    self.publish_velocity(idx, pursuer, target, cmd_smoothed)
                    distance = float(np.linalg.norm(pursuer.position - target.position))
                    diagnostics["agents"].append(
                        {
                            "id": idx,
                            "ready": True,
                            "bootstrap": True,
                            "distance": distance,
                            "paper_clearance_distance": distance - self.safety.radius_sum(pursuer, target),
                            "effective_inner": self.safety.collision_bound(pursuer, target),
                            "effective_outer": self.safety.sensing_bound(pursuer, target),
                            "enforced_inner": self.safety.enforced_collision_bound(pursuer, target),
                            "enforced_outer": self.safety.enforced_sensing_bound(pursuer, target),
                            "cmd": [float(x) for x in cmd],
                            "cmd_smoothed": [float(x) for x in cmd_smoothed],
                            "post_smoothing_guard_feasible": bool(post_guard["velocity_guard_feasible"]),
                            "post_smoothing_guard_max_violation": float(post_guard["velocity_guard_max_violation"]),
                        }
                    )
                    continue
                cmd, diag = self.controllers[idx].step(pursuer, target, obstacles, dt)
                cmd_smoothed = self.smooth_command(idx, cmd, dt)
                cmd_smoothed, post_guard = self.controllers[idx].guard_velocity_command(cmd_smoothed, pursuer, target, [target] + obstacles)
                self.filtered_commands[idx] = cmd_smoothed.copy()
                self.controllers[idx].command_velocity = cmd_smoothed.copy()
                self.publish_velocity(idx, pursuer, target, cmd_smoothed)
                diagnostics["agents"].append(
                    {
                        "id": idx,
                        "ready": True,
                        "distance": diag["barriers"]["sensing"]["distance"],
                        "paper_clearance_distance": diag["barriers"]["sensing"]["distance"] - self.safety.radius_sum(pursuer, target),
                        "effective_inner": diag["barriers"]["target_collision"]["radius"],
                        "effective_outer": diag["barriers"]["sensing"]["radius"],
                        "paper_inner": diag["barriers"]["target_collision"].get("paper_radius", self.safety.collision_bound(pursuer, target)),
                        "paper_outer": diag["barriers"]["sensing"].get("paper_radius", self.safety.sensing_bound(pursuer, target)),
                        "cmd": [float(x) for x in cmd],
                        "cmd_smoothed": [float(x) for x in cmd_smoothed],
                        "safe_accel": [float(x) for x in diag["safe_accel"]],
                        "qp_feasible": bool(diag["qp_feasible"]),
                        "qp_max_violation": float(diag["qp_max_violation"]),
                        "post_smoothing_guard_feasible": bool(post_guard["velocity_guard_feasible"]),
                        "post_smoothing_guard_max_violation": float(post_guard["velocity_guard_max_violation"]),
                    }
                )
            self.diag_pub.publish(String(json.dumps(diagnostics)))
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("trust_up_pursuit_node")
    try:
        TrustUpPursuitNode().spin()
    except rospy.ROSInterruptException:
        pass
