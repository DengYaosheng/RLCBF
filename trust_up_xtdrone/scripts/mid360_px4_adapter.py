#!/usr/bin/env python3
"""Direct PX4/MAVROS deployment adapter for Mid360 localized UAVs."""

import json
import os
import sys
from typing import List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PKG_DIR, "src"))

import numpy as np  # noqa: E402
import rospy  # noqa: E402
from geometry_msgs.msg import PoseArray, TwistStamped  # noqa: E402
from mavros_msgs.srv import CommandBool, SetMode  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from trust_up_xtdrone.config_io import (  # noqa: E402
    load_yaml,
    limits_from_config,
    obstacles_from_config,
    policy_from_config,
    resolve_path,
    safety_from_config,
)
from trust_up_xtdrone.core import AgentState, NominalPolicy, TrustUpController, yaw_rate_to_target  # noqa: E402


def vec_from_point(point) -> np.ndarray:
    return np.array([point.x, point.y, point.z], dtype=float)


def deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


class OdomSlot:
    def __init__(self, name: str):
        self.name = name
        self.state = None

    def callback(self, msg: Odometry):
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec()
        self.state = AgentState(
            vec_from_point(msg.pose.pose.position),
            vec_from_point(msg.twist.twist.linear),
            stamp=stamp,
            name=self.name,
        )

    def fresh(self, timeout: float) -> bool:
        return self.state is not None and (rospy.Time.now().to_sec() - self.state.stamp) <= timeout


class Mid360Px4Adapter:
    def __init__(self):
        deploy_cfg_path = rospy.get_param("~deploy_config", os.path.join(PKG_DIR, "config", "mid360_px4.yaml"))
        self.deploy_cfg = load_yaml(resolve_path(deploy_cfg_path))
        pursuit_cfg_path = rospy.get_param(
            "~pursuit_config",
            self.deploy_cfg.get("paper_target_pursuit_config", os.path.join(PKG_DIR, "config", "paper_target_pursuit.yaml")),
        )
        self.pursuit_cfg = load_yaml(resolve_path(pursuit_cfg_path))
        deep_update(self.pursuit_cfg, self.deploy_cfg.get("controller_overrides", {}))

        px4_cfg = self.deploy_cfg.get("px4", {})
        state_cfg = self.deploy_cfg.get("state_estimation", {})
        target_cfg = self.deploy_cfg.get("target_tracking", {})
        failsafe = self.deploy_cfg.get("failsafe", {})
        self.mavros_ns = str(rospy.get_param("~mavros_ns", px4_cfg.get("mavros_ns", "/mavros"))).rstrip("/")
        self.publish_rate_hz = float(px4_cfg.get("publish_rate_hz", 30.0))
        self.auto_arm = bool(rospy.get_param("~auto_arm", px4_cfg.get("auto_arm", False)))
        self.auto_offboard = bool(rospy.get_param("~auto_offboard", px4_cfg.get("auto_offboard", False)))
        self.odom_timeout = float(state_cfg.get("stale_timeout_s", 0.25))
        self.target_timeout = float(target_cfg.get("target_timeout_s", 0.3))
        self.hover_on_qp_infeasible = bool(failsafe.get("hover_on_qp_infeasible", True))
        self.geofence_xyz = np.asarray(failsafe.get("geofence_xyz", [8.0, 8.0, 5.0]), dtype=float)
        self.min_altitude = float(failsafe.get("min_altitude", 0.6))
        self.max_altitude = float(failsafe.get("max_altitude", 8.0))

        safety = safety_from_config(self.pursuit_cfg)
        limits = limits_from_config(self.pursuit_cfg)
        policy_params = policy_from_config(self.pursuit_cfg)
        self.controller = TrustUpController(safety, limits, NominalPolicy(safety, limits, policy_params))
        self.static_obstacles = obstacles_from_config(self.pursuit_cfg)
        self.dynamic_obstacles: List[AgentState] = []
        self.self_slot = OdomSlot("px4_mid360")
        self.target_slot = OdomSlot("target")
        rospy.Subscriber(state_cfg.get("odom_topic", "/Odometry"), Odometry, self.self_slot.callback, queue_size=1)
        rospy.Subscriber(target_cfg.get("target_odom_topic", "/target_tracker/odom"), Odometry, self.target_slot.callback, queue_size=1)
        rospy.Subscriber(self.deploy_cfg.get("mid360", {}).get("obstacle_pose_topic", "/trust_up/dynamic_obstacles"), PoseArray, self.dynamic_obstacle_callback, queue_size=1)

        self.setpoint_pub = rospy.Publisher("%s/setpoint_velocity/cmd_vel" % self.mavros_ns, TwistStamped, queue_size=1)
        self.diag_pub = rospy.Publisher("/trust_up/mid360_px4/status", String, queue_size=5)
        self.arm_srv = rospy.ServiceProxy("%s/cmd/arming" % self.mavros_ns, CommandBool)
        self.mode_srv = rospy.ServiceProxy("%s/set_mode" % self.mavros_ns, SetMode)
        self.last_mode_cmd = 0.0

    def dynamic_obstacle_callback(self, msg: PoseArray):
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec()
        self.dynamic_obstacles = [
            AgentState.from_xyz(vec_from_point(pose.position), radius=max(float(pose.orientation.w), 0.1), stamp=stamp, name="mid360_%d" % idx)
            for idx, pose in enumerate(msg.poses)
        ]

    def set_mode_if_requested(self):
        if not (self.auto_arm or self.auto_offboard):
            return
        now = rospy.Time.now().to_sec()
        if now - self.last_mode_cmd < 1.0:
            return
        self.last_mode_cmd = now
        try:
            if self.auto_arm:
                self.arm_srv(True)
            if self.auto_offboard:
                self.mode_srv(custom_mode="OFFBOARD")
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "PX4 mode command failed: %s", exc)

    def publish_velocity(self, velocity: np.ndarray, yaw_rate: float):
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.twist.linear.x = float(velocity[0])
        msg.twist.linear.y = float(velocity[1])
        msg.twist.linear.z = float(velocity[2])
        msg.twist.angular.z = float(yaw_rate)
        self.setpoint_pub.publish(msg)

    def hard_failsafe(self, state: AgentState) -> bool:
        p = state.position
        if p[2] < self.min_altitude or p[2] > self.max_altitude:
            return True
        return bool(np.any(np.abs(p) > self.geofence_xyz))

    def spin(self):
        rate = rospy.Rate(self.publish_rate_hz)
        last_t = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            self.set_mode_if_requested()
            now = rospy.Time.now().to_sec()
            dt = min(max(now - last_t, 1.0 / max(self.publish_rate_hz, 1.0)), 0.15)
            last_t = now
            status = {"ready": False}
            if not self.self_slot.fresh(self.odom_timeout) or not self.target_slot.fresh(self.target_timeout):
                self.publish_velocity(np.zeros(3), 0.0)
                status["reason"] = "stale_state_or_target"
                self.diag_pub.publish(String(json.dumps(status)))
                rate.sleep()
                continue
            if self.hard_failsafe(self.self_slot.state):
                self.publish_velocity(np.zeros(3), 0.0)
                status["reason"] = "geofence_or_altitude"
                self.diag_pub.publish(String(json.dumps(status)))
                rate.sleep()
                continue

            cmd, diag = self.controller.step(
                self.self_slot.state,
                self.target_slot.state,
                self.static_obstacles + self.dynamic_obstacles,
                dt,
            )
            if self.hover_on_qp_infeasible and not diag["qp_feasible"]:
                cmd = np.zeros(3)
            yaw_rate = yaw_rate_to_target(self.self_slot.state.position, self.target_slot.state.position)
            self.publish_velocity(cmd, yaw_rate)
            status.update(
                {
                    "ready": True,
                    "distance": float(diag["barriers"]["sensing"]["distance"]),
                    "cmd": [float(x) for x in cmd],
                    "qp_feasible": bool(diag["qp_feasible"]),
                    "qp_max_violation": float(diag["qp_max_violation"]),
                    "dynamic_obstacles": len(self.dynamic_obstacles),
                }
            )
            self.diag_pub.publish(String(json.dumps(status)))
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("mid360_px4_adapter")
    Mid360Px4Adapter().spin()
