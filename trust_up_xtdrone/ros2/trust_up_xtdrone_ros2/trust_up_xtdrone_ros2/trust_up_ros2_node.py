#!/usr/bin/env python3
"""ROS2 velocity-command adapter for TRUST-UP.

The shared core package must be importable, for example:
  export PYTHONPATH=/path/to/trust_up_xtdrone/src:$PYTHONPATH
"""

import json
import os
from typing import List

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

from trust_up_xtdrone.config_io import load_yaml, limits_from_config, obstacles_from_config, policy_from_config, safety_from_config
from trust_up_xtdrone.core import AgentState, NominalPolicy, TrustUpController, yaw_rate_to_target


def vec_from_point(point) -> np.ndarray:
    return np.array([point.x, point.y, point.z], dtype=float)


class OdomSlot:
    def __init__(self, name: str):
        self.name = name
        self.state = None

    def update(self, msg: Odometry):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1.0e-9
        self.state = AgentState(vec_from_point(msg.pose.pose.position), vec_from_point(msg.twist.twist.linear), stamp=stamp, name=self.name)


class TrustUpRos2Node(Node):
    def __init__(self):
        super().__init__("trust_up_ros2_node")
        self.declare_parameter("config_file", "paper_target_pursuit.yaml")
        self.declare_parameter("uav_type", "iris")
        self.declare_parameter("vehicle_id", 0)
        self.declare_parameter("state_topic", "/iris_0/mavros/local_position/odom")
        self.declare_parameter("target_topic", "/trust_up/target_0/odom")
        self.declare_parameter("cmd_topic", "/xtdrone/iris_0/cmd_vel_enu")
        self.declare_parameter("dynamic_obstacles_topic", "/trust_up/dynamic_obstacles")
        self.declare_parameter("rate_hz", 30.0)

        cfg_path = self.get_parameter("config_file").value
        if not os.path.exists(cfg_path):
            raise RuntimeError("config_file must be an absolute path in ROS2 mode: %s" % cfg_path)
        self.cfg = load_yaml(cfg_path)
        safety = safety_from_config(self.cfg)
        limits = limits_from_config(self.cfg)
        policy_params = policy_from_config(self.cfg)
        self.controller = TrustUpController(safety, limits, NominalPolicy(safety, limits, policy_params))
        self.static_obstacles = obstacles_from_config(self.cfg)
        self.dynamic_obstacles: List[AgentState] = []
        self.self_slot = OdomSlot("self")
        self.target_slot = OdomSlot("target")

        self.create_subscription(Odometry, self.get_parameter("state_topic").value, self.self_slot.update, 10)
        self.create_subscription(Odometry, self.get_parameter("target_topic").value, self.target_slot.update, 10)
        self.create_subscription(PoseArray, self.get_parameter("dynamic_obstacles_topic").value, self.obstacles_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, self.get_parameter("cmd_topic").value, 10)
        self.diag_pub = self.create_publisher(String, "/trust_up/ros2/status", 10)
        self.last_t = self.get_clock().now().nanoseconds * 1.0e-9
        self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self.tick)

    def obstacles_callback(self, msg: PoseArray):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1.0e-9
        self.dynamic_obstacles = [
            AgentState.from_xyz(vec_from_point(pose.position), radius=max(float(pose.orientation.w), 0.1), stamp=stamp, name="ros2_dynamic_%d" % idx)
            for idx, pose in enumerate(msg.poses)
        ]

    def tick(self):
        now = self.get_clock().now().nanoseconds * 1.0e-9
        dt = min(max(now - self.last_t, 1.0 / 30.0), 0.15)
        self.last_t = now
        if self.self_slot.state is None or self.target_slot.state is None:
            return
        cmd, diag = self.controller.step(self.self_slot.state, self.target_slot.state, self.static_obstacles + self.dynamic_obstacles, dt)
        msg = Twist()
        msg.linear.x = float(cmd[0])
        msg.linear.y = float(cmd[1])
        msg.linear.z = float(cmd[2])
        msg.angular.z = yaw_rate_to_target(self.self_slot.state.position, self.target_slot.state.position)
        self.cmd_pub.publish(msg)
        self.diag_pub.publish(String(data=json.dumps({"cmd": [float(x) for x in cmd], "qp_feasible": bool(diag["qp_feasible"])})))


def main():
    rclpy.init()
    node = TrustUpRos2Node()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
