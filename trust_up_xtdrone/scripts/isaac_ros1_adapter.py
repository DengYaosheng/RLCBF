#!/usr/bin/env python3
"""ROS1 topic adapter between Isaac Sim and the TRUST-UP/XTDrone interface."""

import os
import sys
from typing import Dict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PKG_DIR, "src"))

import rospy  # noqa: E402
from geometry_msgs.msg import PoseArray, PoseStamped, Twist  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402

from trust_up_xtdrone.config_io import load_yaml, resolve_path  # noqa: E402


def pose_from_odom(msg: Odometry, frame_id: str) -> PoseStamped:
    pose = PoseStamped()
    pose.header = msg.header
    pose.header.frame_id = frame_id or msg.header.frame_id
    pose.pose = msg.pose.pose
    return pose


def retime_odom(msg: Odometry, frame_id: str, child_frame_id: str) -> Odometry:
    out = Odometry()
    out.header = msg.header
    out.header.stamp = msg.header.stamp if msg.header.stamp else rospy.Time.now()
    out.header.frame_id = frame_id or msg.header.frame_id
    out.child_frame_id = child_frame_id or msg.child_frame_id
    out.pose = msg.pose
    out.twist = msg.twist
    return out


class IsaacRos1Adapter:
    def __init__(self):
        cfg_path = rospy.get_param("~config_file", os.path.join(PKG_DIR, "config", "isaac_sim.yaml"))
        self.cfg = load_yaml(resolve_path(cfg_path))
        isaac_cfg = self.cfg.get("isaac", {})
        self.frame_id = str(isaac_cfg.get("world_frame", "map"))
        self.vehicle_publishers: Dict[str, Dict[str, rospy.Publisher]] = {}
        self.target_publishers: Dict[str, Dict[str, rospy.Publisher]] = {}

        for vehicle in self.cfg.get("vehicles", []):
            name = str(vehicle["name"])
            odom_pub = rospy.Publisher(vehicle["mavros_odom_topic"], Odometry, queue_size=3)
            pose_pub = rospy.Publisher(vehicle["mavros_pose_topic"], PoseStamped, queue_size=3)
            cmd_pub = rospy.Publisher(vehicle["isaac_cmd_topic"], Twist, queue_size=3)
            self.vehicle_publishers[name] = {"odom": odom_pub, "pose": pose_pub, "cmd": cmd_pub}
            rospy.Subscriber(vehicle["isaac_odom_topic"], Odometry, self.make_vehicle_odom_cb(name), queue_size=3)
            rospy.Subscriber(vehicle["command_in_topic"], Twist, self.make_vehicle_cmd_cb(name), queue_size=3)

        for target in self.cfg.get("targets", []):
            name = str(target["name"])
            source = str(target.get("source", "trust_up"))
            pose_pub = rospy.Publisher(target["isaac_pose_cmd_topic"], PoseStamped, queue_size=3)
            cmd_pub = rospy.Publisher(target["isaac_cmd_topic"], Twist, queue_size=3)
            trust_pub = rospy.Publisher(target["trust_odom_topic"], Odometry, queue_size=3)
            self.target_publishers[name] = {"pose": pose_pub, "cmd": cmd_pub, "trust": trust_pub}
            if source == "isaac":
                rospy.Subscriber(target["isaac_odom_topic"], Odometry, self.make_target_isaac_odom_cb(name), queue_size=3)
            else:
                rospy.Subscriber(target["trust_odom_topic"], Odometry, self.make_target_trust_odom_cb(name), queue_size=3)

        if bool(isaac_cfg.get("passthrough_dynamic_obstacles", True)):
            self.dynamic_obstacle_pub = rospy.Publisher(
                isaac_cfg.get("dynamic_obstacles_out", "/trust_up/dynamic_obstacles"),
                PoseArray,
                queue_size=3,
            )
            rospy.Subscriber(
                isaac_cfg.get("dynamic_obstacles_in", "/isaac/trust_up/dynamic_obstacles"),
                PoseArray,
                self.dynamic_obstacle_cb,
                queue_size=3,
            )
        rospy.loginfo("Isaac ROS1 adapter ready: %d vehicles, %d targets", len(self.vehicle_publishers), len(self.target_publishers))

    def make_vehicle_odom_cb(self, name: str):
        def callback(msg: Odometry):
            pubs = self.vehicle_publishers[name]
            odom = retime_odom(msg, self.frame_id, name)
            pubs["odom"].publish(odom)
            pubs["pose"].publish(pose_from_odom(odom, self.frame_id))

        return callback

    def make_vehicle_cmd_cb(self, name: str):
        def callback(msg: Twist):
            self.vehicle_publishers[name]["cmd"].publish(msg)

        return callback

    def make_target_trust_odom_cb(self, name: str):
        def callback(msg: Odometry):
            pubs = self.target_publishers[name]
            pose = pose_from_odom(msg, self.frame_id)
            pubs["pose"].publish(pose)
            twist = Twist()
            twist.linear = msg.twist.twist.linear
            twist.angular = msg.twist.twist.angular
            pubs["cmd"].publish(twist)

        return callback

    def make_target_isaac_odom_cb(self, name: str):
        def callback(msg: Odometry):
            self.target_publishers[name]["trust"].publish(retime_odom(msg, self.frame_id, name))

        return callback

    def dynamic_obstacle_cb(self, msg: PoseArray):
        msg.header.frame_id = self.frame_id
        self.dynamic_obstacle_pub.publish(msg)

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node("isaac_ros1_adapter")
    IsaacRos1Adapter().spin()
