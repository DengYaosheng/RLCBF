#!/usr/bin/env python3
"""Extract conservative obstacle centers from a Mid360 registered cloud."""

import os
import sys
from collections import OrderedDict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PKG_DIR, "src"))

import numpy as np  # noqa: E402
import rospy  # noqa: E402
from geometry_msgs.msg import Pose, PoseArray, Quaternion  # noqa: E402
from sensor_msgs.msg import PointCloud2  # noqa: E402
from sensor_msgs import point_cloud2  # noqa: E402
from visualization_msgs.msg import Marker, MarkerArray  # noqa: E402

from trust_up_xtdrone.config_io import load_yaml, resolve_path  # noqa: E402


class Mid360ObstacleGuard:
    def __init__(self):
        cfg_path = rospy.get_param("~config_file", os.path.join(PKG_DIR, "config", "mid360_px4.yaml"))
        cfg = load_yaml(resolve_path(cfg_path))
        params = cfg.get("mid360", {})
        self.min_range = float(rospy.get_param("~min_range", params.get("min_range", 0.35)))
        self.max_range = float(rospy.get_param("~max_range", params.get("max_range", 6.0)))
        self.z_min = float(rospy.get_param("~z_min", params.get("z_min", -0.8)))
        self.z_max = float(rospy.get_param("~z_max", params.get("z_max", 1.2)))
        self.voxel_size = float(rospy.get_param("~voxel_size", params.get("voxel_size", 0.35)))
        self.max_obstacles = int(rospy.get_param("~max_obstacles", params.get("max_obstacles", 12)))
        self.obstacle_radius = float(rospy.get_param("~obstacle_radius", params.get("obstacle_radius", 0.35)))
        self.pose_pub = rospy.Publisher(params.get("obstacle_pose_topic", "/trust_up/dynamic_obstacles"), PoseArray, queue_size=1)
        self.marker_pub = rospy.Publisher("/trust_up/mid360_obstacle_markers", MarkerArray, queue_size=1)
        rospy.Subscriber(params.get("pointcloud_topic", "/cloud_registered"), PointCloud2, self.cloud_callback, queue_size=1)

    def cloud_callback(self, msg: PointCloud2):
        voxels = OrderedDict()
        for x, y, z in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            p = np.array([x, y, z], dtype=float)
            dist = float(np.linalg.norm(p))
            if dist < self.min_range or dist > self.max_range or z < self.z_min or z > self.z_max:
                continue
            key = tuple(np.floor(p / self.voxel_size).astype(int).tolist())
            old = voxels.get(key)
            if old is None or dist < old[1]:
                voxels[key] = (p, dist)

        selected = sorted(voxels.values(), key=lambda item: item[1])[: self.max_obstacles]
        pose_array = PoseArray()
        pose_array.header = msg.header
        markers = MarkerArray()
        for idx, (p, _) in enumerate(selected):
            pose = Pose()
            pose.position.x = float(p[0])
            pose.position.y = float(p[1])
            pose.position.z = float(p[2])
            pose.orientation = Quaternion(0.0, 0.0, 0.0, self.obstacle_radius)
            pose_array.poses.append(pose)

            marker = Marker()
            marker.header = msg.header
            marker.ns = "mid360_obstacles"
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose = pose
            marker.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
            marker.scale.x = marker.scale.y = marker.scale.z = 2.0 * self.obstacle_radius
            marker.color.r = 1.0
            marker.color.g = 0.15
            marker.color.b = 0.05
            marker.color.a = 0.35
            marker.lifetime = rospy.Duration(0.25)
            markers.markers.append(marker)

        self.pose_pub.publish(pose_array)
        self.marker_pub.publish(markers)


if __name__ == "__main__":
    rospy.init_node("mid360_obstacle_guard")
    Mid360ObstacleGuard()
    rospy.spin()
