#!/usr/bin/env python3
"""Minimal XTDrone-compatible velocity bridge to MAVROS/PX4."""

import argparse
import math

import rospy
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from std_msgs.msg import String


class XTDroneVelocityBridge:
    def __init__(self, vehicle_type: str, vehicle_id: int):
        self.name = "%s_%d" % (vehicle_type, vehicle_id)
        self.current_pose = None
        self.current_yaw = 0.0
        self.mavros_state = State()
        self.max_speed = float(rospy.get_param("~max_speed", 4.0))
        self.max_vertical_speed = float(rospy.get_param("~max_vertical_speed", 2.0))
        self.max_yaw_rate = float(rospy.get_param("~max_yaw_rate", 0.8))
        self.target = TwistStamped()
        self.target.header.frame_id = "map"
        rospy.Subscriber("/%s/mavros/local_position/pose" % self.name, PoseStamped, self.pose_callback, queue_size=1)
        rospy.Subscriber("/%s/mavros/state" % self.name, State, self.state_callback, queue_size=1)
        rospy.Subscriber("/xtdrone/%s/cmd_vel_enu" % self.name, Twist, self.velocity_callback, queue_size=1)
        rospy.Subscriber("/xtdrone/%s/cmd" % self.name, String, self.command_callback, queue_size=3)
        self.setpoint_pub = rospy.Publisher("/%s/mavros/setpoint_velocity/cmd_vel" % self.name, TwistStamped, queue_size=1)
        self.arm_srv = rospy.ServiceProxy("/%s/mavros/cmd/arming" % self.name, CommandBool)
        self.mode_srv = rospy.ServiceProxy("/%s/mavros/set_mode" % self.name, SetMode)

    def limit_velocity(self, msg: Twist):
        vx = float(msg.linear.x)
        vy = float(msg.linear.y)
        vz = max(min(float(msg.linear.z), self.max_vertical_speed), -self.max_vertical_speed)
        norm = math.sqrt(vx * vx + vy * vy + vz * vz)
        if self.max_speed > 0.0 and norm > self.max_speed:
            scale = self.max_speed / max(norm, 1.0e-9)
            vx *= scale
            vy *= scale
            vz *= scale
        yaw_rate = max(min(float(msg.angular.z), self.max_yaw_rate), -self.max_yaw_rate)
        return vx, vy, vz, yaw_rate

    def pose_callback(self, msg: PoseStamped):
        self.current_pose = msg.pose
        q = msg.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny, cosy)

    def state_callback(self, msg: State):
        self.mavros_state = msg

    def velocity_callback(self, msg: Twist):
        vx, vy, vz, yaw_rate = self.limit_velocity(msg)
        self.target.header.stamp = rospy.Time.now()
        self.target.twist.linear.x = vx
        self.target.twist.linear.y = vy
        self.target.twist.linear.z = vz
        self.target.twist.angular.z = yaw_rate

    def command_callback(self, msg: String):
        if not msg.data or msg.data == "stop controlling":
            return
        try:
            if msg.data == "ARM":
                if not self.mavros_state.armed:
                    response = self.arm_srv(True)
                    rospy.loginfo("%s ARM result: %s", self.name, response.success)
            elif msg.data == "DISARM":
                response = self.arm_srv(False)
                rospy.loginfo("%s DISARM result: %s", self.name, response.success)
            elif msg.data == "HOVER":
                self.target.twist.linear.x = 0.0
                self.target.twist.linear.y = 0.0
                self.target.twist.linear.z = 0.0
                self.target.twist.angular.z = 0.0
            else:
                if self.mavros_state.mode != msg.data:
                    response = self.mode_srv(custom_mode=msg.data)
                    rospy.loginfo("%s mode %s result: %s", self.name, msg.data, response.mode_sent)
        except Exception as exc:
            rospy.logwarn("%s command %s failed: %s", self.name, msg.data, exc)

    def spin(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            self.target.header.stamp = rospy.Time.now()
            self.setpoint_pub.publish(self.target)
            rate.sleep()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vehicle_type")
    parser.add_argument("vehicle_id", type=int)
    args, _ = parser.parse_known_args()
    rospy.init_node("%s_%d_xtdrone_velocity_bridge" % (args.vehicle_type, args.vehicle_id))
    try:
        XTDroneVelocityBridge(args.vehicle_type, args.vehicle_id).spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
