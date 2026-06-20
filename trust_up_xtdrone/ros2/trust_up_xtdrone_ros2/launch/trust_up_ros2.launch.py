from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("config_file"),
            DeclareLaunchArgument("state_topic", default_value="/iris_0/mavros/local_position/odom"),
            DeclareLaunchArgument("target_topic", default_value="/trust_up/target_0/odom"),
            DeclareLaunchArgument("cmd_topic", default_value="/xtdrone/iris_0/cmd_vel_enu"),
            Node(
                package="trust_up_xtdrone_ros2",
                executable="trust_up_ros2_node",
                output="screen",
                parameters=[
                    {
                        "config_file": LaunchConfiguration("config_file"),
                        "state_topic": LaunchConfiguration("state_topic"),
                        "target_topic": LaunchConfiguration("target_topic"),
                        "cmd_topic": LaunchConfiguration("cmd_topic"),
                    }
                ],
            ),
        ]
    )
