# Interfaces

## XTDrone ROS1 Simulation

Inputs:

- Pursuer pose or odometry: `/<uav_type>_<id>/mavros/local_position/pose`, `/<uav_type>_<id>/mavros/local_position/odom`
- Target odometry: `/trust_up/target_0/odom`, `/trust_up/target_1/odom`
- Optional dynamic obstacles: `/trust_up/dynamic_obstacles` as `geometry_msgs/PoseArray`; `pose.orientation.w` carries obstacle radius.

Outputs:

- XTDrone command velocity: `/xtdrone/<uav_type>_<id>/cmd_vel_enu`
- Optional XTDrone command strings: `/xtdrone/<uav_type>_<id>/cmd`
- Diagnostics: `/trust_up/diagnostics`
- RViz markers: `/trust_up/markers`

The controller publishes ENU velocity commands because XTDrone's multirotor
communication bridge already maps that topic to MAVROS `setpoint_raw/local`.

## PX4 + Mid360 Real Vehicle

Required upstream stack:

- PX4 running with MAVROS.
- A local-position estimator publishing `nav_msgs/Odometry`, typically FAST-LIO
  from Mid360 on `/Odometry`.
- A target tracker publishing `nav_msgs/Odometry` on `/target_tracker/odom`.
- A registered point cloud in the same frame as odometry, typically
  `/cloud_registered`, for the optional obstacle guard.

Outputs:

- MAVROS velocity setpoint: `/mavros/setpoint_raw/local`
- Status: `/trust_up/mid360_px4/status`

Safety notes:

- `auto_arm` and `auto_offboard` default to `false`.
- Use tethered prop-off SITL/HITL checks before enabling auto mode commands.
- The Mid360 obstacle guard assumes the point cloud and odometry are in the same
  metric frame. Raw Livox packets are not suitable without registration.

## Isaac Sim ROS1 Bridge

Config:

- `/home/ysdeng/projects/xtdrone_trust_pursuit/trust_up_xtdrone/config/isaac_sim.yaml`

Inputs from Isaac:

- `/isaac/iris_0/odom`, `/isaac/iris_1/odom` as `nav_msgs/Odometry`
- optional `/isaac/trust_up/dynamic_obstacles` as `geometry_msgs/PoseArray`
- optional `/isaac/trust_target_i/odom` if targets are Isaac-owned

Outputs to Isaac:

- `/isaac/iris_0/cmd_vel_enu`, `/isaac/iris_1/cmd_vel_enu`
- `/isaac/trust_target_0/pose_cmd`, `/isaac/trust_target_1/pose_cmd`
- `/isaac/trust_target_i/cmd_vel_enu`

Internal remaps:

- Isaac vehicle odometry is republished to MAVROS-like topics under
  `/iris_i/mavros/local_position/*`, so the same `trust_up_pursuit_node.py`
  runs unchanged.
- By default `paper_target_server.py` owns target trajectories and Isaac follows
  target pose commands. Set `targets[*].source: isaac` to reverse ownership.
