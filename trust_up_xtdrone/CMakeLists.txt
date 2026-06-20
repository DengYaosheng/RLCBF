cmake_minimum_required(VERSION 3.0.2)
project(trust_up_xtdrone)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  visualization_msgs
  gazebo_msgs
  mavros_msgs
)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  scripts/audit_paper_conformance.py
  scripts/evaluate_headless_metrics.py
  scripts/gazebo_paper_replay_visualizer.py
  scripts/gzclient_raw.py
  scripts/isaac_ros1_adapter.py
  scripts/mid360_obstacle_guard.py
  scripts/mid360_px4_adapter.py
  scripts/paper_target_server.py
  scripts/run_headless_paper_experiment.py
  scripts/spawn_paper_world.py
  scripts/trust_up_pursuit_node.py
  scripts/xtdrone_velocity_bridge.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch worlds rviz docs isaac
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
