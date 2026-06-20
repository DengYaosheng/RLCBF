#!/usr/bin/env bash
# Source this before launching the TRUST-UP XTDrone/PX4 Gazebo reproduction.

export TRUST_UP_CATKIN_WS="${TRUST_UP_CATKIN_WS:-/home/ysdeng/work/catkin_ws}"
export PX4_AUTOPILOT="${PX4_AUTOPILOT:-/home/ysdeng/work/PX4-Autopilot}"
export XTDRONE_ROOT="${XTDRONE_ROOT:-/home/ysdeng/projects/XTDrone}"
export MAVLINK_SITL_GAZEBO="${MAVLINK_SITL_GAZEBO:-${PX4_AUTOPILOT}/Tools/simulation/gazebo-classic/sitl_gazebo-classic}"

source /opt/ros/noetic/setup.bash
source "${TRUST_UP_CATKIN_WS}/devel/setup.bash"

export ROS_PACKAGE_PATH="${PX4_AUTOPILOT}:${MAVLINK_SITL_GAZEBO}:${ROS_PACKAGE_PATH}"
export GAZEBO_MODEL_PATH="${XTDRONE_ROOT}/sitl_config/models:${MAVLINK_SITL_GAZEBO}/models:${GAZEBO_MODEL_PATH}"

if [ -f "${PX4_AUTOPILOT}/Tools/simulation/gazebo-classic/setup_gazebo.bash" ] && \
   [ -d "${PX4_AUTOPILOT}/build/px4_sitl_default" ]; then
  source "${PX4_AUTOPILOT}/Tools/simulation/gazebo-classic/setup_gazebo.bash" \
    "${PX4_AUTOPILOT}" \
    "${PX4_AUTOPILOT}/build/px4_sitl_default"
fi

rospack profile >/dev/null

echo "TRUST-UP env ready:"
echo "  trust_up_xtdrone: $(rospack find trust_up_xtdrone)"
echo "  px4:              $(rospack find px4)"
echo "  mavlink gazebo:   $(rospack find mavlink_sitl_gazebo)"
