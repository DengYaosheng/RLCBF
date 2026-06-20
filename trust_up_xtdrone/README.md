# TRUST-UP XTDrone Target Pursuit Reproduction

This package implements a reproducible XTDrone/PX4/Gazebo target-pursuit stack
for arXiv `2411.17552v3`, "TRUST-UP: Trustworthy Reinforcement learning Using
Safe Techniques for UAV Pursuit".

The code keeps the paper experiment settings in `config/paper_target_pursuit.yaml`:

- Two pursuer UAVs and two target UAVs.
- Two spherical obstacles at `[4.70, 3.25, 3.00]` and `[-4.20, 3.00, 4.75]`.
- Collision radius `r_i = 0.5`, sensing radius `R_i = 1.0`; physical XTDrone
  hull length is added separately as center-to-center padding for Gazebo/PX4.
- Target references for circular and figure-8 maneuvers.
- `dt = 0.1`, `steps = 600`, and SAC-training metadata of `1500` episodes.

XTDrone does not ship a native Tello PX4 airframe. The provided SITL launch uses
PX4 `iris` dynamics with a conservative `tello_limited_px4_sitl` speed and
acceleration profile. The same controller can then be deployed through MAVROS to
a PX4 UAV localized by Mid360/FAST-LIO.

## Gazebo-Native Paper Visualization

For WSL and quick paper-figure inspection, use the native Gazebo replay first.
It spawns four XTDrone-derived UAV visuals directly in Gazebo, runs the same
TRUST-UP controller, and displays the paper obstacle/safety/sensing geometry
without needing PX4, MAVROS, or RViz.
The replay resolves the XTDrone `iris_zhihang` visual SDF to real PX4 `iris`
mesh assets; the paper-conformance audit fails if primitive fallback UAVs are
used.

```bash
source /home/ysdeng/projects/xtdrone_trust_pursuit/trust_up_xtdrone/setup_trust_up_noetic.bash
roslaunch trust_up_xtdrone gazebo_paper_replay.launch scenario:=circle gui:=true
roslaunch trust_up_xtdrone gazebo_paper_replay.launch scenario:=figure8 gui:=true
```

On WSLg, keep `software_rendering:=false` unless you are using an old external
X server.  Forcing software OpenGL can make `gzclient` open as an all-black
window.

Live constraint status is published on:

```bash
rostopic echo /trust_up/gazebo_replay/status
```

If WSL rendering is slow, keep the Gazebo UAV/shell visualization and disable
only the history dots:

```bash
roslaunch trust_up_xtdrone gazebo_paper_replay.launch scenario:=circle enable_trails:=false
```

## Build

On this machine the active Noetic workspace is `/home/ysdeng/work/catkin_ws`.

```bash
ln -sfn /home/ysdeng/projects/xtdrone_trust_pursuit/trust_up_xtdrone \
  /home/ysdeng/work/catkin_ws/src/trust_up_xtdrone
cd /home/ysdeng/work/catkin_ws
catkin_make
source /home/ysdeng/projects/xtdrone_trust_pursuit/trust_up_xtdrone/setup_trust_up_noetic.bash
```

XTDrone and PX4 SITL must already be installed as expected by XTDrone.

## Run the Full XTDrone SITL Reproduction

```bash
roslaunch trust_up_xtdrone xtdrone_two_tello_profile_sitl.launch scenario:=circle
```

For the second paper experiment:

```bash
roslaunch trust_up_xtdrone xtdrone_two_tello_profile_sitl.launch scenario:=figure8
```

For WSL or remote sessions without a reliable Gazebo client:

```bash
source /home/ysdeng/projects/xtdrone_trust_pursuit/trust_up_xtdrone/setup_trust_up_noetic.bash
roslaunch trust_up_xtdrone xtdrone_two_tello_profile_sitl.launch scenario:=circle gui:=false
```

If you only need the reproducibility check and not Gazebo graphics:

```bash
python3 /home/ysdeng/projects/xtdrone_trust_pursuit/trust_up_xtdrone/scripts/run_headless_paper_experiment.py \
  --scenario circle \
  --output /tmp/trust_up_circle.csv
```

Mode commands are disabled by default. After verifying setpoint streaming, you
can explicitly enable:

```bash
roslaunch trust_up_xtdrone xtdrone_two_tello_profile_sitl.launch auto_arm:=true auto_offboard:=true
```

## Run Only the Controller on an Existing XTDrone Session

Start your XTDrone multi-UAV session and the XTDrone communication bridge, then:

```bash
roslaunch trust_up_xtdrone paper_target_pursuit.launch scenario:=circle uav_type:=iris
```

## Headless Numerical Check

```bash
python3 /home/ysdeng/projects/xtdrone_trust_pursuit/trust_up_xtdrone/scripts/run_headless_paper_experiment.py \
  --scenario circle \
  --output /tmp/trust_up_circle.csv
```

## PX4 + Mid360 Deployment Interface

The deployment launch expects:

- MAVROS at `/mavros`.
- Mid360/FAST-LIO odometry at `/Odometry`.
- Registered cloud at `/cloud_registered`.
- Target tracker odometry at `/target_tracker/odom`.

```bash
roslaunch trust_up_xtdrone mid360_px4_deploy.launch
```

Keep `auto_arm:=false` and `auto_offboard:=false` until bench, prop-off, tethered,
and geofence checks pass.

## Gazebo and Isaac-Grade Simulation

Gazebo-native replay is the robust visual inspection path:

```bash
roslaunch trust_up_xtdrone gazebo_paper_replay.launch scenario:=circle gui:=true
roslaunch trust_up_xtdrone gazebo_paper_replay.launch scenario:=figure8 gui:=true
```

Gazebo/XTDrone SITL remains the physics-and-PX4 reference path. For WSL:

```bash
roslaunch trust_up_xtdrone xtdrone_two_tello_profile_sitl_wsl_rviz.launch scenario:=figure8
```

Generate repeatable numerical regression metrics:

```bash
python3 /home/ysdeng/projects/xtdrone_trust_pursuit/trust_up_xtdrone/scripts/evaluate_headless_metrics.py \
  --output-dir /tmp/trust_up_metrics
```

Run the stricter paper-conformance audit before trusting a visualization:

```bash
python3 /home/ysdeng/projects/xtdrone_trust_pursuit/trust_up_xtdrone/scripts/audit_paper_conformance.py \
  --output-dir /tmp/trust_up_paper_audit
```

For a headless Gazebo smoke check in the same audit:

```bash
python3 /home/ysdeng/projects/xtdrone_trust_pursuit/trust_up_xtdrone/scripts/audit_paper_conformance.py \
  --output-dir /tmp/trust_up_paper_audit \
  --gazebo-smoke
```

Isaac Sim integration uses `config/isaac_sim.yaml` and the ROS1 bridge adapter:

```bash
roslaunch trust_up_xtdrone isaac_paper_target_pursuit.launch scenario:=figure8
```

Build the optional Isaac USD scene inside Isaac Sim's Python:

```bash
./python.sh /home/ysdeng/projects/xtdrone_trust_pursuit/trust_up_xtdrone/isaac/trust_up_isaac_scene.py
```

Detailed Gazebo/Isaac procedures are in `docs/SIMULATION_STACKS.md`.
The planner/safety-filter engineering mapping to Fast-Planner, EGO-Swarm,
GCOPTER, safe_control, and CBFpy is documented in `docs/ENGINEERING_BACKENDS.md`.

## ROS2 and Isaac

The ROS2 adapter is in `ros2/trust_up_xtdrone_ros2`. It reuses the same Python
core and is intended for XTDrone2, `ros1_bridge`, or Isaac ROS bridge setups.

```bash
export PYTHONPATH=/home/ysdeng/projects/xtdrone_trust_pursuit/trust_up_xtdrone/src:$PYTHONPATH
colcon build --packages-select trust_up_xtdrone_ros2
```

Isaac topic mapping notes are in `isaac/isaac_ros_bridge_topics.yaml` and
`config/isaac_sim.yaml`.
