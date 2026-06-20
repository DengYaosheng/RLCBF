# Gazebo and Isaac Simulation Stacks

## Gazebo-Native Paper Replay

Use this path first when you need to see the paper experiments in Gazebo,
especially under WSL.  It runs the same TRUST-UP controller and target
trajectory generator, but moves kinematic XTDrone-derived UAV visuals directly
through Gazebo services.  This removes PX4/MAVROS startup failure from the
visualization loop while preserving the paper geometry and constraints.
The launch starts Gazebo ROS only on the server side and uses raw `gzclient` for
the GUI, avoiding the Gazebo Classic warning where `gazebo_ros/gzclient` loads a
SystemPlugin as a GUI plugin.

```bash
source /home/ysdeng/projects/xtdrone_trust_pursuit/trust_up_xtdrone/setup_trust_up_noetic.bash
roslaunch trust_up_xtdrone gazebo_paper_replay.launch scenario:=circle gui:=true
roslaunch trust_up_xtdrone gazebo_paper_replay.launch scenario:=figure8 gui:=true
```

Expected Gazebo scene:

- red and magenta XTDrone UAVs: pursuers
- blue and green XTDrone UAVs: target UAVs
- all four UAVs are spawned from resolved XTDrone/PX4 `iris` mesh visuals
  (`iris.stl` plus propeller `.dae` meshes); the audit fails if fallback
  primitive bodies are used
- black spheres: static obstacles
- dark yellow shell: collision/safety radius
- light yellow shell: sensing radius
- colored dots: time history trails

For WSLg, keep software rendering disabled.  This is the default because forcing
`LIBGL_ALWAYS_SOFTWARE=1` can open a Gazebo window that renders completely black:

```bash
roslaunch trust_up_xtdrone gazebo_paper_replay.launch scenario:=circle software_rendering:=false
```

If the Gazebo window still opens as a black 3D viewport, try OGRE copy-mode
render targets:

```bash
roslaunch trust_up_xtdrone gazebo_paper_replay.launch scenario:=circle software_rendering:=false ogre_rtt_mode:=Copy verbose:=true
```

For old WSL + external X server setups, start the X server before launching and
only then try software rendering:

```bash
export DISPLAY=${DISPLAY:-:0}
roslaunch trust_up_xtdrone gazebo_paper_replay.launch scenario:=circle software_rendering:=true ogre_rtt_mode:=Copy verbose:=true
```

The replay publishes live acceptance counters:

```bash
rostopic echo /trust_up/gazebo_replay/status
```

For slower WSL OpenGL sessions, disable only trail dots while keeping the UAVs
and safety/sensing shells:

```bash
roslaunch trust_up_xtdrone gazebo_paper_replay.launch scenario:=figure8 enable_trails:=false
```

## Gazebo/XTDrone SITL

Use this path for PX4 dynamics, MAVROS setpoint streaming, and XTDrone vehicle
interfaces:

```bash
source /home/ysdeng/projects/xtdrone_trust_pursuit/trust_up_xtdrone/setup_trust_up_noetic.bash
roslaunch trust_up_xtdrone xtdrone_two_tello_profile_sitl_wsl_rviz.launch scenario:=circle
roslaunch trust_up_xtdrone xtdrone_two_tello_profile_sitl_wsl_rviz.launch scenario:=figure8
```

WSL uses headless Gazebo plus RViz.  Native Linux can use:

```bash
roslaunch trust_up_xtdrone xtdrone_two_tello_profile_sitl.launch scenario:=circle gui:=true
```

Generate numerical acceptance metrics for both paper scenarios:

```bash
python3 /home/ysdeng/projects/xtdrone_trust_pursuit/trust_up_xtdrone/scripts/evaluate_headless_metrics.py \
  --output-dir /tmp/trust_up_metrics
```

Run the full paper-conformance audit after any controller, planner, or Gazebo
visual change:

```bash
python3 /home/ysdeng/projects/xtdrone_trust_pursuit/trust_up_xtdrone/scripts/audit_paper_conformance.py \
  --output-dir /tmp/trust_up_paper_audit
```

The current acceptance gates are:

- paper target-pursuit shell violations: `0`
- pursuer-pursuer paper collision violations: `0`
- pursuer-obstacle paper collision violations: `0`
- target-target hull-clearance violations: `0`
- target-obstacle contact violations: `0`
- target minimum hull clearance greater than `trajectory_smoothing.target_pair_clearance`
- command acceleration/jerk metrics reported for regression comparison

The planner and CBF-QP safety filter are structured after mature open-source
UAV stacks; see `docs/ENGINEERING_BACKENDS.md`.

## Isaac Sim ROS1 Bridge

Use this path for high-fidelity rendering, RTX sensors, and Isaac-managed
vehicle dynamics while preserving the same TRUST-UP controller:

```bash
source /home/ysdeng/projects/xtdrone_trust_pursuit/trust_up_xtdrone/setup_trust_up_noetic.bash
roslaunch trust_up_xtdrone isaac_paper_target_pursuit.launch scenario:=figure8
```

Expected Isaac topics are configured in:

```text
/home/ysdeng/projects/xtdrone_trust_pursuit/trust_up_xtdrone/config/isaac_sim.yaml
```

The adapter maps Isaac odometry into MAVROS-like topics consumed by the existing
controller:

- `/isaac/iris_0/odom` -> `/iris_0/mavros/local_position/odom`
- `/isaac/iris_1/odom` -> `/iris_1/mavros/local_position/odom`
- `/xtdrone/iris_i/cmd_vel_enu` -> `/isaac/iris_i/cmd_vel_enu`

By default the TRUST-UP paper target server owns target trajectories and sends
target pose/velocity commands to Isaac:

- `/trust_up/target_0/odom` -> `/isaac/trust_target_0/pose_cmd`
- `/trust_up/target_1/odom` -> `/isaac/trust_target_1/pose_cmd`

If Isaac owns target dynamics, set each target entry in `config/isaac_sim.yaml`
to `source: isaac`; then Isaac odometry is republished to `/trust_up/target_i/odom`.

## Isaac Scene Generation

Run inside Isaac Sim's Python:

```bash
./python.sh /home/ysdeng/projects/xtdrone_trust_pursuit/trust_up_xtdrone/isaac/trust_up_isaac_scene.py \
  --config /home/ysdeng/projects/xtdrone_trust_pursuit/trust_up_xtdrone/config/paper_target_pursuit.yaml \
  --isaac-config /home/ysdeng/projects/xtdrone_trust_pursuit/trust_up_xtdrone/config/isaac_sim.yaml
```

The script writes `/tmp/trust_up_target_pursuit.usd` by default.  Set
`scene.pursuer_usd` and `scene.target_usd` in `config/isaac_sim.yaml` to replace
the lightweight quadrotor primitives with your own USD vehicle assets.
