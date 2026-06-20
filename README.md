# RLCBF

<p align="center">
  <b>Reinforcement Learning + Control Barrier Functions for trustworthy UAV target pursuit.</b>
</p>

<p align="center">
  <a href="https://www.ros.org/"><img alt="ROS Noetic" src="https://img.shields.io/badge/ROS-Noetic-22314E?logo=ros&logoColor=white"></a>
  <a href="https://gazebosim.org/"><img alt="Gazebo Classic" src="https://img.shields.io/badge/Gazebo-Classic%2011-5C7AEA"></a>
  <a href="https://px4.io/"><img alt="PX4" src="https://img.shields.io/badge/PX4-SITL-005BBB"></a>
  <a href="https://github.com/robin-shaun/XTDrone"><img alt="XTDrone" src="https://img.shields.io/badge/XTDrone-supported-16A085"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-GPL--3.0-informational"></a>
</p>

RLCBF collects research prototypes for safe reinforcement-learning control with
control barrier functions. The newest stack, `trust_up_xtdrone`, is a
ROS/Gazebo/XTDrone reproduction-oriented implementation of the TRUST-UP target
pursuit experiments from 10.1016/j.aei.2026.104963 (https://www.sciencedirect.com/science/article/abs/pii/S1474034626006555).

The package focuses on a practical research workflow: reproduce the paper
geometry, inspect the result in Gazebo, validate constraints automatically, and
reuse the same interfaces for PX4/Mid360 deployment experiments.

## Highlights

| Area | What is included |
| --- | --- |
| Paper reproduction | Two pursuer UAVs, two target UAVs, two spherical obstacles, `dt=0.1`, `600` steps, circular and figure-8 target maneuvers. |
| Safety layer | Deterministic CBF-QP projection for pursuit shell, inter-UAV collision, obstacle avoidance, and input limiting. |
| Planning | Dependency-light Fast-Planner-style uniform cubic B-spline smoothing for target references. |
| Gazebo visualization | Real XTDrone/PX4 `iris` mesh visuals for pursuers and targets, safety shells, sensing shells, trails, and WSL-friendly launch path. |
| Regression audit | One-command conformance check for paper parameters, mesh resolution, and zero paper-constraint violations. |
| Deployment bridge | ROS/MAVROS interfaces for PX4 UAVs localized by Mid360/FAST-LIO-style odometry. |
| Isaac path | ROS1 bridge adapter and Isaac scene generation notes for higher-fidelity simulation workflows. |

## Repository Layout

```text
.
├── trust_up_xtdrone/          # ROS Noetic package for TRUST-UP target pursuit
│   ├── config/                # Paper, Isaac, and Mid360/PX4 configs
│   ├── launch/                # Gazebo, XTDrone, Isaac, and deployment launch files
│   ├── scripts/               # ROS nodes, Gazebo replay, audits, and adapters
│   ├── src/trust_up_xtdrone/  # Core CBF-QP, B-spline, and config utilities
│   ├── docs/                  # Simulation, backend, WSL, and interface notes
│   └── tests/                 # Lightweight solver/planner regression tests
├── positions1.csv             # Historical RLCBF experiment artifact
├── target_positions1.csv      # Historical RLCBF experiment artifact
└── *.7z / *.rar               # Archived earlier RLCBF prototypes
```

## TRUST-UP XTDrone Quick Start

Assumed local paths in this workstation:

```bash
/home/ysdeng/work/catkin_ws
/home/ysdeng/projects/XTDrone
/home/ysdeng/work/PX4-Autopilot
```

Build:

```bash
cd /home/ysdeng/work/catkin_ws
ln -sfn /path/to/RLCBF/trust_up_xtdrone src/trust_up_xtdrone
catkin_make
source /path/to/RLCBF/trust_up_xtdrone/setup_trust_up_noetic.bash
```

Run the Gazebo-native paper replay:

```bash
roslaunch trust_up_xtdrone gazebo_paper_replay.launch scenario:=circle gui:=true
roslaunch trust_up_xtdrone gazebo_paper_replay.launch scenario:=figure8 gui:=true
```

Run the strict paper-conformance audit:

```bash
python3 trust_up_xtdrone/scripts/audit_paper_conformance.py \
  --output-dir /tmp/trust_up_paper_audit \
  --gazebo-smoke
```

Run the XTDrone/PX4 SITL path:

```bash
roslaunch trust_up_xtdrone xtdrone_two_tello_profile_sitl.launch scenario:=circle
roslaunch trust_up_xtdrone xtdrone_two_tello_profile_sitl.launch scenario:=figure8
```

For WSL sessions where the Gazebo client is unreliable:

```bash
roslaunch trust_up_xtdrone xtdrone_two_tello_profile_sitl_wsl_rviz.launch scenario:=circle
```

## Current Validation Snapshot

The latest local smoke tests before upload passed:

| Check | Result |
| --- | --- |
| Python compile and unit tests | PASS |
| `catkin_make` in `/home/ysdeng/work/catkin_ws` | PASS |
| Paper conformance audit, circle | `paper_total_violations = 0` |
| Paper conformance audit, figure-8 | `paper_total_violations = 0` |
| Real UAV mesh audit | `iris.stl` + `iris_prop_*.dae`, no fallback primitives |
| Gazebo headless smoke | PASS |
| Gazebo GUI smoke | PASS, no `Missing model.config`, no bad GUI plugin warning |

## Paper Experiment Parameters

The default `trust_up_xtdrone/config/paper_target_pursuit.yaml` keeps the paper
settings as first-class parameters:

- two pursuit UAVs and two target UAVs
- obstacle centers `[4.70, 3.25, 3.00]` and `[-4.20, 3.00, 4.75]`
- obstacle radius `0.3 m`
- collision radius `r_i = 0.5 m`
- sensing radius `R_i = 1.0 m`
- target references using `5 sin(0.1t)`, `5 cos(0.1t)`, and `5 sin(0.2t)`
- `dt = 0.1`, `steps = 600`

For physical XTDrone/PX4 visualization, the UAV hull radius is added separately
as center-to-center padding. This keeps the paper radii explicit while avoiding
unrealistic body overlap in simulation.

## Documentation

- [TRUST-UP package README](trust_up_xtdrone/README.md)
- [Gazebo and Isaac simulation stacks](trust_up_xtdrone/docs/SIMULATION_STACKS.md)
- [Engineering backend mapping](trust_up_xtdrone/docs/ENGINEERING_BACKENDS.md)
- [PX4/Mid360 interfaces](trust_up_xtdrone/docs/INTERFACES.md)
- [WSL visualization notes](trust_up_xtdrone/docs/WSL_VISUALIZATION.md)

## Notes

Gazebo Classic reached end-of-life in January 2025. This repository keeps a
Gazebo Classic path because XTDrone and PX4 SITL workflows still commonly depend
on it, while also documenting Isaac/ROS bridge integration for migration.

The repository is released under GPL-3.0. External dependencies such as ROS,
PX4, XTDrone, Gazebo, Isaac Sim, MAVROS, and sensor drivers retain their own
licenses.
