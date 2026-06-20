# Engineering Backends

This package does not vendor large third-party planners directly.  The goal is
to keep the TRUST-UP/XTDrone reproduction buildable inside the existing Noetic
workspace while matching the interfaces and design boundaries of mature UAV
planning and safety-control stacks.

## Reference Projects

- Fast-Planner: https://github.com/HKUST-Aerial-Robotics/Fast-Planner
  - Interface matched here: kinodynamic reference generation, uniform cubic
    B-spline representation, gradient-based control-point optimization,
    dynamic feasibility checks, and high-level planning module separation.
- EGO-Swarm: https://github.com/ZJU-FAST-Lab/ego-planner-swarm
  - Interface matched here: decentralized multi-UAV target/agent separation,
    pairwise target clearance, and replanning-friendly local trajectory backend.
- GCOPTER: https://github.com/ZJU-FAST-Lab/GCOPTER
  - Interface matched here: geometry/dynamic-constraint awareness and explicit
    reportable trajectory feasibility metrics.
- safe_control: https://github.com/tkkim-robot/safe_control
  - Interface matched here: safety-critical controller object with named
    constraints and solver-swappable filtering.
- CBFpy: https://github.com/StanfordASL/cbfpy
  - Interface matched here: CBF/CLF-QP style structured constraints and
    explicit safety-filter diagnostics.

## Local Implementation

- `src/trust_up_xtdrone/planning.py`
  - `UniformCubicBspline`: cubic basis and derivative evaluation.
  - `FastPlannerBsplineOptimizer`: reference fit, smoothness, obstacle
    clearance, dynamic feasibility penalties, and iterative control-point
    projection.
- `src/trust_up_xtdrone/safety_filter.py`
  - `LinearSafetyConstraint`: named linear CBF/QP rows.
  - `CbfQpSafetyFilter`: active-set projection backend with a swappable solver
    boundary for future OSQP/cbfpy/safe_control integration.
- `src/trust_up_xtdrone/core.py`
  - `TargetTrajectory` uses the planning backend.
  - `TrustUpController` uses the safety-filter backend.

## Acceptance Policy

For the paper reproduction, hard acceptance is:

- paper pursuer-target clearance violations: `0`
- internal guard clearance violations: `0`
- target-target hull-clearance violations: `0`
- Gazebo GUI launches with `gazebo_ros/gzclient`
- the same config is used by headless, Gazebo replay, XTDrone SITL, and Isaac
  adapter paths

The headless metrics command is:

```bash
python3 /home/ysdeng/projects/xtdrone_trust_pursuit/trust_up_xtdrone/scripts/evaluate_headless_metrics.py \
  --output-dir /tmp/trust_up_metrics
```
