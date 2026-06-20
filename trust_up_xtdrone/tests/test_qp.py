import numpy as np

from trust_up_xtdrone.adaptive import AdaptiveParameters, AdaptiveRobustnessEstimator
from trust_up_xtdrone.qp import project_to_halfspaces
from trust_up_xtdrone.core import AgentState, SafetyParameters, TargetTrajectory, TrustUpController, VehicleLimits
from trust_up_xtdrone.planning import BsplineOptimizerConfig, FastPlannerBsplineOptimizer, UniformCubicBspline
from trust_up_xtdrone.safety_filter import CbfQpSafetyFilter, LinearSafetyConstraint


def test_projection_keeps_feasible_nominal():
    result = project_to_halfspaces([1.0, 0.0], [[1.0, 0.0]], [0.5])
    assert result.feasible
    assert np.allclose(result.value, [1.0, 0.0])


def test_projection_repairs_single_constraint():
    result = project_to_halfspaces([0.0, 0.0], [[1.0, 0.0]], [0.5])
    assert result.feasible
    assert result.value[0] >= 0.5 - 1.0e-8


def test_projection_handles_corner():
    result = project_to_halfspaces([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], [1.0, 2.0])
    assert result.feasible
    assert np.allclose(result.value, [1.0, 2.0])


def test_controller_enforces_target_inner_shell():
    controller = TrustUpController(
        SafetyParameters(collision_radius=0.5, sensing_radius=1.0, desired_range=0.75),
        VehicleLimits(max_speed=2.0, max_vertical_speed=1.0, max_accel=1.2),
    )
    pursuer = AgentState.from_xyz([0.3, 0.0, 0.0])
    target = AgentState.from_xyz([0.0, 0.0, 0.0], name="target")
    cmd, diag = controller.step(pursuer, target, [], 0.1)
    assert diag["barriers"]["target_collision"]["distance"] < 0.5
    assert cmd[0] > 0.0


def test_controller_expands_shell_by_both_uav_radii():
    controller = TrustUpController(
        SafetyParameters(collision_radius=0.5, sensing_radius=1.0, desired_range=0.75, use_object_radius=True),
        VehicleLimits(max_speed=4.0, max_vertical_speed=2.0, max_accel=3.0),
    )
    pursuer = AgentState.from_xyz([1.2, 0.0, 0.0], radius=0.28)
    target = AgentState.from_xyz([0.0, 0.0, 0.0], radius=0.28, name="target")
    _, diag = controller.step(pursuer, target, [], 0.1)
    assert np.isclose(diag["barriers"]["target_collision"]["radius"], 1.06)
    assert np.isclose(diag["barriers"]["sensing"]["radius"], 1.56)


def test_guard_shell_tightens_without_changing_paper_shell():
    safety = SafetyParameters(
        collision_radius=0.5,
        sensing_radius=1.8,
        desired_range=1.15,
        use_object_radius=True,
        tracking_inner_margin=0.12,
        tracking_outer_margin=0.18,
    )
    pursuer = AgentState.from_xyz([0.0, 0.0, 0.0], radius=0.4)
    target = AgentState.from_xyz([1.6, 0.0, 0.0], radius=0.4, name="target")
    assert np.isclose(safety.collision_bound(pursuer, target), 1.3)
    assert np.isclose(safety.sensing_bound(pursuer, target), 2.6)
    assert np.isclose(safety.enforced_collision_bound(pursuer, target), 1.42)
    assert np.isclose(safety.enforced_sensing_bound(pursuer, target), 2.42)


def test_adaptive_estimator_adds_margin_under_residual_and_pressure():
    safety = SafetyParameters(
        collision_radius=0.5,
        sensing_radius=1.3,
        desired_range=0.8,
        use_object_radius=True,
        robustness_margin=0.04,
    )
    estimator = AdaptiveRobustnessEstimator(
        AdaptiveParameters(
            adaptation_gain=1.2,
            residual_filter_gain=1.0,
            pressure_margin_gain=0.08,
            residual_margin_gain=0.10,
            max_margin=0.5,
        )
    )
    pursuer = AgentState.from_xyz([0.95, 0.0, 0.0], vel=[0.0, 0.0, 0.0], radius=0.2)
    target = AgentState.from_xyz([0.0, 0.0, 0.0], vel=[0.0, 0.0, 0.0], radius=0.2, name="target")
    estimator.update(pursuer, target, [], [0.0, 0.0, 0.0], 0.1, safety)
    pursuer.velocity = np.array([0.3, 0.0, 0.0])
    estimate = estimator.update(pursuer, target, [], [0.0, 0.0, 0.0], 0.1, safety)
    assert estimate.residual_norm > 0.0
    assert estimate.clearance_pressure > 0.0
    assert estimate.margin_scale > 0.0


def test_controller_reports_adaptive_robustness_margin():
    controller = TrustUpController(
        SafetyParameters(collision_radius=0.5, sensing_radius=1.4, desired_range=0.8, use_object_radius=True),
        VehicleLimits(max_speed=3.0, max_vertical_speed=2.0, max_accel=2.0),
        adaptive=AdaptiveRobustnessEstimator(AdaptiveParameters(pressure_margin_gain=0.08)),
    )
    pursuer = AgentState.from_xyz([1.05, 0.0, 0.0], radius=0.25)
    target = AgentState.from_xyz([0.0, 0.0, 0.0], radius=0.25, name="target")
    _, diag = controller.step(pursuer, target, [], 0.1)
    assert diag["adaptive"]["clearance_pressure"] > 0.0
    assert diag["adaptive_robustness_margin"] > controller.safety.robustness_margin


def test_bspline_target_smoothing_limits_integrated_acceleration():
    trajectory = TargetTrajectory(
        "figure8",
        [],
        smoothing={
            "enabled": True,
            "knot_dt": 0.8,
            "horizon_control_points": 8,
            "iterations": 4,
            "max_ref_speed": 1.45,
            "max_ref_accel": 1.05,
            "limit_integrated_target": True,
        },
    )
    state = AgentState.from_xyz(trajectory.reference(0, 0.0)[0])
    for step in range(80):
        state = trajectory.integrate(0, state, 0.1 * step, 0.1)
        assert np.linalg.norm(state.acceleration) <= 1.05 + 1.0e-9


def test_target_pair_avoidance_pushes_targets_apart():
    trajectory = TargetTrajectory(
        "circle",
        [],
        smoothing={
            "enabled": True,
            "target_pair_clearance": 1.0,
            "target_pair_activation_margin": 1.0,
            "target_pair_gain": 1.0,
            "target_pair_hard_gain": 5.0,
            "target_pair_damping": 0.0,
            "max_ref_accel": 5.0,
            "max_ref_jerk": 0.0,
        },
    )
    first = AgentState.from_xyz([0.0, 0.0, 0.0], radius=0.4)
    second = AgentState.from_xyz([1.0, 0.0, 0.0], radius=0.4)
    accels = trajectory.target_pair_accelerations([first, second])
    assert accels[0][0] < 0.0
    assert accels[1][0] > 0.0


def test_fast_planner_backend_projects_bspline_dynamics():
    raw = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
            [16.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    optimizer = FastPlannerBsplineOptimizer(
        BsplineOptimizerConfig(
            iterations=1,
            step_size=0.0,
            max_ref_speed=1.0,
            max_ref_accel=1.0,
            dynamic_projection_iterations=100,
        )
    )
    ctrl = optimizer.optimize(raw, knot_dt=1.0, obstacles=[])
    assert np.max(np.linalg.norm(np.diff(ctrl, axis=0), axis=1)) <= 1.0 + 1.0e-6


def test_uniform_bspline_evaluates_position_velocity_acceleration():
    ctrl = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    pos, vel, acc = UniformCubicBspline.evaluate(ctrl, 0.5, 1.0)
    assert np.allclose(pos, [1.5, 0.0, 0.0])
    assert np.allclose(vel, [1.0, 0.0, 0.0])
    assert np.allclose(acc, [0.0, 0.0, 0.0])


def test_structured_cbf_safety_filter_reports_active_constraint_name():
    filt = CbfQpSafetyFilter("unit_test")
    result = filt.project(
        np.array([0.0, 0.0]),
        [LinearSafetyConstraint.make("x_min", [1.0, 0.0], 1.0, kind="test")],
    )
    assert result.feasible
    assert np.allclose(result.value, [1.0, 0.0])
    assert result.active_names() == ["x_min"]
    assert result.max_violation_by_kind()["test"] <= 1.0e-8
