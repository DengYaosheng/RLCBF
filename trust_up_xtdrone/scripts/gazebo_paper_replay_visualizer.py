#!/usr/bin/env python3
"""Native Gazebo replay for the TRUST-UP paper target-pursuit experiments.

This node is intentionally independent of PX4/MAVROS.  It runs the same
TRUST-UP controller and target dynamics as the headless evaluator, then moves
XTDrone-derived visual UAVs directly in Gazebo.  The result is a reliable WSL
visualization path for the paper's circle and figure-8 experiments.
"""

import json
import math
import os
import sys
from collections import defaultdict, deque
from typing import Dict, List, Sequence, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PKG_DIR, "src"))
sys.path.insert(0, SCRIPT_DIR)

import rospy  # noqa: E402
from gazebo_msgs.msg import ModelState  # noqa: E402
from gazebo_msgs.srv import DeleteModel, GetWorldProperties, SetModelState, SpawnModel  # noqa: E402
from geometry_msgs.msg import Point, Pose, Quaternion  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from run_headless_paper_experiment import smooth_command  # noqa: E402
from spawn_paper_world import sphere_sdf, spawn_model, wire_shell_sdf, xtdrone_visual_uav_sdf  # noqa: E402
from trust_up_xtdrone.config_io import (  # noqa: E402
    default_config_path,
    load_yaml,
    obstacles_from_config,
    policy_from_config,
    safety_from_config,
    scenario_offset,
    target_initial_states,
    limits_from_config,
)
from trust_up_xtdrone.core import AgentState, NominalPolicy, TargetTrajectory, TrustUpController  # noqa: E402


PURSUER_COLORS = ["0.96 0.08 0.06 1.0", "0.88 0.05 0.86 1.0"]
TARGET_COLORS = ["0.02 0.30 1.0 1.0", "0.05 0.78 0.24 1.0"]
TRAIL_COLORS = {
    "pursuer_0": "1.0 0.08 0.04 0.92",
    "pursuer_1": "0.95 0.08 0.92 0.92",
    "target_0": "0.02 0.34 1.0 0.88",
    "target_1": "0.05 0.82 0.28 0.88",
}


def yaw_from_velocity(velocity: Sequence[float]) -> float:
    vx = float(velocity[0])
    vy = float(velocity[1])
    if math.hypot(vx, vy) < 1.0e-4:
        return 0.0
    return math.atan2(vy, vx)


def quaternion_from_yaw(yaw: float) -> Quaternion:
    return Quaternion(0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


def make_odom(state: AgentState, frame_id: str) -> Odometry:
    msg = Odometry()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.child_frame_id = state.name
    msg.pose.pose.position = Point(float(state.position[0]), float(state.position[1]), float(state.position[2]))
    msg.pose.pose.orientation = quaternion_from_yaw(yaw_from_velocity(state.velocity))
    msg.twist.twist.linear.x = float(state.velocity[0])
    msg.twist.twist.linear.y = float(state.velocity[1])
    msg.twist.twist.linear.z = float(state.velocity[2])
    return msg


def clone_state(state: AgentState) -> AgentState:
    return AgentState(
        state.position.copy(),
        state.velocity.copy(),
        state.acceleration.copy(),
        radius=state.radius,
        stamp=state.stamp,
        name=state.name,
    )


def interpolate_state(first: AgentState, second: AgentState, alpha: float, duration: float) -> AgentState:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    beta = 1.0 - alpha
    duration = max(float(duration), 1.0e-6)
    h00 = 2.0 * alpha ** 3 - 3.0 * alpha ** 2 + 1.0
    h10 = alpha ** 3 - 2.0 * alpha ** 2 + alpha
    h01 = -2.0 * alpha ** 3 + 3.0 * alpha ** 2
    h11 = alpha ** 3 - alpha ** 2
    pos = (
        h00 * first.position
        + h10 * duration * first.velocity
        + h01 * second.position
        + h11 * duration * second.velocity
    )
    vel = (
        (6.0 * alpha ** 2 - 6.0 * alpha) * first.position / duration
        + (3.0 * alpha ** 2 - 4.0 * alpha + 1.0) * first.velocity
        + (-6.0 * alpha ** 2 + 6.0 * alpha) * second.position / duration
        + (3.0 * alpha ** 2 - 2.0 * alpha) * second.velocity
    )
    return AgentState(
        pos,
        vel,
        beta * first.acceleration + alpha * second.acceleration,
        radius=second.radius,
        stamp=beta * first.stamp + alpha * second.stamp,
        name=second.name,
    )


def tinted_uav_sdf(name: str, source_sdf: str, rgba: str) -> str:
    """Return an XTDrone visual UAV SDF with collision/plugins stripped and tint applied."""
    sdf = xtdrone_visual_uav_sdf(name, source_sdf)
    # Gazebo keeps model geometry/materials from the original SDF.  A diffuse
    # replacement is a simple deterministic tint for paper-style identity.
    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(sdf)
        for visual in root.findall(".//visual"):
            material = visual.find("material")
            if material is None:
                material = ET.SubElement(visual, "material")
            ambient = material.find("ambient")
            if ambient is None:
                ambient = ET.SubElement(material, "ambient")
            diffuse = material.find("diffuse")
            if diffuse is None:
                diffuse = ET.SubElement(material, "diffuse")
            ambient.text = rgba
            diffuse.text = rgba
        return ET.tostring(root, encoding="unicode")
    except Exception:
        return sdf


class GazeboPaperReplay:
    def __init__(self):
        self.cfg_path = rospy.get_param("~config_file", default_config_path())
        self.cfg = load_yaml(self.cfg_path)
        if rospy.has_param("~scenario"):
            self.cfg.setdefault("scenario", {})["name"] = rospy.get_param("~scenario")
        self.scenario = self.cfg.get("scenario", {}).get("name", "circle")
        self.frame_id = rospy.get_param("~frame_id", "world")
        self.loop = bool(rospy.get_param("~loop", True))
        self.visual_rate_hz = float(rospy.get_param("~visual_rate_hz", 60.0))
        self.visual_yaw_rate = float(rospy.get_param("~visual_yaw_rate", 3.0))
        self.trail_stride = max(int(rospy.get_param("~trail_stride", 4)), 1)
        self.max_trail_points = max(int(rospy.get_param("~max_trail_points", 95)), 0)
        self.enable_trails = bool(rospy.get_param("~enable_trails", True))
        self.spawn_visual_volumes = bool(rospy.get_param("~spawn_visual_volumes", True))
        self.shutdown_when_done = bool(rospy.get_param("~shutdown_when_done", False))

        self.dt = float(self.cfg.get("paper", {}).get("dt", 0.1))
        self.steps = int(self.cfg.get("paper", {}).get("steps", 600))
        max_replay_steps = int(rospy.get_param("~max_replay_steps", 0))
        if max_replay_steps > 0:
            self.steps = min(self.steps, max_replay_steps)
        self.safety = safety_from_config(self.cfg)
        self.limits = limits_from_config(self.cfg)
        self.policy_params = policy_from_config(self.cfg)
        self.obstacles = obstacles_from_config(self.cfg)
        self.trajectory = self.make_trajectory()
        self.source_sdf = self.cfg.get("target_visual", {}).get(
            "xtdrone_sdf",
            "/home/ysdeng/projects/XTDrone/sitl_config/models/iris_zhihang/iris_zhihang.sdf",
        )
        if not os.path.exists(os.path.expanduser(os.path.expandvars(self.source_sdf))):
            raise RuntimeError("XTDrone UAV SDF does not exist: %s" % self.source_sdf)

        self.spawn_srv = None
        self.delete_srv = None
        self.set_state_srv = None
        self.existing_names = set()
        self.visual_yaws: Dict[str, Tuple[float, float]] = {}
        self.trail_names: Dict[Tuple[str, int], deque] = defaultdict(deque)
        self.trail_counter = 0

        self.status_pub = rospy.Publisher("/trust_up/gazebo_replay/status", String, queue_size=10)
        self.target_odom_pubs = [
            rospy.Publisher("/trust_up/target_%d/odom" % idx, Odometry, queue_size=10)
            for idx in range(len(self.cfg.get("targets", [])))
        ]
        self.pursuer_odom_pubs = [
            rospy.Publisher("/%s/mavros/local_position/odom" % item.get("name", "iris_%d" % idx), Odometry, queue_size=10)
            for idx, item in enumerate(self.cfg.get("pursuers", []))
        ]

        self.connect_gazebo()
        self.reset_experiment(spawn_static=True)

    def make_trajectory(self) -> TargetTrajectory:
        return TargetTrajectory(
            self.scenario,
            self.obstacles,
            offset=scenario_offset(self.cfg),
            potential_gain=float(self.cfg.get("scenario", {}).get("target_potential_gain", 1.0)),
            potential_clip=float(self.cfg.get("scenario", {}).get("target_potential_clip", 1.5)),
            use_dimension_consistent_reference_velocity=bool(
                self.cfg.get("scenario", {}).get("use_dimension_consistent_reference_velocity", True)
            ),
            reference_scale=float(self.cfg.get("scenario", {}).get("reference_scale", 1.0)),
            angular_rate=float(self.cfg.get("scenario", {}).get("angular_rate", 0.1)),
            smoothing=self.cfg.get("trajectory_smoothing", {}),
        )

    def initial_targets(self) -> List[AgentState]:
        targets = target_initial_states(self.cfg)
        if bool(self.cfg.get("scenario", {}).get("initialize_targets_from_reference", True)):
            initialized = []
            for i, target in enumerate(targets):
                pos, vel, acc = self.trajectory.reference(i, 0.0)
                initialized.append(AgentState.from_xyz(pos, vel, acc, radius=target.radius, name=target.name))
            targets = initialized
        return targets

    def initial_pursuers(self, targets: Sequence[AgentState]) -> List[AgentState]:
        default_uav_radius = float(self.cfg.get("physical_envelope", {}).get("uav_radius", 0.0))
        pursuer_cfg = self.cfg.get("pursuers", [])
        pursuers = []
        for i, target in enumerate(targets):
            direction = target.velocity.copy()
            if float(np.linalg.norm(direction)) < 1.0e-6:
                direction = target.position - self.obstacles[0].position
            direction = direction / max(float(np.linalg.norm(direction)), 1.0e-6)
            pursuer_radius = float(pursuer_cfg[i].get("radius", default_uav_radius)) if i < len(pursuer_cfg) else default_uav_radius
            pursuer_name = str(pursuer_cfg[i].get("name", "iris_%d" % i)) if i < len(pursuer_cfg) else "iris_%d" % i
            bootstrap = AgentState.from_xyz([0.0, 0.0, 0.0], radius=pursuer_radius)
            initial_range = self.safety.desired_bound(bootstrap, target)
            pursuers.append(
                AgentState.from_xyz(
                    target.position - initial_range * direction,
                    radius=pursuer_radius,
                    name=pursuer_name,
                )
            )
        return pursuers

    def connect_gazebo(self):
        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        rospy.wait_for_service("/gazebo/delete_model")
        rospy.wait_for_service("/gazebo/set_model_state")
        self.spawn_srv = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
        self.delete_srv = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
        self.set_state_srv = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        try:
            rospy.wait_for_service("/gazebo/get_world_properties", timeout=4.0)
            world_srv = rospy.ServiceProxy("/gazebo/get_world_properties", GetWorldProperties)
            self.existing_names = set(world_srv().model_names)
        except Exception:
            self.existing_names = set()

    def spawn_static_scene(self):
        for obs in self.obstacles:
            spawn_model(
                self.spawn_srv,
                self.delete_srv,
                obs.name,
                sphere_sdf(obs.name, obs.radius, "0.015 0.015 0.018 1.0", True),
                obs.position,
                self.existing_names,
            )

    def spawn_uavs_and_shells(self):
        use_object_radius = bool(self.cfg.get("safety", {}).get("use_object_radius", False))
        collision_radius = float(self.safety.collision_radius)
        sensing_radius = float(self.safety.sensing_radius)
        for idx, pursuer in enumerate(self.pursuers):
            spawn_model(
                self.spawn_srv,
                self.delete_srv,
                pursuer.name,
                tinted_uav_sdf(pursuer.name, self.source_sdf, PURSUER_COLORS[idx % len(PURSUER_COLORS)]),
                pursuer.position,
                self.existing_names,
            )
        for idx, target in enumerate(self.targets):
            color = TARGET_COLORS[idx % len(TARGET_COLORS)]
            shell_padding = 0.0
            if use_object_radius:
                peer_radius = self.pursuers[min(idx, len(self.pursuers) - 1)].radius if self.pursuers else 0.0
                shell_padding = peer_radius + float(target.radius)
            spawn_model(
                self.spawn_srv,
                self.delete_srv,
                target.name,
                tinted_uav_sdf(target.name, self.source_sdf, color),
                target.position,
                self.existing_names,
            )
            self.spawn_target_shell(target, collision_radius + shell_padding, sensing_radius + shell_padding)

    def spawn_target_shell(self, target: AgentState, collision_radius: float, sensing_radius: float):
        collision_shell = "%s_collision_shell" % target.name
        sensing_shell = "%s_sensing_shell" % target.name
        spawn_model(
            self.spawn_srv,
            self.delete_srv,
            collision_shell,
            wire_shell_sdf(collision_shell, collision_radius, "1.0 0.58 0.02 1.0", False, tube_radius=0.018, segments=64),
            target.position,
            self.existing_names,
        )
        spawn_model(
            self.spawn_srv,
            self.delete_srv,
            sensing_shell,
            wire_shell_sdf(sensing_shell, sensing_radius, "1.0 0.93 0.18 1.0", False, tube_radius=0.012, segments=64),
            target.position,
            self.existing_names,
        )
        if self.spawn_visual_volumes:
            collision_volume = "%s_collision_volume" % target.name
            sensing_volume = "%s_sensing_volume" % target.name
            spawn_model(
                self.spawn_srv,
                self.delete_srv,
                collision_volume,
                sphere_sdf(
                    collision_volume,
                    collision_radius,
                    "1.0 0.50 0.03 0.22",
                    False,
                    visual_only=True,
                    transparency=0.80,
                ),
                target.position,
                self.existing_names,
            )
            spawn_model(
                self.spawn_srv,
                self.delete_srv,
                sensing_volume,
                sphere_sdf(
                    sensing_volume,
                    sensing_radius,
                    "1.0 0.96 0.16 0.13",
                    False,
                    visual_only=True,
                    transparency=0.90,
                ),
                target.position,
                self.existing_names,
            )

    def reset_experiment(self, *, spawn_static: bool = False):
        self.trajectory = self.make_trajectory()
        self.targets = self.initial_targets()
        self.pursuers = self.initial_pursuers(self.targets)
        self.controllers = [
            TrustUpController(self.safety, self.limits, NominalPolicy(self.safety, self.limits, self.policy_params))
            for _ in self.targets
        ]
        self.filtered_commands = [None for _ in self.targets]
        self.step_idx = 0
        self.sim_t = 0.0
        self.metrics = {
            "paper_violations": 0,
            "guard_violations": 0,
            "pursuer_collision_violations": 0,
            "target_pair_violations": 0,
            "min_pair_clearance": None,
            "min_target_clearance": None,
            "max_target_clearance": None,
            "qp_max_violation": 0.0,
        }
        self.visual_yaws = {}
        self.prev_targets = [clone_state(s) for s in self.targets]
        self.next_targets = [clone_state(s) for s in self.targets]
        self.prev_pursuers = [clone_state(s) for s in self.pursuers]
        self.next_pursuers = [clone_state(s) for s in self.pursuers]
        self.last_step_time = rospy.Time.now().to_sec()
        if spawn_static:
            self.spawn_static_scene()
            self.spawn_uavs_and_shells()
            # Gazebo inserts SDF models asynchronously; a short grace period
            # avoids racing set_model_state against model construction.
            rospy.sleep(0.5)
        else:
            self.clear_trails()
        self.move_all(self.pursuers, self.targets)
        rospy.loginfo("Gazebo TRUST-UP replay reset: scenario=%s, steps=%d, dt=%.3f", self.scenario, self.steps, self.dt)

    def clear_trails(self):
        for names in self.trail_names.values():
            while names:
                name = names.popleft()
                try:
                    self.delete_srv(name)
                    self.existing_names.discard(name)
                except Exception:
                    pass
        self.trail_names.clear()

    def smooth_visual_yaw(self, model_name: str, state: AgentState) -> float:
        speed_xy = math.hypot(float(state.velocity[0]), float(state.velocity[1]))
        previous = self.visual_yaws.get(model_name)
        if speed_xy < 1.0e-4 and previous is not None:
            desired = previous[0]
        else:
            desired = yaw_from_velocity(state.velocity)
        if previous is None:
            self.visual_yaws[model_name] = (desired, float(state.stamp))
            return desired
        prev_yaw, prev_stamp = previous
        dt = max(float(state.stamp) - float(prev_stamp), 1.0 / max(self.visual_rate_hz, 1.0))
        error = (desired - prev_yaw + math.pi) % (2.0 * math.pi) - math.pi
        limit = max(self.visual_yaw_rate, 0.0) * dt
        if limit > 0.0:
            error = float(np.clip(error, -limit, limit))
        yaw = prev_yaw + error
        self.visual_yaws[model_name] = (yaw, float(state.stamp))
        return yaw

    def move_model(self, model_name: str, state: AgentState):
        model_state = ModelState()
        model_state.model_name = model_name
        model_state.reference_frame = "world"
        model_state.pose.position = Point(float(state.position[0]), float(state.position[1]), float(state.position[2]))
        model_state.pose.orientation = quaternion_from_yaw(self.smooth_visual_yaw(model_name, state))
        model_state.twist.linear.x = float(state.velocity[0])
        model_state.twist.linear.y = float(state.velocity[1])
        model_state.twist.linear.z = float(state.velocity[2])
        for attempt in range(3):
            try:
                self.set_state_srv(model_state)
                return
            except Exception as exc:
                if attempt == 2:
                    rospy.logwarn_throttle(4.0, "Gazebo set_model_state delayed for %s: %s", model_name, exc)
                rospy.rostime.wallsleep(0.02)

    def move_target_shells(self, target: AgentState):
        for suffix in ("collision_shell", "sensing_shell", "collision_volume", "sensing_volume"):
            name = "%s_%s" % (target.name, suffix)
            if name in self.existing_names:
                self.move_model(name, target)

    def move_all(self, pursuers: Sequence[AgentState], targets: Sequence[AgentState]):
        for state in pursuers:
            self.move_model(state.name, state)
        for state in targets:
            self.move_model(state.name, state)
            self.move_target_shells(state)

    def publish_odometry(self, pursuers: Sequence[AgentState], targets: Sequence[AgentState]):
        for idx, state in enumerate(targets):
            if idx < len(self.target_odom_pubs):
                self.target_odom_pubs[idx].publish(make_odom(state, self.frame_id))
        for idx, state in enumerate(pursuers):
            if idx < len(self.pursuer_odom_pubs):
                self.pursuer_odom_pubs[idx].publish(make_odom(state, self.frame_id))

    def other_obstacles_for(self, index: int) -> List[AgentState]:
        other_obstacles = self.obstacles[:]
        for j, pursuer in enumerate(self.pursuers):
            if index != j:
                other_obstacles.append(pursuer)
        for j, target in enumerate(self.targets):
            if index != j:
                other_obstacles.append(target)
        return other_obstacles

    def advance_control_step(self):
        if self.step_idx >= self.steps:
            if self.loop:
                self.reset_experiment(spawn_static=False)
            elif self.shutdown_when_done:
                rospy.signal_shutdown("Gazebo TRUST-UP replay complete")
            return

        self.prev_targets = [clone_state(s) for s in self.targets]
        self.prev_pursuers = [clone_state(s) for s in self.pursuers]
        t = self.step_idx * self.dt
        self.targets = self.trajectory.integrate_all(self.targets, t, self.dt)
        for i, controller in enumerate(self.controllers):
            cmd, diag = controller.step(self.pursuers[i], self.targets[i], self.other_obstacles_for(i), self.dt)
            cmd_smoothed = smooth_command(cmd, self.filtered_commands[i], self.dt, self.limits, self.cfg)
            cmd_guarded, guard_diag = controller.guard_velocity_command(
                cmd_smoothed,
                self.pursuers[i],
                self.targets[i],
                [self.targets[i]] + self.other_obstacles_for(i),
            )
            self.filtered_commands[i] = cmd_guarded.copy()
            controller.command_velocity = cmd_guarded.copy()
            self.pursuers[i].velocity = cmd_guarded
            self.pursuers[i].position = self.pursuers[i].position + cmd_guarded * self.dt
            self.pursuers[i].stamp = t + self.dt
            self.metrics["qp_max_violation"] = max(
                self.metrics["qp_max_violation"],
                float(diag.get("qp_max_violation", 0.0)),
                float(guard_diag.get("velocity_guard_max_violation", 0.0)),
            )

        self.step_idx += 1
        self.sim_t = self.step_idx * self.dt
        self.next_targets = [clone_state(s) for s in self.targets]
        self.next_pursuers = [clone_state(s) for s in self.pursuers]
        self.update_metrics()
        if self.enable_trails and self.max_trail_points > 0 and self.step_idx % self.trail_stride == 0:
            self.spawn_trails(self.pursuers, self.targets)

    def update_metrics(self):
        if not self.targets:
            return
        radius_sum = 0.0
        if self.safety.use_object_radius:
            radius_sum = self.pursuers[0].radius + self.targets[0].radius
        paper_min = float(self.safety.collision_radius)
        paper_max = float(self.safety.sensing_radius)
        guard_min = paper_min + float(self.safety.tracking_inner_margin)
        guard_max = paper_max - float(self.safety.tracking_outer_margin)
        for pursuer, target in zip(self.pursuers, self.targets):
            clearance = float(np.linalg.norm(pursuer.position - target.position) - radius_sum)
            current_min = self.metrics["min_target_clearance"]
            current_max = self.metrics["max_target_clearance"]
            self.metrics["min_target_clearance"] = clearance if current_min is None else min(current_min, clearance)
            self.metrics["max_target_clearance"] = clearance if current_max is None else max(current_max, clearance)
            if clearance < paper_min - 1.0e-6 or clearance > paper_max + 1.0e-6:
                self.metrics["paper_violations"] += 1
            if clearance < guard_min - 1.0e-6 or clearance > guard_max + 1.0e-6:
                self.metrics["guard_violations"] += 1

        for i, pursuer in enumerate(self.pursuers):
            for j, other in enumerate(self.pursuers):
                if j <= i:
                    continue
                radius_sum = pursuer.radius + other.radius if self.safety.use_object_radius else 0.0
                clearance = float(np.linalg.norm(pursuer.position - other.position) - radius_sum)
                if clearance < paper_min - 1.0e-6:
                    self.metrics["pursuer_collision_violations"] += 1
            for obstacle in self.obstacles:
                radius_sum = pursuer.radius + obstacle.radius if self.safety.use_object_radius else 0.0
                clearance = float(np.linalg.norm(pursuer.position - obstacle.position) - radius_sum)
                if clearance < paper_min - 1.0e-6:
                    self.metrics["pursuer_collision_violations"] += 1

        if len(self.targets) >= 2:
            target_radius_sum = self.targets[0].radius + self.targets[1].radius if self.safety.use_object_radius else 0.0
            pair_clearance = float(np.linalg.norm(self.targets[0].position - self.targets[1].position) - target_radius_sum)
            current_pair = self.metrics["min_pair_clearance"]
            self.metrics["min_pair_clearance"] = pair_clearance if current_pair is None else min(current_pair, pair_clearance)
            pair_min = float(self.cfg.get("trajectory_smoothing", {}).get("target_pair_clearance", 0.0))
            if pair_clearance < pair_min - 1.0e-6:
                self.metrics["target_pair_violations"] += 1

    def spawn_trails(self, pursuers: Sequence[AgentState], targets: Sequence[AgentState]):
        entities = [("pursuer", idx, state) for idx, state in enumerate(pursuers)]
        entities.extend(("target", idx, state) for idx, state in enumerate(targets))
        for kind, idx, state in entities:
            key = "%s_%d" % (kind, idx)
            name = "gz_trail_%s_%05d" % (key, self.trail_counter)
            color = TRAIL_COLORS.get(key, "0.9 0.9 0.9 0.8")
            spawn_model(
                self.spawn_srv,
                self.delete_srv,
                name,
                sphere_sdf(name, 0.045, color, True, visual_only=True, transparency=0.0),
                state.position,
                self.existing_names,
            )
            self.trail_names[(kind, idx)].append(name)
            while len(self.trail_names[(kind, idx)]) > self.max_trail_points:
                old_name = self.trail_names[(kind, idx)].popleft()
                try:
                    self.delete_srv(old_name)
                    self.existing_names.discard(old_name)
                except Exception:
                    pass
        self.trail_counter += 1

    def render_states(self, now: float) -> Tuple[List[AgentState], List[AgentState]]:
        alpha = (now - self.last_step_time) / max(self.dt, 1.0e-6)
        pursuers = [
            interpolate_state(prev, nxt, alpha, self.dt)
            for prev, nxt in zip(self.prev_pursuers, self.next_pursuers)
        ]
        targets = [
            interpolate_state(prev, nxt, alpha, self.dt)
            for prev, nxt in zip(self.prev_targets, self.next_targets)
        ]
        return pursuers, targets

    def publish_status(self):
        status = dict(self.metrics)
        status.update(
            {
                "scenario": self.scenario,
                "step": self.step_idx,
                "sim_t": self.sim_t,
                "dt": self.dt,
                "steps": self.steps,
                "loop": self.loop,
                "paper_clearance_range": [float(self.safety.collision_radius), float(self.safety.sensing_radius)],
                "guard_clearance_range": [
                    float(self.safety.collision_radius + self.safety.tracking_inner_margin),
                    float(self.safety.sensing_radius - self.safety.tracking_outer_margin),
                ],
                "target_pair_clearance_required": float(
                    self.cfg.get("trajectory_smoothing", {}).get("target_pair_clearance", 0.0)
                ),
            }
        )
        self.status_pub.publish(String(json.dumps(status, sort_keys=True)))

    def spin(self):
        rate = rospy.Rate(self.visual_rate_hz)
        self.last_step_time = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            while now - self.last_step_time >= self.dt and not rospy.is_shutdown():
                self.advance_control_step()
                self.last_step_time += self.dt
                if self.step_idx >= self.steps and not self.loop:
                    break
            if rospy.is_shutdown():
                break
            render_pursuers, render_targets = self.render_states(now)
            self.move_all(render_pursuers, render_targets)
            self.publish_odometry(render_pursuers, render_targets)
            self.publish_status()
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break


def main() -> int:
    rospy.init_node("gazebo_paper_replay_visualizer")
    replay = GazeboPaperReplay()
    try:
        replay.spin()
    except rospy.ROSInterruptException:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
