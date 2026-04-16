# Copyright (c) 2026 Qiongpei Technology
# SPDX-License-Identifier: Apache-2.0

"""Thunder fall recovery environment — fully self-contained.

Upload to server, register task, train. No runner changes needed.

Based on 'Learning to Recover' (arXiv:2506.05516):
- Random fallen init (1.1m drop, random orientation)
- 5s fixed episode, no early termination
- ED reward shaping built into reward functions
- 9 reward terms from paper Table I
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

# Import recovery reward functions
import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.thunder_hist.rough_env_cfg import (
    ThunderHistRoughEnvCfg,
)


@configclass
class RecoveryRewardsCfg:
    """Recovery reward terms — all from paper Table I.

    ED shaping is built into each task reward function.
    Just set weights here, env handles the rest.
    """

    # Step counter (tiny weight — must be called every step for ED counter)
    recovery_step_counter = RewTerm(
        func=mdp.recovery_step_counter,
        weight=1e-10,
    )

    # ── Task rewards (×ED, paper Table I) ──
    recovery_stand_joint_pos = RewTerm(
        func=mdp.recovery_stand_joint_pos,
        weight=42.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "sigma": 0.5},
    )
    recovery_base_height = RewTerm(
        func=mdp.recovery_base_height,
        weight=120.0,
        params={"target_height": 0.426, "sigma": 0.1, "asset_cfg": SceneEntityCfg("robot")},
    )
    recovery_base_orientation = RewTerm(
        func=mdp.recovery_base_orientation,
        weight=50.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # ── Behavior rewards (×CW, paper Table I) ──
    recovery_body_collision = RewTerm(
        func=mdp.recovery_body_collision,
        weight=-5e-2,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", ".*thigh.*", ".*calf.*"])},
    )
    recovery_action_rate_legs = RewTerm(
        func=mdp.recovery_action_rate_legs,
        weight=-1e-2,
    )

    # ── Support state (paper Section E, auxiliary shaping) ──
    recovery_support_state = RewTerm(
        func=mdp.recovery_support_state,
        weight=5.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*"), "threshold": 1.0},
    )

    # ── Wheel-leg coordination (paper core contribution, early-phase only) ──
    recovery_wheel_leg_coord = RewTerm(
        func=mdp.recovery_wheel_leg_coord,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "max_wheel_speed": 40.0},
    )

    # ── Constant penalties (paper Table I) ──
    recovery_joint_velocity = RewTerm(
        func=mdp.recovery_joint_velocity,
        weight=-2e-2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    recovery_torques = RewTerm(
        func=mdp.recovery_torques,
        weight=-2.5e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    recovery_joint_acceleration = RewTerm(
        func=mdp.recovery_joint_acceleration,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    recovery_wheel_velocity = RewTerm(
        func=mdp.recovery_wheel_velocity,
        weight=-2e-2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class ThunderRecoveryEnvCfg(ThunderHistRoughEnvCfg):
    """Thunder fall recovery environment.

    Inherits robot/scene from locomotion env, replaces everything else:
    - Flat terrain (Phase 0)
    - 5s episode, no early termination
    - Random fallen initialization (1.1m drop)
    - Recovery-specific rewards with ED shaping
    - No velocity commands
    - No DR (Phase 0 baseline)
    """

    def __post_init__(self):
        super().__post_init__()

        # ── Flat terrain ──
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None

        # ── Episode: 5s at 50Hz ──
        self.episode_length_s = 5.0

        # ── Initialization: random fallen state ──
        # Override reset event with free-fall initialization
        self.events.randomize_reset_base = EventTerm(
            func=mdp.reset_with_freefall,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "drop_height": 1.1,
            },
        )

        # ── Replace ALL rewards with recovery rewards ──
        self.rewards = RecoveryRewardsCfg()

        # ── Disable ALL terminations except timeout ──
        self.terminations.illegal_contact = None

        # ── Disable ALL curriculum ──
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels = None
        self.curriculum.disturbance_levels = None
        self.curriculum.mass_randomization_levels = None
        self.curriculum.com_randomization_levels = None

        # ── Zero velocity commands (recovery doesn't need them) ──
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # ── Free-fall: zero joint commands for first 2s (paper Section III-A) ──
        self.events.freefall_zero_action = EventTerm(
            func=mdp.zero_action_freefall,
            mode="interval",
            interval_range_s=(0.02, 0.02),  # every control step (50Hz)
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # ── DR enabled (paper Section IV-A) ──
        # friction: base class default (0.3, 1.0) static, (0.3, 0.8) dynamic
        # mass: disabled (class-based API incompatible)
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_mass_others = None
        # COM: base class default +/-5cm
        # external_force: re-create (parent set to None)
        from isaaclab.envs.mdp import apply_external_force_torque
        self.events.randomize_apply_external_force_torque = EventTerm(
            func=apply_external_force_torque,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "force_range": (-10.0, 10.0),
                "torque_range": (-10.0, 10.0),
            },
        )
        # actuator_gains: disabled (class-based API incompatible)
        self.events.randomize_actuator_gains = None
        # push_robot: base class default (push every 10-15s)

        # ── Wheel vel_scale < 1.0 for recovery (paper Section C) ──
        self.actions.joint_vel.scale = 0.8

        # ── Single frame observations (matching paper Fig.3) ──
        self.observations.policy.history_length = 1
        self.observations.critic.history_length = 1

        # ── Observations: asymmetric actor-critic (paper Fig.3) ──
        from isaaclab.managers import ObservationTermCfg as ObsTerm
        from isaaclab.utils.noise import UniformNoiseCfg as Unoise
        import robot_lab.tasks.manager_based.locomotion.velocity.mdp as obs_mdp

        # Actor: onboard-realistic, noisy IMU + encoders only
        # Remove velocity commands from actor (recovery has no cmd)
        self.observations.policy.velocity_commands = None
        # Noisy body linear velocity (IMU approximation)
        self.observations.policy.base_lin_vel = ObsTerm(
            func=obs_mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # Critic: everything the actor has + privileged sim-only signals
        self.observations.critic.velocity_commands = None

        # (1) Clean base linear/angular velocity (no IMU noise)
        self.observations.critic.priv_base_lin_vel = ObsTerm(
            func=mdp.priv_base_lin_vel_clean,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        self.observations.critic.priv_base_ang_vel = ObsTerm(
            func=mdp.priv_base_ang_vel_clean,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # (2) Base z-height — not observable from onboard sensors
        self.observations.critic.priv_base_height = ObsTerm(
            func=mdp.priv_base_height,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-5.0, 5.0),
            scale=1.0,
        )
        # (3) Foot contact binary state (support state signal)
        self.observations.critic.priv_foot_contact = ObsTerm(
            func=mdp.priv_foot_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*"),
                    "threshold": 1.0},
            clip=(0.0, 1.0),
            scale=1.0,
        )
        # (4) Body (shank/thigh/base) contact magnitude — collision state
        self.observations.critic.priv_body_contact_force = ObsTerm(
            func=mdp.priv_body_contact_force,
            params={"sensor_cfg": SceneEntityCfg("contact_forces",
                    body_names=["base_link", ".*thigh.*", ".*calf.*"])},
            clip=(0.0, 500.0),
            scale=0.01,
        )

        # ── Remove height scan observations ──
        if hasattr(self.observations, 'height_scan_group'):
            self.observations.height_scan_group = None
