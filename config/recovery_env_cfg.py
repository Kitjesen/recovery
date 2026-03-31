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

    # Step counter (weight=0, just keeps ED counter in sync)
    recovery_step_counter = RewTerm(
        func=mdp.recovery_step_counter,
        weight=0.0,
    )

    # ── Task rewards (ED built-in) ──
    recovery_stand_joint_pos = RewTerm(
        func=mdp.recovery_stand_joint_pos,
        weight=42.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "sigma": 0.5},
    )
    recovery_base_height = RewTerm(
        func=mdp.recovery_base_height,
        weight=120.0,
        params={"target_height": 0.55, "sigma": 0.1, "asset_cfg": SceneEntityCfg("robot")},
    )
    recovery_base_orientation = RewTerm(
        func=mdp.recovery_base_orientation,
        weight=-50.0,  # negative: orientation error is a penalty
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # ── Behavior rewards (CW built-in) ──
    recovery_body_collision = RewTerm(
        func=mdp.recovery_body_collision,
        weight=-5e-2,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*")},
    )
    recovery_action_rate_legs = RewTerm(
        func=mdp.recovery_action_rate_legs,
        weight=-1e-2,
    )

    # ── Constant penalties ──
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

        # ── Disable ALL DR (Phase 0) ──
        self.events.randomize_rigid_body_material = None
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_mass_others = None
        self.events.randomize_com_positions = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_push_robot = None

        # ── Remove height scan observations ──
        if hasattr(self.observations, 'height_scan_group'):
            self.observations.height_scan_group = None
