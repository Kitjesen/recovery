# Copyright (c) 2026 Qiongpei Technology
# SPDX-License-Identifier: Apache-2.0

"""Recovery MDP package.

Public re-exports split by Isaac Lab convention:
  events.py         — Method 1 reset / free-fall torque override
  events_getup.py   — Method 2 reset (mujoco_playground 60/40 drop)
  observations.py   — privileged critic-only observations
  rewards.py        — Method 1 rewards (Deng et al., ED/CW) + success metric
  rewards_getup.py  — Method 2 rewards (mujoco_playground gated exp)

Private helpers (ED, CW, step counter, joint split, dt) live in _utils.py
and are re-exported only for advanced users.
"""

from ._utils import (
    FREEFALL_STEPS,
    RECOVERY_STEPS_PER_ITER,
)

from .events import (
    reset_with_freefall,
    zero_action_freefall,
)

from .events_getup import (
    reset_getup,
)

from .observations import (
    joint_pos_legs,
    joint_vel_legs,
    previous_joint_pos_legs,
    previous_joint_vel_legs,
    previous_wheel_vel,
    priv_base_ang_vel_clean,
    priv_base_height,
    priv_base_lin_vel_clean,
    priv_body_contact_force,
    priv_foot_contact,
    wheel_vel,
)

from .rewards import (
    check_recovery_success,
    recovery_action_rate_legs,
    recovery_base_height,
    recovery_base_orientation,
    recovery_body_collision,
    recovery_joint_acceleration,
    recovery_joint_velocity,
    recovery_stand_joint_pos,
    recovery_step_counter,
    recovery_success_rate,
    recovery_support_state,
    recovery_torques,
    recovery_wheel_leg_coord,
    recovery_joint_deviation,
    recovery_wheel_velocity,
)

from .rewards_getup import (
    getup_action_rate,
    getup_dof_acc,
    getup_dof_pos_limits,
    getup_dof_vel,
    getup_orientation,
    getup_posture,
    getup_stand_still,
    getup_torques,
    getup_torso_height,
)

__all__ = [
    # Constants
    "FREEFALL_STEPS",
    "RECOVERY_STEPS_PER_ITER",
    # Events
    "reset_with_freefall",
    "zero_action_freefall",
    "reset_getup",
    # Actor observations (paper 78-dim spec)
    "joint_pos_legs",
    "joint_vel_legs",
    "previous_joint_pos_legs",
    "previous_joint_vel_legs",
    "previous_wheel_vel",
    "wheel_vel",
    # Privileged observations
    "priv_base_ang_vel_clean",
    "priv_base_height",
    "priv_base_lin_vel_clean",
    "priv_body_contact_force",
    "priv_foot_contact",
    # Method 1 rewards (Deng et al., ED/CW shaping)
    "check_recovery_success",
    "recovery_action_rate_legs",
    "recovery_base_height",
    "recovery_base_orientation",
    "recovery_body_collision",
    "recovery_joint_acceleration",
    "recovery_joint_velocity",
    "recovery_stand_joint_pos",
    "recovery_step_counter",
    "recovery_success_rate",
    "recovery_support_state",
    "recovery_torques",
    "recovery_wheel_leg_coord",
    "recovery_joint_deviation",
    "recovery_wheel_velocity",
    # Method 2 rewards (mujoco_playground getup)
    "getup_action_rate",
    "getup_dof_acc",
    "getup_dof_pos_limits",
    "getup_dof_vel",
    "getup_orientation",
    "getup_posture",
    "getup_stand_still",
    "getup_torques",
    "getup_torso_height",
]
