# Copyright (c) 2026 Qiongpei Technology
# SPDX-License-Identifier: Apache-2.0

"""Recovery MDP package.

Public re-exports split by Isaac Lab convention:
  events.py        — reset / free-fall torque override
  observations.py  — privileged critic-only observations
  rewards.py       — reward functions + paper success metric

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

from .observations import (
    priv_base_ang_vel_clean,
    priv_base_height,
    priv_base_lin_vel_clean,
    priv_body_contact_force,
    priv_foot_contact,
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
    recovery_wheel_velocity,
)

__all__ = [
    # Constants
    "FREEFALL_STEPS",
    "RECOVERY_STEPS_PER_ITER",
    # Events
    "reset_with_freefall",
    "zero_action_freefall",
    # Privileged observations
    "priv_base_ang_vel_clean",
    "priv_base_height",
    "priv_base_lin_vel_clean",
    "priv_body_contact_force",
    "priv_foot_contact",
    # Rewards
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
    "recovery_wheel_velocity",
]
