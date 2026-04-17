# Copyright (c) 2026 Qiongpei Technology
# SPDX-License-Identifier: Apache-2.0

"""Recovery MDP package.

Layout (method-split):

  mdp/
    _utils.py              — shared infra (ED, CW, step counter, joint split, dt)
    observations.py        — shared privileged critic observations
    method1_deng/
      events.py            — Deng et al. free-fall reset + torque override
      rewards.py           — Deng et al. 13 ED/CW-shaped rewards
    method2_getup/
      events.py            — mujoco_playground 60/40 drop reset
      rewards.py           — mujoco_playground 9 gated-exp rewards

Public names are re-exported flat here so env cfgs import them via a
single `from thunder_recovery import mdp as recovery_mdp` — they do not
need to know which subpackage a reward/event actually lives in.
"""

from ._utils import (
    FREEFALL_STEPS,
    RECOVERY_STEPS_PER_ITER,
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

# ── Method 1 — Deng et al. (arXiv:2506.05516) ────────────────────────────
from .method1_deng.events import (
    reset_with_freefall,
    zero_action_freefall,
)
from .method1_deng.rewards import (
    check_recovery_success,
    recovery_action_rate_legs,
    recovery_base_height,
    recovery_base_orientation,
    recovery_body_collision,
    recovery_joint_acceleration,
    recovery_joint_deviation,
    recovery_joint_velocity,
    recovery_stand_joint_pos,
    recovery_step_counter,
    recovery_success_rate,
    recovery_support_state,
    recovery_torques,
    recovery_wheel_leg_coord,
    recovery_wheel_velocity,
)

# ── Method 2 — mujoco_playground getup ───────────────────────────────────
from .method2_getup.events import (
    reset_getup,
)
from .method2_getup.rewards import (
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
    # Shared observations
    "joint_pos_legs",
    "joint_vel_legs",
    "previous_joint_pos_legs",
    "previous_joint_vel_legs",
    "previous_wheel_vel",
    "wheel_vel",
    "priv_base_ang_vel_clean",
    "priv_base_height",
    "priv_base_lin_vel_clean",
    "priv_body_contact_force",
    "priv_foot_contact",
    # Method 1 — events
    "reset_with_freefall",
    "zero_action_freefall",
    # Method 1 — rewards
    "check_recovery_success",
    "recovery_action_rate_legs",
    "recovery_base_height",
    "recovery_base_orientation",
    "recovery_body_collision",
    "recovery_joint_acceleration",
    "recovery_joint_deviation",
    "recovery_joint_velocity",
    "recovery_stand_joint_pos",
    "recovery_step_counter",
    "recovery_success_rate",
    "recovery_support_state",
    "recovery_torques",
    "recovery_wheel_leg_coord",
    "recovery_wheel_velocity",
    # Method 2 — events
    "reset_getup",
    # Method 2 — rewards
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
