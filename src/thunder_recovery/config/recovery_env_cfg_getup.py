# Copyright (c) 2026 Qiongpei Technology
# SPDX-License-Identifier: Apache-2.0

"""Thunder fall-recovery env — Method 2 (mujoco_playground getup style).

Reference:
  github.com/google-deepmind/mujoco_playground
  → mujoco_playground/_src/locomotion/go1/getup.py

Overview
--------
A second implementation of the fall-recovery task, following the Google
DeepMind mujoco_playground Go1 `getup` environment rather than the
Deng et al. (2506.05516) recipe used by `ThunderRecoveryEnvCfg`
(Method 1).

What changes vs. Method 1
-------------------------

Scene / actions / observations / termination / PPO cfg: unchanged —
reused from Method 1 so the two methods are directly comparable on a
tensorboard side-by-side.

Events (reset):
  Method 1 uses `reset_with_freefall` (always-drop from 1.1 m) +
  `zero_action_freefall` (2 s gains=0 window).
  Method 2 uses `reset_getup` with a 60/40 Bernoulli branch:
    * 60% — drop from 0.5 m with random SO(3) quat and random joint
      angles clamped to soft limits;
    * 40% — nominal default pose.
  No free-fall torque override: the policy acts from t=0 with nominal PD
  gains. Curriculum comes purely from the reset distribution.

Rewards:
  Method 1 wires 13 terms with ED(t)=(t/T)^3 and CW=β·decay^i.
  Method 2 wires 9 plain-weight terms (no time-varying shaping):
    * orientation  — exp(-2·‖[0,0,-1]−g_body‖²)
    * torso_height — exp(min(h, z_des)) − 1
    * posture      — gate · exp(-0.5·‖q − q_default‖²)       [gated]
    * stand_still  — gate · exp(-0.5·‖a‖²)                   [gated]
    * action_rate  — 1st + 2nd order action differences      [cost]
    * torques      — √(Στ²) + Σ|τ|                           [cost]
    * dof_pos_limits — soft limit violation                  [cost]
    * dof_acc      — Σq̈²                                      [cost]
    * dof_vel      — Σ clamp(|q̇|−2π, 0)²                     [cost]
  where `gate = (upright) ∧ (at_desired_height)`. Scale signs are set
  on the RewTerm weight (positive = reward, negative = penalty).

Success metric (`recovery_success_rate`, last-second paper criteria)
is still wired so the two methods report the same KPI.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp_core

from thunder_recovery import mdp as recovery_mdp
from thunder_recovery.config.recovery_env_cfg import (
    BASE_LINK_NAME,
    ThunderRecoveryEnvCfg,
)


# -- Events -------------------------------------------------------------
@configclass
class GetupEventsCfg:
    """Startup DR + mujoco_playground-style 60/40 reset."""

    # Startup: friction randomisation (kept from Method 1).
    randomize_friction = EventTerm(
        func=mdp_core.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    # Reset: mujoco_playground-style 60/40 drop.
    reset_getup = EventTerm(
        func=recovery_mdp.reset_getup,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "drop_prob": 0.6,
            "drop_height": 0.5,
            "root_vel_range": 0.5,
        },
    )

    # Reset: base mass DR +-10 % (kept from Method 1).
    randomize_mass = EventTerm(
        func=mdp_core.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    # Reset: external force/torque perturbation (kept from Method 1).
    apply_external_force = EventTerm(
        func=mdp_core.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "force_range": (-10.0, 10.0),
            "torque_range": (-10.0, 10.0),
        },
    )


# -- Rewards (mujoco_playground Go1 getup) ------------------------------
@configclass
class GetupRewardsCfg:
    """9 reward terms from mujoco_playground Go1 getup.

    Weights mirror the mujoco_playground defaults for the task terms
    and are kept conservative on the behaviour costs. All terms are
    plain constants — no ED/CW time-varying shaping.
    """

    # Task (always on)
    getup_orientation = RewTerm(
        func=recovery_mdp.getup_orientation,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    getup_torso_height = RewTerm(
        func=recovery_mdp.getup_torso_height,
        weight=1.0,
        params={"z_des": 0.426, "asset_cfg": SceneEntityCfg("robot")},
    )

    # Task (gated on upright ∧ at-height)
    getup_posture = RewTerm(
        func=recovery_mdp.getup_posture,
        weight=1.0,
        params={
            "z_des": 0.426,
            "ori_tol": 0.01,
            "pos_tol": 0.005,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    getup_stand_still = RewTerm(
        func=recovery_mdp.getup_stand_still,
        weight=1.0,
        params={
            "z_des": 0.426,
            "ori_tol": 0.01,
            "pos_tol": 0.005,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Behaviour costs (negative weights)
    getup_action_rate = RewTerm(
        func=recovery_mdp.getup_action_rate,
        weight=-0.001,
    )
    getup_torques = RewTerm(
        func=recovery_mdp.getup_torques,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    getup_dof_pos_limits = RewTerm(
        func=recovery_mdp.getup_dof_pos_limits,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    getup_dof_acc = RewTerm(
        func=recovery_mdp.getup_dof_acc,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    getup_dof_vel = RewTerm(
        func=recovery_mdp.getup_dof_vel,
        weight=-0.1,
        params={"max_velocity": 6.2832, "asset_cfg": SceneEntityCfg("robot")},
    )

    # Logging only: paper success indicator, last 1 s of the episode.
    # Kept so tensorboard reports the same KPI as Method 1.
    recovery_success_rate = RewTerm(
        func=recovery_mdp.recovery_success_rate,
        weight=1e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # Side-effect term that advances the step counter consumed by
    # `recovery_success_rate` (last-second gate).
    recovery_step_counter = RewTerm(
        func=recovery_mdp.recovery_step_counter,
        weight=1e-6,
    )


# -- Terminations -------------------------------------------------------
@configclass
class GetupTerminationsCfg:
    """Timeout only — mujoco_playground's energy termination is disabled
    by default (threshold=inf), so we skip it entirely."""

    time_out = DoneTerm(func=mdp_core.time_out, time_out=True)


# -- Main env cfg -------------------------------------------------------
@configclass
class ThunderRecoveryGetupEnvCfg(ThunderRecoveryEnvCfg):
    """Method 2 env cfg — inherits Scene / Actions / Observations from
    Method 1 and swaps in the mujoco_playground-style events, rewards,
    and terminations."""

    events: GetupEventsCfg = GetupEventsCfg()
    rewards: GetupRewardsCfg = GetupRewardsCfg()
    terminations: GetupTerminationsCfg = GetupTerminationsCfg()
