# Copyright (c) 2026 Qiongpei Technology
# SPDX-License-Identifier: Apache-2.0

"""Observation functions for the recovery MDP.

Two groups:

- Actor (`joint_pos_legs`, `joint_vel_legs`, `wheel_vel`, `previous_*`)
  — deployment-realistic observations matching the 78-dim spec inferred
  from the author's public checkpoint-loading code
  (`boyuandeng/Recovery_go2w/simulate_python/test/runtest.py`):

        pre_actions              16
        projected_gravity         3
        ang_vel                   3   (no linear velocity!)
        joint_pos_legs           12
        joint_vel_legs           12
        wheel_vel                 4
        previous_joint_pos_legs  12
        previous_joint_vel_legs  12
        previous_wheel_vel        4
                                ──
                                78

  The `previous_*` terms cache the t-1 values in per-env buffers attached
  to the env — the caches are reset in `reset_with_freefall` so the first
  post-reset step observes the fresh fallen pose, not the last frame of
  the previous episode.

- Critic (`priv_*`) — the actor's 78-dim obs plus 5 privileged signals,
  all instantaneous (history_length=0) and sim-ground-truth.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from ._utils import _get_joint_split

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ── Actor (paper 78-dim spec) ──


def joint_pos_legs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Leg joint positions, shape (N, 12). Wheels excluded — paper's
    `dof_pos` is legs-only."""
    asset: Articulation = env.scene[asset_cfg.name]
    leg_ids, _ = _get_joint_split(env, asset)
    return asset.data.joint_pos[:, leg_ids]


def joint_vel_legs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Leg joint velocities, shape (N, 12)."""
    asset: Articulation = env.scene[asset_cfg.name]
    leg_ids, _ = _get_joint_split(env, asset)
    return asset.data.joint_vel[:, leg_ids]


def wheel_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Wheel velocities, shape (N, 4). Paper separates wheel speeds from
    leg dof_vel; no wheel position is observed."""
    asset: Articulation = env.scene[asset_cfg.name]
    _, wheel_ids = _get_joint_split(env, asset)
    return asset.data.joint_vel[:, wheel_ids]


def _cache_prev(env: ManagerBasedRLEnv, key: str, current: torch.Tensor) -> torch.Tensor:
    """Return the cached `_recovery_prev_<key>` value, then overwrite it
    with `current` for the next step. On the first call we initialise the
    cache to `current` so the first-step previous equals current (a noop
    signal — nothing more we can do without a past)."""
    attr = f"_recovery_prev_{key}"
    if not hasattr(env, attr):
        setattr(env, attr, current.clone())
    prev = getattr(env, attr).clone()
    setattr(env, attr, current.clone())
    return prev


def previous_joint_pos_legs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Paper's `previous_joint_dof` (legs) — t-1 leg positions (N, 12)."""
    return _cache_prev(env, "joint_pos_legs", joint_pos_legs(env, asset_cfg))


def previous_joint_vel_legs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Paper's `previous_joint_vel` (legs) — t-1 leg velocities (N, 12)."""
    return _cache_prev(env, "joint_vel_legs", joint_vel_legs(env, asset_cfg))


def previous_wheel_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Paper's `previous_wheel_vel` — t-1 wheel speeds (N, 4)."""
    return _cache_prev(env, "wheel_vel", wheel_vel(env, asset_cfg))


# ── Critic (privileged) ──


def priv_base_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Base z-position `root_pos_w[:, 2]`, shape (N, 1).

    Actor cannot recover absolute base height from joint encoders — the
    critic uses it directly to evaluate recovery progress.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2:3]


def priv_base_lin_vel_clean(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ground-truth body-frame linear velocity, shape (N, 3).

    The actor has no `base_lin_vel` at all (IMU integration drift is too
    severe for recovery); the critic gets the clean sim value.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def priv_base_ang_vel_clean(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ground-truth body-frame angular velocity, shape (N, 3).

    Actor's `base_ang_vel` is noise-injected; critic sees the clean value.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


def priv_foot_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Per-foot binary contact state, shape (N, num_feet).

    The strongest support-state signal for the critic. Real robots do not
    have a clean contact sensor at each foot; the critic exploits the
    perfect sim signal.
    """
    if sensor_cfg.body_ids is None or sensor_cfg.body_ids == slice(None):
        raise RuntimeError(
            "priv_foot_contact requires sensor_cfg.body_ids to be resolved "
            "from a body_names regex; no safe index fallback exists."
        )
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, 0, :, :]
    magnitude = torch.norm(forces, dim=-1)[:, sensor_cfg.body_ids]
    return (magnitude > threshold).float()


def priv_body_contact_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Per-body contact-force magnitude on base / thigh / calf, shape
    (N, num_bodies).

    Tells the critic when the robot is dragging limbs or hitting the
    ground with its body — signal not accessible to the actor.
    """
    if sensor_cfg.body_ids is None or sensor_cfg.body_ids == slice(None):
        raise RuntimeError(
            "priv_body_contact_force requires sensor_cfg.body_ids to be resolved "
            "from a body_names regex; no safe index fallback exists."
        )
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, 0, :, :]
    return torch.norm(forces, dim=-1)[:, sensor_cfg.body_ids]
