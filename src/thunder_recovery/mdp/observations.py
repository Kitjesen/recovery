# Copyright (c) 2026 Qiongpei Technology
# SPDX-License-Identifier: Apache-2.0

"""Privileged observation functions for the critic (paper Fig. 3
asymmetric actor-critic).

The actor uses onboard-realistic, noise-injected observations inherited
from `ThunderHistRoughEnvCfg`. The critic gets the same stack plus the
five signals below — all instantaneous (history_length=0), all
sim-ground-truth, all invisible to the robot's on-robot sensors.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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

    The policy's `base_lin_vel` has Unoise injected to emulate IMU drift;
    the critic gets the clean sim value here.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def priv_base_ang_vel_clean(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ground-truth body-frame angular velocity, shape (N, 3)."""
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
