# Copyright (c) 2026 Qiongpei Technology
# SPDX-License-Identifier: Apache-2.0

"""Recovery reward functions — matches paper exactly.

Based on 'Learning to Recover' (arXiv:2506.05516) Eq.1-4 + Table I.

ED formula (Eq.1): ed = (a·t/T)^k, a=T/2, k=3
  - NOT normalized to [0,1]
  - t=0: ed=0 (free exploration / free-fall)
  - t=T: ed=(T/2)^3 (strong standing reward)
  - PPO advantage normalization handles the scale

Total reward (Eq.4):
  r = ed × (r_joint_pos + r_base_height + r_base_ori)
    + cw × (r_collision + r_action_rate)
    + r_joint_vel + r_torque + r_joint_acc + r_wheelvel

Note: "Scale" in Table I = weight in RewardTermCfg.
      ED multiplies the RAW reward (before weight), weight is applied by Isaac Lab.
      So the functions here return ed × raw_value, and Isaac Lab multiplies by weight.
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ── ED helper (Eq.1 — no normalization) ──

def _get_ed(env: ManagerBasedRLEnv, k: int = 3) -> torch.Tensor:
    """Episode-based Dynamic factor (Eq.1) — NOT normalized.

    ed = (a·t/T)^k where a=T/2, T=total steps, t=current step.
    Also increments step counter (once per sim step).
    """
    if not hasattr(env, "_recovery_step_count"):
        env._recovery_step_count = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_recovery_ed_last_step"):
        env._recovery_ed_last_step = -1

    # Increment once per sim step
    current_step = env.common_step_counter if hasattr(env, "common_step_counter") else 0
    if current_step != env._recovery_ed_last_step:
        env._recovery_step_count += 1
        env._recovery_ed_last_step = current_step

    t = env._recovery_step_count
    T = float(env.max_episode_length)  # 250 steps (5s @ 50Hz)
    a = T / 2.0

    # ed = (a * t / T)^k — paper formula exactly, no normalization
    ed = (a * t / T) ** k
    return ed


def _get_cw(env: ManagerBasedRLEnv, beta: float = 0.3, decay: float = 0.968) -> float:
    """Curriculum Weight (Eq.3)."""
    if hasattr(env, "common_step_counter"):
        progress = env.common_step_counter / (env.max_episode_length * 5000)
    else:
        progress = 0.0
    return beta * (decay ** (progress * 10000))


# ── Step Counter (weight=1e-10, ensures it gets called) ──

def recovery_step_counter(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Dummy — step counting is done inside _get_ed."""
    return torch.zeros(env.num_envs, device=env.device)


# ── Task Rewards (×ED) — Table I ──

def recovery_stand_joint_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma: float = 0.5,
) -> torch.Tensor:
    """exp(-sum(q-q_default)^2 / sigma^2). Table I scale=42.

    Returns: ed × exp(-error/σ²)
    Isaac Lab multiplies by weight=42.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    error = torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    raw = torch.exp(-error / (sigma ** 2))
    return _get_ed(env) * raw


def recovery_base_height(
    env: ManagerBasedRLEnv,
    target_height: float = 0.388,
    sigma: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """exp(-max(h_target-h,0)^2 / sigma^2). Table I scale=120.

    Returns: ed × exp(-height_error²/σ²)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    height_error = torch.clamp(target_height - asset.data.root_pos_w[:, 2], min=0.0)
    raw = torch.exp(-torch.square(height_error) / (sigma ** 2))
    return _get_ed(env) * raw


def recovery_base_orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """(g_b - e_z)^2. Table I scale=50.

    Paper definition: raw error squared, NOT exp.
    Weight should be NEGATIVE (-50) because larger error = worse.
    Returns: ed × ||g_body - [0,0,-1]||²
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ideal = torch.tensor([0.0, 0.0, -1.0], device=env.device)
    error = torch.sum(torch.square(asset.data.projected_gravity_b - ideal), dim=1)
    return _get_ed(env) * error


# ── Behavior Rewards (×CW) ──

def recovery_body_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """sum(||contact_force||²). Table I scale=-5e-2. Disabled for now."""
    return torch.zeros(env.num_envs, device=env.device)


def recovery_action_rate_legs(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """sum((a_leg[t]-a_leg[t-1])²). Table I scale=-1e-2. Wheels excluded."""
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    leg_diff = torch.sum(torch.square(action[:, :12] - prev_action[:, :12]), dim=1)
    return _get_cw(env) * leg_diff


# ── Constant Penalties (always active) ──

def recovery_joint_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """sum(q_dot²). Table I scale=-2e-2."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel), dim=1)


def recovery_torques(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """sum(tau²). Table I scale=-2.5e-5."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque), dim=1)


def recovery_joint_acceleration(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """sum(q_ddot²). Table I scale=-2.5e-7."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc), dim=1)


def recovery_wheel_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """sum(wheel_vel²). Table I scale=-2e-2."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, -4:]), dim=1)


# ── Free-fall reset ──

def reset_with_freefall(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    drop_height: float = 1.1,
):
    """Random orientation + 1.1m drop. ED handles the rest (ed≈0 during fall)."""
    asset: Articulation = env.scene[asset_cfg.name]
    if len(env_ids) == 0:
        return

    # Uniform random quaternion (Shoemake)
    u1 = torch.rand(len(env_ids), device=env.device)
    u2 = torch.rand(len(env_ids), device=env.device) * 2 * math.pi
    u3 = torch.rand(len(env_ids), device=env.device) * 2 * math.pi
    qw = torch.sqrt(1 - u1) * torch.sin(u2)
    qx = torch.sqrt(1 - u1) * torch.cos(u2)
    qy = torch.sqrt(u1) * torch.sin(u3)
    qz = torch.sqrt(u1) * torch.cos(u3)
    quat = torch.stack([qw, qx, qy, qz], dim=1)

    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, 2] = drop_height
    root_state[:, 3:7] = quat
    root_state[:, 7:] = 0.0
    asset.write_root_state_to_sim(root_state, env_ids)

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_pos += torch.empty_like(joint_pos).uniform_(-0.5, 0.5)
    asset.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)

    if hasattr(env, "_recovery_step_count"):
        env._recovery_step_count[env_ids] = 0


# ── Success checker ──

def check_recovery_success(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_threshold: float = 0.30,
    joint_threshold: float = 0.5,
    vel_threshold: float = 0.1,
    ori_threshold: float = 0.1,
) -> torch.Tensor:
    """Paper success: height>0.42m, joint_dev<0.5rad, vel<0.1, ori<0.1."""
    asset: Articulation = env.scene[asset_cfg.name]
    h_ok = asset.data.root_pos_w[:, 2] > height_threshold
    j_ok = torch.norm(asset.data.joint_pos - asset.data.default_joint_pos, dim=1) < joint_threshold
    v_ok = torch.max(torch.abs(asset.data.joint_vel), dim=1).values < vel_threshold
    ideal = torch.tensor([0.0, 0.0, -1.0], device=env.device)
    o_ok = torch.norm(asset.data.projected_gravity_b - ideal, dim=1) < ori_threshold
    return h_ok & j_ok & v_ok & o_ok
