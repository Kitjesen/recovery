# Copyright (c) 2026 Qiongpei Technology
# SPDX-License-Identifier: Apache-2.0

"""Recovery reward functions — Isaac Lab RewardTermCfg compatible.

Based on 'Learning to Recover' (arXiv:2506.05516) Table I + Eq.1-4.

Episode structure (5s = 250 steps @ 50Hz):
  Steps 0-99:   FREE-FALL phase — all rewards return 0, robot collapses naturally
  Steps 100-249: RECOVERY phase — ED shaping active, policy learns to stand up

ED only applies to recovery phase (steps 100-249):
  t_recovery = step - 100, T_recovery = 150
  ed = (T_recovery/2 * t_recovery / T_recovery)^3
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ── Constants ──
FREEFALL_STEPS = 100   # 2s at 50Hz — robot collapses on ground
RECOVERY_STEPS = 150   # 3s at 50Hz — policy active
TOTAL_STEPS = 250      # 5s total


# ── Helpers ──

def _is_freefall(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns True mask for envs still in free-fall phase."""
    if not hasattr(env, "_recovery_step_count"):
        env._recovery_step_count = torch.zeros(env.num_envs, device=env.device)
    return env._recovery_step_count < FREEFALL_STEPS


def _get_ed(env: ManagerBasedRLEnv, k: int = 3) -> torch.Tensor:
    """Episode-based Dynamic factor — only active during recovery phase.

    Also increments step counter (no separate counter function needed).
    Steps 0-99: returns 0 (free-fall)
    Steps 100-249: ed normalized 0→1
    """
    if not hasattr(env, "_recovery_step_count"):
        env._recovery_step_count = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_recovery_ed_last_step"):
        env._recovery_ed_last_step = -1

    # Increment counter once per sim step (avoid double-counting from multiple reward calls)
    current_step = env.common_step_counter if hasattr(env, "common_step_counter") else 0
    if current_step != env._recovery_ed_last_step:
        env._recovery_step_count += 1
        env._recovery_ed_last_step = current_step

    t = env._recovery_step_count
    t_recovery = (t - FREEFALL_STEPS).clamp(min=0)  # 0 during free-fall
    T = float(RECOVERY_STEPS)
    a = T / 2.0
    # Normalize: at t_recovery=T, ed = (a*T/T)^k / (a)^k = 1.0
    ed = (a * t_recovery / T) ** k / (a ** k)
    # Zero during free-fall
    ed = ed * (~_is_freefall(env)).float()
    return ed


def _get_cw(env: ManagerBasedRLEnv, beta: float = 0.3, decay: float = 0.968) -> float:
    """Curriculum Weight (Eq.3). Decays over training."""
    if hasattr(env, "common_step_counter"):
        progress = env.common_step_counter / (TOTAL_STEPS * 5000)
    else:
        progress = 0.0
    return beta * (decay ** (progress * 10000))


# ── Step Counter (must be registered with weight=0) ──

def recovery_step_counter(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Increment step counter. Register with weight=0."""
    if not hasattr(env, "_recovery_step_count"):
        env._recovery_step_count = torch.zeros(env.num_envs, device=env.device)
    env._recovery_step_count += 1
    return torch.zeros(env.num_envs, device=env.device)


# ── Task Rewards (×ED, zero during free-fall) ──

def recovery_stand_joint_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma: float = 0.5,
) -> torch.Tensor:
    """Joints return to default angles. Paper scale=42."""
    if not hasattr(env, "_recovery_step_count"):
        return torch.zeros(env.num_envs, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    error = torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    reward = torch.exp(-error / (sigma ** 2))
    return _get_ed(env) * reward


def recovery_base_height(
    env: ManagerBasedRLEnv,
    target_height: float = 0.388,
    sigma: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Body reaches target height. Paper scale=120."""
    if not hasattr(env, "_recovery_step_count"):
        return torch.zeros(env.num_envs, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    height_error = torch.clamp(target_height - asset.data.root_pos_w[:, 2], min=0.0)
    reward = torch.exp(-torch.square(height_error) / (sigma ** 2))
    return _get_ed(env) * reward


def recovery_base_orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for upright orientation. Paper scale=50.

    reward = ED * exp(-||g_body - [0,0,-1]||^2)
    Upright → error=0 → reward=1. Fallen → error~4 → reward≈0.
    """
    if not hasattr(env, "_recovery_step_count"):
        return torch.zeros(env.num_envs, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    ideal = torch.tensor([0.0, 0.0, -1.0], device=env.device)
    error = torch.sum(torch.square(asset.data.projected_gravity_b - ideal), dim=1)
    reward = torch.exp(-error)
    return _get_ed(env) * reward


# ── Behavior Rewards (×CW, zero during free-fall) ──

def recovery_body_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Body collision penalty. Paper scale=-5e-2."""
    # Disabled until sensor shape verified
    return torch.zeros(env.num_envs, device=env.device)


def recovery_action_rate_legs(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Leg action change penalty (wheels excluded!). Paper scale=-1e-2."""
    if _is_freefall(env).all():
        return torch.zeros(env.num_envs, device=env.device)
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    leg_diff = torch.sum(torch.square(action[:, :12] - prev_action[:, :12]), dim=1)
    # Zero during free-fall
    leg_diff = leg_diff * (~_is_freefall(env)).float()
    return _get_cw(env) * leg_diff


# ── Constant Penalties (zero during free-fall) ──

def recovery_joint_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint velocity penalty. Paper scale=-2e-2."""
    if not hasattr(env, "_recovery_step_count"):
        return torch.zeros(env.num_envs, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = torch.sum(torch.square(asset.data.joint_vel), dim=1)
    # Zero during free-fall — robot is just falling, no penalty
    penalty = penalty * (~_is_freefall(env)).float()
    return penalty


def recovery_torques(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Torque penalty. Paper scale=-2.5e-5."""
    if not hasattr(env, "_recovery_step_count"):
        return torch.zeros(env.num_envs, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = torch.sum(torch.square(asset.data.applied_torque), dim=1)
    penalty = penalty * (~_is_freefall(env)).float()
    return penalty


def recovery_joint_acceleration(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint acceleration penalty. Paper scale=-2.5e-7."""
    if not hasattr(env, "_recovery_step_count"):
        return torch.zeros(env.num_envs, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = torch.sum(torch.square(asset.data.joint_acc), dim=1)
    penalty = penalty * (~_is_freefall(env)).float()
    return penalty


def recovery_wheel_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Wheel spin penalty. Paper scale=-2e-2."""
    if not hasattr(env, "_recovery_step_count"):
        return torch.zeros(env.num_envs, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = torch.sum(torch.square(asset.data.joint_vel[:, -4:]), dim=1)
    penalty = penalty * (~_is_freefall(env)).float()
    return penalty


# ── Free-fall reset event ──

def reset_with_freefall(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    drop_height: float = 1.1,
):
    """Reset robot with random orientation at drop_height.

    Paper: random orientation + random joints + zero torques + 1.1m drop.
    First 100 steps (2s): all rewards are zero, robot collapses naturally.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if len(env_ids) == 0:
        return

    # Uniform random quaternion (Shoemake method)
    u1 = torch.rand(len(env_ids), device=env.device)
    u2 = torch.rand(len(env_ids), device=env.device) * 2 * math.pi
    u3 = torch.rand(len(env_ids), device=env.device) * 2 * math.pi
    qw = torch.sqrt(1 - u1) * torch.sin(u2)
    qx = torch.sqrt(1 - u1) * torch.cos(u2)
    qy = torch.sqrt(u1) * torch.sin(u3)
    qz = torch.sqrt(u1) * torch.cos(u3)
    quat = torch.stack([qw, qx, qy, qz], dim=1)

    # Set root state at drop height
    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, 2] = drop_height
    root_state[:, 3:7] = quat
    root_state[:, 7:] = 0.0
    asset.write_root_state_to_sim(root_state, env_ids)

    # Random joint positions + zero velocity
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_pos += torch.empty_like(joint_pos).uniform_(-0.5, 0.5)
    asset.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)

    # Reset step counter
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
    """Boolean mask: which envs successfully recovered."""
    asset: Articulation = env.scene[asset_cfg.name]
    h_ok = asset.data.root_pos_w[:, 2] > height_threshold
    j_ok = torch.norm(asset.data.joint_pos - asset.data.default_joint_pos, dim=1) < joint_threshold
    v_ok = torch.max(torch.abs(asset.data.joint_vel), dim=1).values < vel_threshold
    ideal = torch.tensor([0.0, 0.0, -1.0], device=env.device)
    o_ok = torch.norm(asset.data.projected_gravity_b - ideal, dim=1) < ori_threshold
    return h_ok & j_ok & v_ok & o_ok
