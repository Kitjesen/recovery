# Copyright (c) 2026 Qiongpei Technology
# SPDX-License-Identifier: Apache-2.0

"""Recovery reward functions — Isaac Lab RewardTermCfg compatible.

All functions: func(env, **params) -> Tensor(N,)
ED shaping is built into each task reward via env step counter.
Just register with RewardTermCfg in recovery_env_cfg.py, no runner changes needed.

Based on 'Learning to Recover' (arXiv:2506.05516) Table I.
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


# ── ED / CW helpers ──

def _get_ed(env: ManagerBasedRLEnv, k: int = 3) -> torch.Tensor:
    """Episode-based Dynamic factor (Eq.1). Stored on env for reuse."""
    if not hasattr(env, "_recovery_step_count"):
        env._recovery_step_count = torch.zeros(env.num_envs, device=env.device)
        env._recovery_episode_steps = env.max_episode_length

    T = env._recovery_episode_steps
    a = T / 2.0
    t = env._recovery_step_count.clamp(max=T)
    ed = (a * t / T) ** k
    return ed


def _get_cw(env: ManagerBasedRLEnv, beta: float = 0.3, decay: float = 0.968) -> float:
    """Curriculum Weight (Eq.3). Decays over training."""
    if hasattr(env, "common_step_counter"):
        progress = env.common_step_counter / (env.max_episode_length * 5000)
    else:
        progress = 0.0
    return beta * (decay ** (progress * 10000))


def recovery_step_counter(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Increment ED step counter each step. Register as a reward with weight=0.
    Must be called every step to keep ED counter in sync."""
    if not hasattr(env, "_recovery_step_count"):
        env._recovery_step_count = torch.zeros(env.num_envs, device=env.device)
        env._recovery_episode_steps = env.max_episode_length
    env._recovery_step_count += 1
    return torch.zeros(env.num_envs, device=env.device)


# ── Task Rewards (multiplied by ED) ──

def recovery_stand_joint_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma: float = 0.5,
) -> torch.Tensor:
    """Joints return to default angles. Paper scale=42."""
    asset: Articulation = env.scene[asset_cfg.name]
    error = torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    reward = torch.exp(-error / (sigma ** 2))
    return _get_ed(env) * reward


def recovery_base_height(
    env: ManagerBasedRLEnv,
    target_height: float = 0.55,
    sigma: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Body reaches target height. Paper scale=120. One-sided (no penalty above target)."""
    asset: Articulation = env.scene[asset_cfg.name]
    height_error = torch.clamp(target_height - asset.data.root_pos_w[:, 2], min=0.0)
    reward = torch.exp(-torch.square(height_error) / (sigma ** 2))
    return _get_ed(env) * reward


def recovery_base_orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation from upright. Paper scale=50 (use negative weight)."""
    asset: Articulation = env.scene[asset_cfg.name]
    ideal = torch.tensor([0.0, 0.0, -1.0], device=env.device)
    error = torch.sum(torch.square(asset.data.projected_gravity_b - ideal), dim=1)
    return _get_ed(env) * error


# ── Behavior Rewards (multiplied by CW) ──

def recovery_body_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Body collision penalty. Paper scale=-5e-2."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, 0, :]
    penalty = torch.sum(torch.square(forces), dim=1)
    return _get_cw(env) * penalty


def recovery_action_rate_legs(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Leg action change penalty (wheels excluded!). Paper scale=-1e-2."""
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    leg_diff = torch.sum(torch.square(action[:, :12] - prev_action[:, :12]), dim=1)
    return _get_cw(env) * leg_diff


# ── Constant Penalties ──

def recovery_joint_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint velocity penalty. Paper scale=-2e-2."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel), dim=1)


def recovery_torques(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Torque penalty. Paper scale=-2.5e-5."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque), dim=1)


def recovery_joint_acceleration(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint acceleration penalty. Paper scale=-2.5e-7."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc), dim=1)


def recovery_wheel_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Wheel spin penalty. Paper scale=-2e-2."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, -4:]), dim=1)


# ── Free-fall reset event ──

def reset_with_freefall(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    drop_height: float = 1.1,
):
    """Reset robot with random orientation at drop_height. Gravity does the rest.

    Paper protocol: random orientation + random joints + zero torques + 1.1m drop.
    The 2s free-fall happens naturally in the first ~100 steps of the episode.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if len(env_ids) == 0:
        return

    # Random quaternion
    u1 = torch.rand(len(env_ids), device=env.device)
    u2 = torch.rand(len(env_ids), device=env.device) * 2 * math.pi
    u3 = torch.rand(len(env_ids), device=env.device) * 2 * math.pi
    # Uniform random rotation (Shoemake method)
    qw = torch.sqrt(1 - u1) * torch.sin(u2)
    qx = torch.sqrt(1 - u1) * torch.cos(u2)
    qy = torch.sqrt(u1) * torch.sin(u3)
    qz = torch.sqrt(u1) * torch.cos(u3)
    quat = torch.stack([qw, qx, qy, qz], dim=1)

    # Set root state
    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, 2] = drop_height
    root_state[:, 3:7] = quat
    root_state[:, 7:] = 0.0  # zero velocity
    asset.write_root_state_to_sim(root_state, env_ids)

    # Random joint positions + zero velocity
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_pos += torch.empty_like(joint_pos).uniform_(-0.5, 0.5)
    asset.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)

    # Reset ED step counter
    if hasattr(env, "_recovery_step_count"):
        env._recovery_step_count[env_ids] = 0


# ── Success checker (for logging) ──

def check_recovery_success(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_threshold: float = 0.42,
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
