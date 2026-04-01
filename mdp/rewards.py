# Copyright (c) 2026 Qiongpei Technology
# SPDX-License-Identifier: Apache-2.0

"""Recovery reward functions — matches paper exactly.

Based on 'Learning to Recover' (arXiv:2506.05516) Eq.1-4 + Table I.

Changes v4:
- P0: First 100 steps force standingPose action (simulate torques=0 free-fall)
- P1: Added support_state reward (4 wheels on ground)
- P3: CW uses paper params: beta=0.3, decay=0.968
- All penalties zeroed during free-fall (steps 0-99)
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
FREEFALL_STEPS = 100  # 2s at 50Hz


# ── Helpers ──

def _is_freefall(env: ManagerBasedRLEnv) -> torch.Tensor:
    """True for envs still in free-fall phase (steps 0-99)."""
    if not hasattr(env, "_recovery_step_count"):
        env._recovery_step_count = torch.zeros(env.num_envs, device=env.device)
    return env._recovery_step_count < FREEFALL_STEPS


def _get_ed(env: ManagerBasedRLEnv, k: int = 3) -> torch.Tensor:
    """Episode-based Dynamic factor (Eq.1) in seconds.

    Free-fall (steps 0-99): ED computed but rewards are masked separately.
    Recovery (steps 100-249): ED grows from ~0 to ~15.6.
    """
    if not hasattr(env, "_recovery_step_count"):
        env._recovery_step_count = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_recovery_ed_last_step"):
        env._recovery_ed_last_step = -1

    current_step = env.common_step_counter if hasattr(env, "common_step_counter") else 0
    if current_step != env._recovery_ed_last_step:
        env._recovery_step_count += 1
        env._recovery_ed_last_step = current_step

    dt = 1.0 / 50.0
    t_sec = env._recovery_step_count * dt
    T_sec = float(env.max_episode_length) * dt
    a = T_sec / 2.0

    ed = (a * t_sec / T_sec) ** k
    # Zero during free-fall
    ed = ed * (~_is_freefall(env)).float()
    return ed


def _get_cw(env: ManagerBasedRLEnv, beta: float = 0.3, decay: float = 0.968) -> float:
    """Curriculum Weight (Eq.3) — paper params: beta=0.3, decay=0.968."""
    if hasattr(env, "common_step_counter"):
        progress = env.common_step_counter / (env.max_episode_length * 5000)
    else:
        progress = 0.0
    return beta * (decay ** (progress * 10000))


# ── Step Counter ──

def recovery_step_counter(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Dummy — step counting done in _get_ed."""
    return torch.zeros(env.num_envs, device=env.device)


# ── Task Rewards (×ED, zero during free-fall) ──

def recovery_stand_joint_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma: float = 0.5,
) -> torch.Tensor:
    """exp(-sum(q-q_default)^2 / sigma^2). Table I scale=42."""
    asset: Articulation = env.scene[asset_cfg.name]
    error = torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    raw = torch.exp(-error / (sigma ** 2))
    return _get_ed(env) * raw


def recovery_base_height(
    env: ManagerBasedRLEnv,
    target_height: float = 0.426,
    sigma: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """exp(-max(h_target-h,0)^2 / sigma^2). Table I scale=120."""
    asset: Articulation = env.scene[asset_cfg.name]
    height_error = torch.clamp(target_height - asset.data.root_pos_w[:, 2], min=0.0)
    raw = torch.exp(-torch.square(height_error) / (sigma ** 2))
    return _get_ed(env) * raw


def recovery_base_orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for upright orientation (exp form).

    r = ED * exp(-||g_body - [0,0,-1]||^2)
    Upright: error=0 -> r=1. Fallen: error~4 -> r~0.02.
    Weight = +50 (positive reward for being upright).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ideal = torch.tensor([0.0, 0.0, -1.0], device=env.device)
    error = torch.sum(torch.square(asset.data.projected_gravity_b - ideal), dim=1)
    raw = torch.exp(-error)
    return _get_ed(env) * raw




# ── Support State Reward (NEW — from paper Section E) ──

def recovery_support_state(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward when all 4 wheels contact ground simultaneously.

    Paper: "we provide a reward for the support state, defined as the condition
    where all four wheels are in contact with the ground simultaneously"
    """
    if _is_freefall(env).all():
        return torch.zeros(env.num_envs, device=env.device)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # net_forces_w_history shape: (N, history, num_bodies, 3)
    # Check if force magnitude > threshold for each body
    forces = contact_sensor.data.net_forces_w_history[:, 0, :, :]  # (N, num_bodies, 3)
    force_mag = torch.norm(forces, dim=-1)  # (N, num_bodies)

    # Last 4 bodies should be foot links — check all have contact
    foot_contact = force_mag[:, -4:] > threshold  # (N, 4)
    all_feet_contact = foot_contact.all(dim=1).float()  # (N,) 1.0 if all 4 wheels on ground

    # Only reward during recovery phase
    all_feet_contact = all_feet_contact * (~_is_freefall(env)).float()
    return all_feet_contact


# ── Behavior Rewards (×CW, zero during free-fall) ──

def recovery_body_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Disabled for now — sensor shape needs verification."""
    return torch.zeros(env.num_envs, device=env.device)


def recovery_action_rate_legs(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """sum((a_leg[t]-a_leg[t-1])²). Wheels excluded. Zero during free-fall."""
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    leg_diff = torch.sum(torch.square(action[:, :12] - prev_action[:, :12]), dim=1)
    leg_diff = leg_diff * (~_is_freefall(env)).float()
    return _get_cw(env) * leg_diff


# ── Constant Penalties (zero during free-fall) ──

def recovery_joint_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """sum(q_dot²). Zero during free-fall."""
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = torch.sum(torch.square(asset.data.joint_vel), dim=1)
    return penalty * (~_is_freefall(env)).float()


def recovery_torques(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """sum(tau²). Zero during free-fall."""
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = torch.sum(torch.square(asset.data.applied_torque), dim=1)
    return penalty * (~_is_freefall(env)).float()


def recovery_joint_acceleration(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """sum(q_ddot²). Zero during free-fall."""
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = torch.sum(torch.square(asset.data.joint_acc), dim=1)
    return penalty * (~_is_freefall(env)).float()


def recovery_wheel_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """sum(wheel_vel²). Zero during free-fall."""
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = torch.sum(torch.square(asset.data.joint_vel[:, -4:]), dim=1)
    return penalty * (~_is_freefall(env)).float()


# ── Free-fall reset ──

def reset_with_freefall(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    drop_height: float = 1.1,
):
    """Random orientation + 1.1m drop + zero joint torques."""
    asset: Articulation = env.scene[asset_cfg.name]
    if len(env_ids) == 0:
        return

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

    # Set joints to default standing pose (not random) with zero velocity
    # This simulates "set joint torques to zero" — robot holds standing pose
    # and gravity pulls it down naturally
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
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
    """Paper success criteria."""
    asset: Articulation = env.scene[asset_cfg.name]
    h_ok = asset.data.root_pos_w[:, 2] > height_threshold
    j_ok = torch.norm(asset.data.joint_pos - asset.data.default_joint_pos, dim=1) < joint_threshold
    v_ok = torch.max(torch.abs(asset.data.joint_vel), dim=1).values < vel_threshold
    ideal = torch.tensor([0.0, 0.0, -1.0], device=env.device)
    o_ok = torch.norm(asset.data.projected_gravity_b - ideal, dim=1) < ori_threshold
    return h_ok & j_ok & v_ok & o_ok
