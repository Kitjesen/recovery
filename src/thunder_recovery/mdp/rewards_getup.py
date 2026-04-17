# Copyright (c) 2026 Qiongpei Technology
# SPDX-License-Identifier: Apache-2.0

"""Getup (Method 2) — mujoco_playground-style rewards for fall recovery.

Reference:
  github.com/google-deepmind/mujoco_playground
  → mujoco_playground/_src/locomotion/go1/getup.py

Design contrast with Method 1 (Deng et al., arXiv:2506.05516, rewards.py):

  Method 1 (ED/CW shaping)                Method 2 (gated exp)
  ----------------------------------      ---------------------------------
  13 terms, time-varying weights          9 terms, plain scalar weights
  ED(t)=(t/T)^3 ramps task reward         Task reward always on
  CW=β·decay^i ramps behaviour penalty    Behaviour penalty always on
  2 s free-fall window (gains = 0)        No free-fall; 0.6 prob drop
  Success-like gating via ED              Gate = upright ∧ at_desired_height
  Wheel-leg coord reward (paper)          —

The key idea of mujoco_playground's approach is that `posture` and
`stand_still` only pay out after the robot is ALREADY upright AND near
the target torso height — so the policy cannot collect them by cheating
(e.g. holding default pose while lying down). `orientation` and
`torso_height` stay non-gated to give a dense gradient out of any
fallen configuration.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from ._utils import _get_joint_split

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Gravity direction in the body frame when the robot is perfectly upright.
_UP_VEC = (0.0, 0.0, -1.0)


# ── Gates ──────────────────────────────────────────────────────────────

def _upright(env: "ManagerBasedRLEnv", asset: Articulation, ori_tol: float) -> torch.Tensor:
    ideal = torch.tensor(_UP_VEC, device=env.device)
    err = torch.sum(torch.square(asset.data.projected_gravity_b - ideal), dim=1)
    return (err < ori_tol).float()


def _at_height(
    env: "ManagerBasedRLEnv", asset: Articulation, z_des: float, pos_tol: float
) -> torch.Tensor:
    h = asset.data.root_pos_w[:, 2]
    h_clamped = torch.minimum(h, torch.full_like(h, z_des))
    return ((z_des - h_clamped) < pos_tol).float()


# ── Task rewards (always on) ───────────────────────────────────────────

def getup_orientation(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """exp(-2 · ‖[0,0,-1] − g_body‖²). Max 1 upright, decays smoothly."""
    asset: Articulation = env.scene[asset_cfg.name]
    ideal = torch.tensor(_UP_VEC, device=env.device)
    err = torch.sum(torch.square(asset.data.projected_gravity_b - ideal), dim=1)
    return torch.exp(-2.0 * err)


def getup_torso_height(
    env: "ManagerBasedRLEnv",
    z_des: float = 0.426,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """exp(min(h, z_des)) − 1. One-sided: no upside for overshooting."""
    asset: Articulation = env.scene[asset_cfg.name]
    h = asset.data.root_pos_w[:, 2]
    h_clamped = torch.minimum(h, torch.full_like(h, z_des))
    return torch.exp(h_clamped) - 1.0


# ── Gated task rewards ────────────────────────────────────────────────

def getup_posture(
    env: "ManagerBasedRLEnv",
    z_des: float = 0.426,
    ori_tol: float = 0.01,
    pos_tol: float = 0.005,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """gate · exp(-0.5 · ‖q − q_default‖²). Gated on upright ∧ at-height."""
    asset: Articulation = env.scene[asset_cfg.name]
    gate = _upright(env, asset, ori_tol) * _at_height(env, asset, z_des, pos_tol)
    cost = torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return gate * torch.exp(-0.5 * cost)


def getup_stand_still(
    env: "ManagerBasedRLEnv",
    z_des: float = 0.426,
    ori_tol: float = 0.01,
    pos_tol: float = 0.005,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """gate · exp(-0.5 · ‖a‖²). Pays out for small actions once standing."""
    asset: Articulation = env.scene[asset_cfg.name]
    gate = _upright(env, asset, ori_tol) * _at_height(env, asset, z_des, pos_tol)
    a = env.action_manager.action
    return gate * torch.exp(-0.5 * torch.sum(torch.square(a), dim=1))


# ── Behaviour costs (always on; negate via weight) ─────────────────────

def getup_action_rate(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """‖aₜ−aₜ₋₁‖² + ‖aₜ−2aₜ₋₁+aₜ₋₂‖²   (mujoco_playground 1st + 2nd order).

    Isaac Lab only exposes `prev_action`, so we cache `prev_prev_action`
    on the env. The cache is updated once per call and reset in
    `reset_getup` to zero on episode boundaries.
    """
    a = env.action_manager.action
    prev = env.action_manager.prev_action
    if not hasattr(env, "_getup_prev_prev_action"):
        env._getup_prev_prev_action = torch.zeros_like(a)
    prev_prev = env._getup_prev_prev_action
    c1 = torch.sum(torch.square(a - prev), dim=1)
    c2 = torch.sum(torch.square(a - 2.0 * prev + prev_prev), dim=1)
    # Advance cache: next call's prev_prev is this call's prev.
    env._getup_prev_prev_action = prev.clone()
    return c1 + c2


def getup_torques(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """√(Στ²) + Σ|τ| on leg joints. mujoco_playground's Go1 is legs-only;
    restricting to legs on Thunder avoids fighting the wheel-velocity
    actuator which is naturally low-torque but high-speed."""
    asset: Articulation = env.scene[asset_cfg.name]
    leg_ids, _ = _get_joint_split(env, asset)
    t = asset.data.applied_torque[:, leg_ids]
    l2 = torch.sqrt(torch.sum(torch.square(t), dim=1) + 1e-12)
    l1 = torch.sum(torch.abs(t), dim=1)
    return l2 + l1


def getup_dof_pos_limits(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Σ soft-limit violation on leg joints (wheels are continuous)."""
    asset: Articulation = env.scene[asset_cfg.name]
    leg_ids, _ = _get_joint_split(env, asset)
    q = asset.data.joint_pos[:, leg_ids]
    soft = asset.data.soft_joint_pos_limits[:, leg_ids]  # (N, J_leg, 2)
    below = -torch.clamp(q - soft[..., 0], max=0.0)
    above = torch.clamp(q - soft[..., 1], min=0.0)
    return torch.sum(below + above, dim=1)


def getup_dof_acc(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Σq̈² on leg joints (wheel acc can be orders of magnitude larger
    during spin-up and would dominate the gradient)."""
    asset: Articulation = env.scene[asset_cfg.name]
    leg_ids, _ = _get_joint_split(env, asset)
    return torch.sum(torch.square(asset.data.joint_acc[:, leg_ids]), dim=1)


def getup_dof_vel(
    env: "ManagerBasedRLEnv",
    max_velocity: float = 6.2832,  # 2π rad/s
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise leg joint velocities exceeding `max_velocity` (2π default).

    Applied to legs only: wheels routinely exceed 2π rad/s during
    recovery (up to ~40 rad/s for wheel-assisted flipping), and
    penalising that would short-circuit the core behaviour.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    leg_ids, _ = _get_joint_split(env, asset)
    over = torch.clamp(torch.abs(asset.data.joint_vel[:, leg_ids]) - max_velocity, min=0.0)
    return torch.sum(torch.square(over), dim=1)
