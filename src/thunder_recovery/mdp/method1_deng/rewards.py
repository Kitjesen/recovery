# Copyright (c) 2026 Qiongpei Technology
# SPDX-License-Identifier: Apache-2.0

"""Recovery reward functions (paper Table I + paper §E support state +
wheel-leg coordination contribution).

Based on 'Learning to Recover: Dynamic Reward Shaping with Wheel-Leg
Coordination for Fallen Robots' (arXiv:2506.05516).

Episode timeline (T = 5 s, 50 Hz, 250 steps):

  ┌──────────────────┬─────────────────────┬──────────────────────┐
  │ Free-fall        │ Exploration         │ Convergence          │
  │ t ∈ [0, 2 s]     │ t ∈ [2, ~3.5 s]     │ t ∈ [~3.5, 5 s]      │
  │ steps 0 – 99     │ steps 100 – ~174    │ steps ~175 – 249     │
  ├──────────────────┼─────────────────────┼──────────────────────┤
  │ ED 0 → 0.064     │ ED 0.064 → 0.34     │ ED 0.34 → 1.0        │
  │ actuator gains 0 │ policy output used  │ policy output used   │
  │ joints floppy    │ (torques active)    │                      │
  │ (true torques=0) │                     │                      │
  ├──────────────────┼─────────────────────┼──────────────────────┤
  │ Diverse fallen   │ Task rewards weak;  │ Task rewards dominate│
  │ states emerge:   │ wheel-leg coord     │ → policy converges   │
  │ reset noise +    │ reward (×(1-ED)·    │ to precise standing  │
  │ floppy free-fall │ tilt) drives wheel- │ posture.             │
  │                  │ assisted flipping.  │                      │
  └──────────────────┴─────────────────────┴──────────────────────┘

ED(t) = (t / T)^3  ∈ [0, 1] — paper Eq. 1 (normalised).
CW(i) = β · decay^i with β=0.3, decay=0.968 — paper Eq. 3.

The event-side machinery (reset_with_freefall, zero_action_freefall) and
the critic's privileged observations (priv_*) live in sibling modules
`events.py` and `observations.py`.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .._utils import (
    _advance_step_counter,
    _env_dt,
    _ensure_step_counter,
    _get_cw,
    _get_ed,
    _get_joint_split,
    _is_freefall,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ── Step counter (side-effect term that advances the ED clock) ──

def recovery_step_counter(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Advances the per-env ED step counter. Must run every step.

    Return is zero; real side effect is the counter increment. Weight is
    1e-6 in the env cfg (tiny-but-nonzero — avoids reward-manager pruning
    seen at exactly 0.0 or ≤1e-9 in some Isaac Lab releases) so it does
    not meaningfully affect the optimisation loss.
    """
    _advance_step_counter(env)
    return torch.zeros(env.num_envs, device=env.device)


# ── Task rewards (× ED) ──

def recovery_stand_joint_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma: float = 0.5,
) -> torch.Tensor:
    """ED · exp(-‖q − q_default‖² / σ²). Table I scale = 42."""
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
    """ED · exp(-clip(h_t − h, 0)² / σ²). Table I scale = 120.

    One-sided clamp: no credit for being below target, no penalty for
    being above (rare during recovery). Upside is dominated by the
    joint-pos and orientation terms once the robot is upright.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    height_error = torch.clamp(target_height - asset.data.root_pos_w[:, 2], min=0.0)
    raw = torch.exp(-torch.square(height_error) / (sigma ** 2))
    return _get_ed(env) * raw


def recovery_base_orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """ED · exp(-‖g_body − [0, 0, -1]‖²). Table I scale = 50.

    Paper: "penalizes gravitational-projection misalignment for roll/pitch".
    The ‖⋅‖² form is zero when upright and ~4 upside-down.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ideal = torch.tensor([0.0, 0.0, -1.0], device=env.device)
    error = torch.sum(torch.square(asset.data.projected_gravity_b - ideal), dim=1)
    raw = torch.exp(-error)
    return _get_ed(env) * raw


# ── Support state (paper §E, per-step binary) ──

def recovery_support_state(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 1.0,
    min_base_height: float = 0.30,
) -> torch.Tensor:
    """Support state — 4 feet contact AND base held above the ground.

    Paper text only required "all four wheels in contact". On a wheeled-
    legged robot that condition is satisfied by a belly-flat pose with
    wheels grounded (hip/thigh/calf fully folded), which turned out to
    be a strong local optimum during training: policy collected the
    support reward by staying down rather than standing up.

    Adding a `min_base_height` gate (default 0.30 m, matches success
    criteria) means the policy only earns the reward while actually
    holding the body up. Empirically: without the gate the robot at
    iter ~500 ends up with base_height ~0.16 m and support_state ~0.69
    (collected by lying flat with 4 wheels touching).

    Suppressed during free-fall.
    """
    freefall = _is_freefall(env)
    if sensor_cfg.body_ids is None or sensor_cfg.body_ids == slice(None):
        raise RuntimeError(
            "recovery_support_state requires sensor_cfg.body_ids to be "
            "resolved from a body_names regex; no safe index fallback exists."
        )
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, 0, :, :]
    magnitude = torch.norm(forces, dim=-1)[:, sensor_cfg.body_ids]
    all_feet = (magnitude > threshold).all(dim=1)

    asset: Articulation = env.scene[asset_cfg.name]
    base_up = asset.data.root_pos_w[:, 2] > min_base_height

    return (all_feet & base_up & ~freefall).float()


# ── Behavior rewards (× CW) ──

def recovery_body_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    force_clip: float = 50.0,
) -> torch.Tensor:
    """CW · Σ clip(‖λ_b‖, 0, force_clip)² on base / thigh / calf.

    Paper form is Σ ‖λ_b‖² with no clip; we clip per-body magnitude so
    impulse contacts on reset do not spike the gradient early in
    training. Scale -5e-2 matches Table I.
    """
    if _is_freefall(env).all():
        return torch.zeros(env.num_envs, device=env.device)
    if sensor_cfg.body_ids is None or sensor_cfg.body_ids == slice(None):
        raise RuntimeError(
            "recovery_body_collision requires sensor_cfg.body_ids to be "
            "resolved from a body_names regex; no safe index fallback exists."
        )
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, 0, :, :]
    magnitude = torch.norm(forces, dim=-1)[:, sensor_cfg.body_ids]
    magnitude = torch.clamp(magnitude, max=force_clip)
    penalty = torch.sum(torch.square(magnitude), dim=1)
    return _get_cw(env) * penalty


def recovery_action_rate_legs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """CW · Σ (aₜ − aₜ₋₁)² on leg action dims only.

    Assumes the action manager layout matches joint layout (first N
    action dims = leg joints). Thunder's action cfg is joint_pos(legs) +
    joint_vel(wheels) concatenated in that order.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    leg_ids, _ = _get_joint_split(env, asset)
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    leg_diff = torch.sum(torch.square(action[:, leg_ids] - prev_action[:, leg_ids]), dim=1)
    return _get_cw(env) * leg_diff


# ── Constant penalties (legs only — wheels have dedicated terms) ──

def recovery_joint_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Σ q̇² over leg joints. Wheels excluded to avoid fighting the
    wheel-leg coordination reward in the exploration phase."""
    asset: Articulation = env.scene[asset_cfg.name]
    leg_ids, _ = _get_joint_split(env, asset)
    return torch.sum(torch.square(asset.data.joint_vel[:, leg_ids]), dim=1)


def recovery_torques(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Σ τ² over leg joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    leg_ids, _ = _get_joint_split(env, asset)
    return torch.sum(torch.square(asset.data.applied_torque[:, leg_ids]), dim=1)


def recovery_joint_acceleration(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Σ q̈² over leg joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    leg_ids, _ = _get_joint_split(env, asset)
    return torch.sum(torch.square(asset.data.joint_acc[:, leg_ids]), dim=1)


def recovery_wheel_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """ED · Σ ω_wheel². Early phase (ED≈0): wheels free to spin for
    wheel-leg coordinated flipping. Late phase (ED→1): full penalty →
    converge to still stance."""
    asset: Articulation = env.scene[asset_cfg.name]
    _, wheel_ids = _get_joint_split(env, asset)
    penalty = torch.sum(torch.square(asset.data.joint_vel[:, wheel_ids]), dim=1)
    return _get_ed(env) * penalty


# ── Wheel-leg coordination (paper core contribution) ──


def recovery_joint_deviation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Post-freefall L2 joint deviation from default pose (constant penalty).

    Unlike recovery_stand_joint_pos (x ED, only active late in the episode),
    this penalty is always on after free-fall ends, giving the policy a
    persistent gradient signal to move leg joints toward the standing pose.

    Free-fall steps are masked: stiffness is zero so the policy cannot
    control joints and penalising deviation would only bias the value
    baseline without producing useful gradients.

    raw = sum((q_leg - q_default_leg)^2)  (only leg joints, not wheels)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    leg_ids, _ = _get_joint_split(env, asset)
    deviation = torch.sum(
        torch.square(asset.data.joint_pos[:, leg_ids] - asset.data.default_joint_pos[:, leg_ids]),
        dim=1,
    )
    freefall = _is_freefall(env)
    return deviation * (~freefall).float()


def recovery_wheel_leg_coord(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_wheel_speed: float = 40.0,
) -> torch.Tensor:
    """Rewards spinning wheels while the body is tilted, i.e. using wheel
    ground-reaction to assist flipping instead of relying only on legs.
    The paper credits this synergy for -15.8% to -26.2% joint torque.

    r = (1 - ED) · (|ω_wheel| / max_wheel_speed) · ‖g_xy‖

    Three-way gated:
      - (1 - ED) ∈ [1, 0]: only active in exploration, decays to 0 in convergence.
      - (|ω_wheel| / max_wheel_speed) ∈ [0, 1]: rewards wheel spin.
      - ‖g_xy‖ ∈ [0, 1]: tilt factor, 0 upright.

    Zero upright, zero in convergence — does not interfere with the
    task/ED convergence phase.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    _, wheel_ids = _get_joint_split(env, asset)
    early_gate = 1.0 - _get_ed(env)
    wheel_speed = torch.sum(torch.abs(asset.data.joint_vel[:, wheel_ids]), dim=1)
    wheel_speed = torch.clamp(wheel_speed, max=max_wheel_speed) / max_wheel_speed
    tilt = torch.clamp(torch.norm(asset.data.projected_gravity_b[:, :2], dim=1), 0.0, 1.0)
    return early_gate * wheel_speed * tilt


# ── Success metric (paper criteria, logging only) ──

def check_recovery_success(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_threshold: float = 0.30,
    joint_threshold: float = 0.5,
    vel_threshold: float = 0.1,
    ori_threshold: float = 0.1,
) -> torch.Tensor:
    """Bool per env — paper's four-way success condition (height, joint
    deviation, joint velocity, orientation)."""
    asset: Articulation = env.scene[asset_cfg.name]
    h_ok = asset.data.root_pos_w[:, 2] > height_threshold
    j_ok = torch.norm(asset.data.joint_pos - asset.data.default_joint_pos, dim=1) < joint_threshold
    v_ok = torch.max(torch.abs(asset.data.joint_vel), dim=1).values < vel_threshold
    ideal = torch.tensor([0.0, 0.0, -1.0], device=env.device)
    o_ok = torch.norm(asset.data.projected_gravity_b - ideal, dim=1) < ori_threshold
    return h_ok & j_ok & v_ok & o_ok


def recovery_success_rate(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_threshold: float = 0.30,
    joint_threshold: float = 0.5,
    vel_threshold: float = 0.1,
    ori_threshold: float = 0.1,
) -> torch.Tensor:
    """0/1 success indicator active only in the final second of the
    episode, so tensorboard sees an end-of-episode success rate rather
    than a mid-episode partial-credit signal. Wired as a 1e-6-weight
    reward term (logging only, no effect on optimisation)."""
    success = check_recovery_success(
        env, asset_cfg, height_threshold, joint_threshold, vel_threshold, ori_threshold
    ).float()
    _ensure_step_counter(env)
    last_second = (
        env._recovery_step_count >= (env.max_episode_length - int(1.0 / _env_dt(env)))
    ).float()
    return success * last_second
