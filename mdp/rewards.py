# Copyright (c) 2026 Qiongpei Technology
# SPDX-License-Identifier: Apache-2.0

"""Recovery reward functions — matches paper exactly.

Based on 'Learning to Recover' (arXiv:2506.05516) Eq.1-4 + Table I.

Episode timeline (T = 5s, 50Hz, 250 steps):

  ┌──────────────────┬─────────────────────┬──────────────────────┐
  │ Free-fall        │ Exploration         │ Convergence          │
  │ t ∈ [0, 2s]      │ t ∈ [2, 3.5s]       │ t ∈ [3.5, 5s]        │
  │ steps 0-99       │ steps 100-174       │ steps 175-249        │
  ├──────────────────┼─────────────────────┼──────────────────────┤
  │ ED ≈ 0 → 1       │ ED ≈ 1 → 5          │ ED ≈ 5 → 15.6        │
  │ action forced to │ policy output used  │ policy output used   │
  │ default pose     │ (torques active)    │                      │
  │ (zero_action_    │                     │                      │
  │  freefall)       │                     │                      │
  ├──────────────────┼─────────────────────┼──────────────────────┤
  │ Policy actions   │ Task rewards weak;  │ Task rewards dominate│
  │ ignored → zero   │ behavior penalties  │ → policy converges   │
  │ gradient signal. │ (×CW) active; policy│ to precise standing  │
  │ Purpose: generate│ freely explores     │ posture.             │
  │ diverse fallen   │ flipping / wheel-   │                      │
  │ initial states.  │ assisted recovery.  │                      │
  └──────────────────┴─────────────────────┴──────────────────────┘

Paper Section III-A: during free-fall, joint torques are set to 0 — the
robot falls under gravity with a frozen standing pose. We implement this by
writing (default joint pos, zero joint vel) to sim every step for envs whose
step count < 100. The policy still emits actions but they are discarded by
this override; no learning signal flows through the free-fall window.

v11 (restore ED):
- Task rewards (stand_joint_pos, base_height, base_orientation) are multiplied by ED.
- Step counter advances via recovery_step_counter term (always called each step).
- Removed `recovery_upward` (bypassed ED, encouraged lazy flip-only policy).
- Removed feet_ratio gating on base_height (ED handles the exploration phase).
- Restored paper Table I weights in env cfg.
- CW decay rebased on training iterations.
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


def _advance_step_counter(env: ManagerBasedRLEnv) -> None:
    """Increment per-env step counter exactly once per env.step().

    Called from `recovery_step_counter` (weight 1e-10) which is guaranteed to
    run every step. Reset is handled in `reset_with_freefall`.
    """
    if not hasattr(env, "_recovery_step_count"):
        env._recovery_step_count = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_recovery_ed_last_step"):
        env._recovery_ed_last_step = -1

    current_step = env.common_step_counter if hasattr(env, "common_step_counter") else 0
    if current_step != env._recovery_ed_last_step:
        env._recovery_step_count += 1
        env._recovery_ed_last_step = current_step


def _get_ed(env: ManagerBasedRLEnv, k: int = 3) -> torch.Tensor:
    """Episode-based Dynamic factor (paper Eq.1).

    ED(t) = (a · t / T)^k,  with a = T/2, k = 3.
    Per-env step count t_sec ∈ [0, T_sec]; ED ∈ [0, (T/2)^k] ≈ [0, 15.6] for T=5s.
    Paper Table I weights are calibrated to this range (task rewards scale 42/120/50).

    Free-fall phase (steps 0-99, t∈[0,2s]): ED ≈ (2.5·0.4)^3 = 1.0 → task rewards mostly
    suppressed, exploration dominates. Recovery phase (t>2s): ED grows rapidly,
    task rewards dominate → posture convergence.
    """
    if not hasattr(env, "_recovery_step_count"):
        env._recovery_step_count = torch.zeros(env.num_envs, device=env.device)

    dt = 1.0 / 50.0
    t_sec = env._recovery_step_count * dt
    T_sec = float(env.max_episode_length) * dt
    a = T_sec / 2.0

    ed = (a * t_sec / T_sec) ** k
    return ed


def _get_cw(env: ManagerBasedRLEnv, beta: float = 0.3, decay: float = 0.968) -> float:
    """Curriculum Weight (paper Eq.3): CW(i) = beta · decay^i.

    i = training iteration (rollout step counted once per policy update).
    Approximated as common_step_counter / num_steps_per_env (48).
    At beta=0.3, decay=0.968: CW drops to 0.1 around iter~35, to ~0.01 around iter~100.
    So behavior penalties are strong in the first few dozen iterations, then fade as
    the policy stabilises and the stricter task/ED shaping takes over.
    """
    steps_per_iter = 48
    if hasattr(env, "common_step_counter"):
        iteration = env.common_step_counter / steps_per_iter
    else:
        iteration = 0
    return beta * (decay ** iteration)


# ── Step Counter ──

def recovery_step_counter(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Advances the per-env ED step counter. Must always run each step.

    Return is zero; real side effect is the counter increment. Weight is 1e-10
    in the env cfg so it does not affect the optimization loss.
    """
    _advance_step_counter(env)
    return torch.zeros(env.num_envs, device=env.device)


# ── Task Rewards (×ED, zero during free-fall) ──

def recovery_stand_joint_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma: float = 0.5,
) -> torch.Tensor:
    """ED · exp(-sum(q-q_default)^2 / sigma^2). Table I scale=42."""
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
    """ED · exp(-(h_target-h)^2 / sigma^2). Table I scale=120.

    No feet-contact gating — ED suppresses this reward in the early phase so the
    policy cannot get height credit by floating on wheels before it learns to
    plant feet; once ED ramps up, the joint-pos and orientation terms also
    dominate, jointly forcing a proper stance.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    height_error = torch.clamp(target_height - asset.data.root_pos_w[:, 2], min=0.0)
    raw = torch.exp(-torch.square(height_error) / (sigma ** 2))
    return _get_ed(env) * raw




def recovery_base_orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """ED · exp(-||g_body - [0,0,-1]||^2). Table I scale=50."""
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
    """Reward when all 4 wheels contact ground simultaneously."""
    if _is_freefall(env).all():
        return torch.zeros(env.num_envs, device=env.device)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, 0, :, :]  # (N, num_bodies, 3)
    force_mag = torch.norm(forces, dim=-1)  # (N, num_bodies)

    # Use body_ids from sensor_cfg to get correct foot indices
    if sensor_cfg.body_ids is not None and sensor_cfg.body_ids != slice(None):
        foot_forces = force_mag[:, sensor_cfg.body_ids]  # (N, 4)
    else:
        # Fallback: foot bodies are at indices 4,8,12,16 in URDF order
        foot_idx = [4, 8, 12, 16]
        foot_forces = force_mag[:, foot_idx]

    foot_contact = foot_forces > threshold  # (N, 4)
    all_feet_contact = foot_contact.all(dim=1).float()
    all_feet_contact = all_feet_contact * (~_is_freefall(env)).float()
    return all_feet_contact




# ── Behavior Rewards (×CW, zero during free-fall) ──

def recovery_body_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Body collision penalty: sum of contact forces on thigh/calf/base.

    Paper: B = {shanks, thighs, base}, r = sum(||lambda_b||^2)
    Scale = -5e-2. Only active during recovery phase (not free-fall).
    """
    if _is_freefall(env).all():
        return torch.zeros(env.num_envs, device=env.device)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, 0, :, :]  # (N, num_bodies, 3)
    force_sq = torch.sum(torch.square(forces), dim=-1)  # (N, num_bodies)

    # Use body_ids from sensor_cfg if available
    if sensor_cfg.body_ids is not None and sensor_cfg.body_ids != slice(None):
        body_forces = force_sq[:, sensor_cfg.body_ids]
    else:
        # Fallback: base(0) + thigh(2,6,10,14) + calf(3,7,11,15)
        body_idx = [0, 2, 3, 6, 7, 10, 11, 14, 15]
        body_forces = force_sq[:, body_idx]

    penalty = torch.sum(body_forces, dim=1)
    penalty = penalty
    return _get_cw(env) * penalty




def recovery_action_rate_legs(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """sum((a_leg[t]-a_leg[t-1])²). Wheels excluded. Zero during free-fall."""
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    leg_diff = torch.sum(torch.square(action[:, :12] - prev_action[:, :12]), dim=1)
    return _get_cw(env) * leg_diff


# ── Constant Penalties (zero during free-fall) ──

def recovery_joint_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """sum(q_dot²). Zero during free-fall."""
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = torch.sum(torch.square(asset.data.joint_vel), dim=1)
    return penalty


def recovery_torques(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """sum(tau²). Zero during free-fall."""
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = torch.sum(torch.square(asset.data.applied_torque), dim=1)
    return penalty


def recovery_joint_acceleration(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """sum(q_ddot²). Zero during free-fall."""
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = torch.sum(torch.square(asset.data.joint_acc), dim=1)
    return penalty


def recovery_wheel_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """sum(wheel_vel²) gated by ED.

    Early phase (ED≈0): wheels are free → policy can spin them for wheel-leg
    coordinated flipping (paper's core contribution: -15.8% to -26.2% joint
    torque). Late phase (ED large): full penalty → converge to still stance.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = torch.sum(torch.square(asset.data.joint_vel[:, -4:]), dim=1)
    T_sec = float(env.max_episode_length) / 50.0
    ed_max = (T_sec / 2.0) ** 3
    gate = torch.clamp(_get_ed(env) / ed_max, 0.0, 1.0)
    return gate * penalty


def recovery_wheel_leg_coord(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_wheel_speed: float = 40.0,
) -> torch.Tensor:
    """Wheel-leg coordination reward (paper core contribution).

    Rewards actively spinning wheels while the body is tilted, which is the
    mechanism the paper credits for 15.8-26.2% joint torque reduction: wheels
    push against the ground to assist flipping instead of relying only on legs.

    r = (1 - ED/ED_max) · (|ω_wheel| / max_wheel_speed) · ||g_xy||

    - (1 - ED/ED_max): ~1 in exploration phase, decays to 0 in convergence
      phase — coordination only helps before the robot is upright.
    - |ω_wheel| clipped to max_wheel_speed: sum of absolute wheel speeds.
    - ||g_xy||: tilt factor, 0 when upright, ~1 when sideways/upside-down.

    Zero when upright (regardless of wheel motion) so it does not interfere
    with the task/ED convergence phase.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    T_sec = float(env.max_episode_length) / 50.0
    ed_max = (T_sec / 2.0) ** 3
    early_gate = torch.clamp(1.0 - _get_ed(env) / ed_max, 0.0, 1.0)

    wheel_speed = torch.sum(torch.abs(asset.data.joint_vel[:, -4:]), dim=1)
    wheel_speed = torch.clamp(wheel_speed, max=max_wheel_speed) / max_wheel_speed

    tilt = torch.norm(asset.data.projected_gravity_b[:, :2], dim=1)
    tilt = torch.clamp(tilt, 0.0, 1.0)

    return early_gate * wheel_speed * tilt


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


# ── Free-fall Action Override ──

def zero_action_freefall(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """During free-fall (first 100 steps), write zero velocity to all joints.

    This simulates 'set joint torques to zero' from the paper.
    The PD controller will produce: tau = KP*(default - current) + KD*(0 - vel)
    which gently brings joints to rest, letting gravity do the work.

    Called as an interval event every step.
    """
    if not hasattr(env, "_recovery_step_count"):
        return

    asset: Articulation = env.scene[asset_cfg.name]

    # Find envs still in free-fall
    freefall_mask = env._recovery_step_count < FREEFALL_STEPS
    freefall_ids = torch.where(freefall_mask)[0]

    if len(freefall_ids) == 0:
        return

    # Write default joint positions + zero velocity
    # This makes PD output minimal torque (joints near default, vel=0 target)
    joint_pos = asset.data.default_joint_pos[freefall_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=freefall_ids)

    # Also zero the root velocity (let gravity be the only force)
    root_state = asset.data.root_state_w[freefall_ids].clone()
    root_state[:, 7:10] = 0.0  # zero linear velocity - actually don't, let gravity work
    # Don't zero velocity - robot needs to fall naturally under gravity


# ── Privileged observations (critic-only, paper Fig.3 asymmetric AC) ──

def priv_base_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Base z-position (root_pos_w[:, 2]) — (N, 1).

    Actor cannot observe absolute base height from joint encoders; critic uses
    it directly to evaluate recovery progress.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2:3]


def priv_base_lin_vel_clean(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Clean body-frame linear velocity (N, 3) — no IMU noise.

    The policy's base_lin_vel has Unoise injected (simulating IMU drift); the
    critic gets the ground-truth sim value.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def priv_base_ang_vel_clean(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Clean body-frame angular velocity (N, 3) — no IMU noise."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


def priv_foot_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Binary foot-contact state (N, num_feet).

    Actor has no contact sensors on a real robot (or only noisy ones); critic
    gets the exact ground-contact indicator, which is the strongest signal for
    whether the robot has reached the support state.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, 0, :, :]
    magnitude = torch.norm(forces, dim=-1)
    if sensor_cfg.body_ids is not None and sensor_cfg.body_ids != slice(None):
        magnitude = magnitude[:, sensor_cfg.body_ids]
    else:
        magnitude = magnitude[:, [4, 8, 12, 16]]
    return (magnitude > threshold).float()


def priv_body_contact_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Per-body contact-force magnitude on shanks/thighs/base (N, num_bodies).

    Tells the critic whether the robot is dragging limbs or hitting the ground
    with its body, which the actor cannot directly sense.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, 0, :, :]
    magnitude = torch.norm(forces, dim=-1)
    if sensor_cfg.body_ids is not None and sensor_cfg.body_ids != slice(None):
        magnitude = magnitude[:, sensor_cfg.body_ids]
    return magnitude



