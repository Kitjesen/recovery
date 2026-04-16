# Copyright (c) 2026 Qiongpei Technology
# SPDX-License-Identifier: Apache-2.0

"""Recovery reward functions — full paper alignment.

Based on 'Learning to Recover: Dynamic Reward Shaping with Wheel-Leg
Coordination for Fallen Robots' (arXiv:2506.05516).

Episode timeline (T = 5s, 50 Hz, 250 steps):

  ┌──────────────────┬─────────────────────┬──────────────────────┐
  │ Free-fall        │ Exploration         │ Convergence          │
  │ t ∈ [0, 2s]      │ t ∈ [2, ~3.5s]      │ t ∈ [~3.5, 5s]       │
  │ steps 0-99       │ steps 100-~175      │ steps ~175-249       │
  ├──────────────────┼─────────────────────┼──────────────────────┤
  │ ED ≈ 0 → 0.064   │ ED ≈ 0.064 → 0.34   │ ED ≈ 0.34 → 1.0      │
  │ actuator gains 0 │ policy output used  │ policy output used   │
  │ joints floppy    │ (torques active)    │                      │
  │ (true torques=0) │                     │                      │
  ├──────────────────┼─────────────────────┼──────────────────────┤
  │ Diverse fallen   │ Task rewards weak;  │ Task rewards dominate│
  │ states emerge:   │ wheel-leg coord     │ → policy converges   │
  │ reset noise +    │ reward (×(1-ED)·    │ to precise standing  │
  │ floppy-joint     │ tilt) drives wheel- │ posture.             │
  │ free-fall.       │ assisted flipping.  │                      │
  └──────────────────┴─────────────────────┴──────────────────────┘

ED(t)  = (t/T)^3   ∈ [0, 1]   — paper Eq. 1, normalised.
CW(i)  = β · decay^i  with β=0.3, decay=0.968  — paper Eq. 3, iter-indexed.

Free-fall (Section III-A): paper verbatim "randomly initializing the
robot's base orientation and joint angles, setting the joint torques to
zero, and letting the robot free-fall from a height of 1.1 m for 2
seconds". We reproduce this by:
  - reset_with_freefall: random SO(3) orientation, 1.1 m drop, ±0.3 rad
    uniform perturbation on leg joints.
  - zero_action_freefall: for each env with step < 100, zero the per-env
    actuator stiffness/damping and pin the PD target to the current
    joint_pos so effective torque is 0. Cached gains are restored when
    the env exits free-fall. Falls back to a rigid-at-default teleport
    if the actuator class does not expose mutable per-env gain tensors.

Support state (Section E): paper verbatim "a reward for the support
state, defined as the condition where all four wheels are in contact
with the ground simultaneously". Per-step binary reward.

Wheel-leg coordination (paper core contribution, -15.8% to -26.2% joint
torque reduction): encouraged via recovery_wheel_leg_coord (positive
reward when wheels spin while body is tilted, active only in the
exploration phase) and by ED-gating the wheel velocity penalty so wheels
are free to assist flipping before the convergence phase.
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

# Joint index resolution: DO NOT hardcode leg/wheel split. Thunder has 16
# joints but the URDF order is not guaranteed to be legs-first. We cache the
# resolved indices on the env the first time a reward needs them.


# ── Helpers ──

def _ensure_step_counter(env: ManagerBasedRLEnv) -> None:
    """Create the per-env step counter lazily (int64, exact)."""
    if not hasattr(env, "_recovery_step_count"):
        env._recovery_step_count = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )


def _get_joint_split(env: ManagerBasedRLEnv, asset: Articulation) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (leg_ids, wheel_ids) as long tensors, cached on the env.

    Resolves joint names matching `.*wheel.*` as wheels, the rest as legs.
    Raises if either group is empty (misconfigured URDF or name regex).
    """
    if not hasattr(env, "_recovery_joint_split"):
        wheel_ids, _ = asset.find_joints(".*wheel.*")
        all_ids = list(range(asset.data.joint_pos.shape[1]))
        leg_ids = [i for i in all_ids if i not in wheel_ids]
        if not wheel_ids or not leg_ids:
            raise RuntimeError(
                f"recovery: could not split joints by '.*wheel.*' regex. "
                f"Got wheel_ids={wheel_ids}, leg_ids={leg_ids}. "
                f"Asset joint_names={asset.data.joint_names}."
            )
        device = asset.data.joint_pos.device
        env._recovery_joint_split = (
            torch.tensor(leg_ids, dtype=torch.long, device=device),
            torch.tensor(wheel_ids, dtype=torch.long, device=device),
        )
    return env._recovery_joint_split


def _env_dt(env: ManagerBasedRLEnv) -> float:
    """Control-step duration (seconds). Uses env.step_dt when available."""
    if hasattr(env, "step_dt"):
        return float(env.step_dt)
    if (
        hasattr(env, "cfg")
        and hasattr(env.cfg, "sim")
        and hasattr(env.cfg.sim, "dt")
        and hasattr(env.cfg, "decimation")
    ):
        return float(env.cfg.sim.dt) * float(env.cfg.decimation)
    return 1.0 / 50.0


def _is_freefall(env: ManagerBasedRLEnv) -> torch.Tensor:
    """True for envs still in free-fall phase (steps 0-99)."""
    _ensure_step_counter(env)
    return env._recovery_step_count < FREEFALL_STEPS


def _advance_step_counter(env: ManagerBasedRLEnv) -> None:
    """Increment per-env step counter exactly once per env.step().

    Called from `recovery_step_counter` (weight 1e-10) which is guaranteed to
    run every step. Reset is handled in `reset_with_freefall`. Counter is
    clamped to `max_episode_length` so ED never exceeds 1.0 even if a reward
    term reads the counter after the final step of an episode.
    """
    _ensure_step_counter(env)
    if not hasattr(env, "_recovery_ed_last_step"):
        env._recovery_ed_last_step = -1

    current_step = env.common_step_counter if hasattr(env, "common_step_counter") else 0
    if current_step != env._recovery_ed_last_step:
        env._recovery_step_count += 1
        env._recovery_step_count.clamp_(max=int(env.max_episode_length))
        env._recovery_ed_last_step = current_step


def _get_ed(env: ManagerBasedRLEnv, k: int = 3) -> torch.Tensor:
    """Episode-based Dynamic factor (paper Eq.1, normalized).

    ED(t) = (t / T)^k  ∈ [0, 1],  with k = 3.

    t = per-env step count in seconds, T = episode length in seconds.
    Normalizing to [0, 1] lets paper Table I weights (42/120/50) be used
    directly without absorbing a multiplicative constant; it also makes the
    gating factors `ED` and `1-ED` self-documenting (no ED_max division).

    Free-fall (t∈[0, 2s], T=5s): ED ≈ 0.064 → task rewards nearly zero.
    Exploration (t∈[2, 3.5s]): ED rises 0.064 → 0.34 → task rewards still
    weak, wheel-leg coord reward (×(1-ED)) dominates.
    Convergence (t∈[3.5, 5s]): ED rises 0.34 → 1.0 → task rewards dominate.
    """
    _ensure_step_counter(env)

    dt = _env_dt(env)
    t_sec = env._recovery_step_count.float() * dt
    T_sec = float(env.max_episode_length) * dt
    return (t_sec / T_sec).clamp_(0.0, 1.0) ** k


# Number of env.step() calls per PPO rollout iteration. MUST match
# `num_steps_per_env` in recovery_ppo_cfg.py — CW decay timing depends on
# this ratio. Isaac Lab does not expose the PPO rollout length to the reward
# manager, so we mirror the constant here.
RECOVERY_STEPS_PER_ITER = 48


def _get_cw(env: ManagerBasedRLEnv, beta: float = 0.3, decay: float = 0.968) -> float:
    """Curriculum Weight (paper Eq.3): CW(i) = beta · decay^i.

    i = training iteration, approximated as
        common_step_counter / RECOVERY_STEPS_PER_ITER.
    At beta=0.3, decay=0.968: CW=0.1 around iter 35, ~0.01 around iter 100.
    Behavior penalties are strong for a few dozen iterations, then fade as
    the policy stabilises and task/ED shaping takes over.
    """
    if hasattr(env, "common_step_counter"):
        iteration = env.common_step_counter / RECOVERY_STEPS_PER_ITER
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
    """Per-step binary reward for the 4-foot support state (paper §E).

    Paper (verbatim snippet): "a reward for the support state, defined as
    the condition where all four wheels are in contact with the ground
    simultaneously." → state-conditional, not event-triggered.

    Returns 1.0 while all four feet contact the ground, 0 otherwise.
    Suppressed during free-fall (the signal is meaningless while falling).
    """
    freefall = _is_freefall(env)

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, 0, :, :]
    magnitude = torch.norm(forces, dim=-1)
    if sensor_cfg.body_ids is None or sensor_cfg.body_ids == slice(None):
        raise RuntimeError(
            "recovery_support_state requires sensor_cfg.body_ids to be resolved "
            "from a body_names regex; no safe index fallback exists."
        )
    foot_forces = magnitude[:, sensor_cfg.body_ids]
    all_feet = (foot_forces > threshold).all(dim=1) & (~freefall)
    return all_feet.float()




# ── Behavior Rewards (×CW, zero during free-fall) ──

def recovery_body_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    force_clip: float = 100.0,
) -> torch.Tensor:
    """Body collision penalty on thigh / calf / base (paper Table I).

    r = CW · sum_b ( clip(||λ_b||, 0, force_clip)² )

    The paper's formula is r = sum(||λ_b||²); we clip the per-body force
    magnitude at `force_clip` Newtons before squaring so high-impact contacts
    on reset (||F|| often >1000 N) do not produce ~1e6-magnitude spikes that
    would drown the gradient early in training. Scale −5e-2 matches Table I.
    """
    if _is_freefall(env).all():
        return torch.zeros(env.num_envs, device=env.device)
    if sensor_cfg.body_ids is None or sensor_cfg.body_ids == slice(None):
        raise RuntimeError(
            "recovery_body_collision requires sensor_cfg.body_ids to be resolved "
            "from a body_names regex; no safe index fallback exists."
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
    """CW · sum((a_leg[t] - a_leg[t-1])²) on leg action dims only.

    Assumes the action manager layout matches the joint layout (first N
    action dims correspond to leg joints). Thunder's action cfg is
    joint_pos(legs) + joint_vel(wheels) concatenated in that order, which
    makes `leg_ids` from _get_joint_split a valid slice into the action
    tensor.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    leg_ids, _ = _get_joint_split(env, asset)
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    leg_diff = torch.sum(torch.square(action[:, leg_ids] - prev_action[:, leg_ids]), dim=1)
    return _get_cw(env) * leg_diff


# ── Constant Penalties (legs only — wheels have dedicated terms) ──

def recovery_joint_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """sum(q_dot²) over LEG joints only (paper Table I).

    Wheels are excluded — they have their own ED-gated penalty (recovery_
    wheel_velocity) and a positive wheel-leg coordination reward during the
    exploration phase. Penalising wheel velocity flat here would double-count
    and fight the paper's wheel-leg synergy contribution.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    leg_ids, _ = _get_joint_split(env, asset)
    return torch.sum(torch.square(asset.data.joint_vel[:, leg_ids]), dim=1)


def recovery_torques(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """sum(tau²) over LEG joints only (paper Table I)."""
    asset: Articulation = env.scene[asset_cfg.name]
    leg_ids, _ = _get_joint_split(env, asset)
    return torch.sum(torch.square(asset.data.applied_torque[:, leg_ids]), dim=1)


def recovery_joint_acceleration(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """sum(q_ddot²) over LEG joints only (paper Table I)."""
    asset: Articulation = env.scene[asset_cfg.name]
    leg_ids, _ = _get_joint_split(env, asset)
    return torch.sum(torch.square(asset.data.joint_acc[:, leg_ids]), dim=1)


def recovery_wheel_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """sum(wheel_vel²) gated by ED ∈ [0, 1].

    Early phase (ED≈0): wheels are free → policy can spin them for wheel-leg
    coordinated flipping (paper's core contribution: -15.8% to -26.2% joint
    torque). Late phase (ED→1): full penalty → converge to still stance.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    _, wheel_ids = _get_joint_split(env, asset)
    penalty = torch.sum(torch.square(asset.data.joint_vel[:, wheel_ids]), dim=1)
    return _get_ed(env) * penalty


def recovery_wheel_leg_coord(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_wheel_speed: float = 40.0,
) -> torch.Tensor:
    """Wheel-leg coordination reward (paper core contribution).

    Rewards actively spinning wheels while the body is tilted, which is the
    mechanism the paper credits for 15.8-26.2% joint torque reduction: wheels
    push against the ground to assist flipping instead of relying only on legs.

    r = (1 - ED) · (|ω_wheel| / max_wheel_speed) · ||g_xy||

    - (1 - ED) ∈ [1, 0]: active in exploration, decays to 0 in convergence.
    - |ω_wheel| clipped to max_wheel_speed: sum of absolute wheel speeds.
    - ||g_xy||: tilt factor, 0 when upright, ~1 when sideways/upside-down.

    Zero when upright (regardless of wheel motion) so it does not interfere
    with the task/ED convergence phase.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    _, wheel_ids = _get_joint_split(env, asset)

    early_gate = 1.0 - _get_ed(env)

    wheel_speed = torch.sum(torch.abs(asset.data.joint_vel[:, wheel_ids]), dim=1)
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
    leg_joint_pos_noise: float = 0.3,
):
    """Paper Section III-A reset: random orientation, 1.1 m drop, random
    initial joint angles.

    The paper's verbatim wording is "randomly initializing the robot's
    base orientation and joint angles" — so both orientation and joint
    pose are randomised at spawn. Combined with zero_action_freefall's
    floppy-joint free-fall, the 2 s fall then produces the diverse fallen
    states the recovery policy trains against.

    Args:
      drop_height: spawn z in metres (paper: 1.1).
      leg_joint_pos_noise: leg joint angle noise, Uniform(±noise) rad.
        Wheels keep their default pose.
    """
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

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    leg_ids, _ = _get_joint_split(env, asset)
    # Perturb only leg joints for pose diversity; wheels start at default.
    noise = (torch.rand_like(joint_pos[:, leg_ids]) * 2.0 - 1.0) * leg_joint_pos_noise
    joint_pos[:, leg_ids] = joint_pos[:, leg_ids] + noise
    asset.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)

    _ensure_step_counter(env)
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
    """Paper success criteria (bool per env)."""
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
    """Per-env success indicator (0/1 float), active only in the final second
    of the episode so tensorboard can plot a real end-of-episode success rate
    instead of a mid-episode partial-credit signal.

    Wired as a zero-weight reward term — its return value appears in the
    episode mean reward log as `recovery_success_rate`.
    """
    success = check_recovery_success(
        env, asset_cfg, height_threshold, joint_threshold, vel_threshold, ori_threshold
    ).float()
    _ensure_step_counter(env)
    last_second = (env._recovery_step_count >= (env.max_episode_length - int(1.0 / _env_dt(env)))).float()
    return success * last_second


# ── Free-fall Action Override ──

def _cache_actuator_gains(env: ManagerBasedRLEnv, asset: Articulation) -> bool:
    """Validate that each actuator exposes per-env (num_envs, num_joints)
    stiffness/damping tensors we can mutate in-place. Caches originals and
    returns True on success; returns False if the actuator class does not
    meet the contract (caller must fall back to the teleport path).
    """
    if hasattr(env, "_recovery_actuator_gain_cache"):
        return env._recovery_actuator_gain_cache is not None

    actuators = getattr(asset, "actuators", None)
    if not actuators:
        env._recovery_actuator_gain_cache = None
        return False

    cache = {}
    expected_shape = (env.num_envs,)
    for name, actuator in actuators.items():
        stiffness = getattr(actuator, "stiffness", None)
        damping = getattr(actuator, "damping", None)
        if not torch.is_tensor(stiffness) or not torch.is_tensor(damping):
            env._recovery_actuator_gain_cache = None
            return False
        if stiffness.ndim != 2 or stiffness.shape[0] != env.num_envs:
            env._recovery_actuator_gain_cache = None
            return False
        cache[name] = (stiffness.clone(), damping.clone())
    env._recovery_actuator_gain_cache = cache
    return True


def zero_action_freefall(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Paper Section III-A: first 2 s of each episode = "joint torques zero".

    Primary path (preferred): zero each actuator's per-env stiffness and
    damping tensors for envs still in free-fall and pin their PD position
    target to the current joint_pos — the PD controller then produces ≈0
    torque so joints are floppy and legs swing freely under gravity,
    matching the paper's fallen-state distribution.

    Fallback path (rigid-at-default teleport): used if the actuator class
    does not expose mutable (num_envs, num_joints) stiffness / damping
    tensors (e.g. IdealPDActuator with scalar gains). Less diverse but
    safe.

    When an env leaves free-fall, the cached original gains are restored.

    Called as an interval event every control step. If Isaac Lab passes a
    specific `env_ids` subset, we restrict our action to that subset.
    """
    _ensure_step_counter(env)
    asset: Articulation = env.scene[asset_cfg.name]

    freefall_mask = env._recovery_step_count < FREEFALL_STEPS
    # Respect the event manager's env_ids subset if given.
    if env_ids is not None and not isinstance(env_ids, slice):
        subset_mask = torch.zeros_like(freefall_mask)
        subset_mask[env_ids] = True
        freefall_mask = freefall_mask & subset_mask

    gains_ok = _cache_actuator_gains(env, asset)

    if gains_ok:
        freefall_idx = torch.where(freefall_mask)[0]
        # Running envs to restore: union of (subset, not-freefall). If env_ids
        # given, restore only within that subset so we don't touch envs the
        # manager did not request this tick.
        if env_ids is not None and not isinstance(env_ids, slice):
            running_mask = (~(env._recovery_step_count < FREEFALL_STEPS)) & subset_mask
        else:
            running_mask = ~(env._recovery_step_count < FREEFALL_STEPS)
        running_idx = torch.where(running_mask)[0]

        for name, actuator in asset.actuators.items():
            stiffness_ref, damping_ref = env._recovery_actuator_gain_cache[name]
            if len(freefall_idx) > 0:
                actuator.stiffness[freefall_idx] = 0.0
                actuator.damping[freefall_idx] = 0.0
            if len(running_idx) > 0:
                actuator.stiffness[running_idx] = stiffness_ref[running_idx]
                actuator.damping[running_idx] = damping_ref[running_idx]

        if len(freefall_idx) > 0:
            current_pos = asset.data.joint_pos[freefall_idx]
            asset.set_joint_position_target(current_pos, env_ids=freefall_idx)
        return

    # Fallback: rigid-at-default teleport (previous implementation).
    freefall_idx = torch.where(freefall_mask)[0]
    if len(freefall_idx) == 0:
        return
    joint_pos = asset.data.default_joint_pos[freefall_idx]
    joint_vel = torch.zeros_like(joint_pos)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=freefall_idx)


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
    """Per-body contact-force magnitude on shanks/thighs/base (N, num_bodies).

    Tells the critic whether the robot is dragging limbs or hitting the ground
    with its body, which the actor cannot directly sense.
    """
    if sensor_cfg.body_ids is None or sensor_cfg.body_ids == slice(None):
        raise RuntimeError(
            "priv_body_contact_force requires sensor_cfg.body_ids to be resolved "
            "from a body_names regex; no safe index fallback exists."
        )
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, 0, :, :]
    return torch.norm(forces, dim=-1)[:, sensor_cfg.body_ids]



