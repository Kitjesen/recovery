# Copyright (c) 2026 Qiongpei Technology
# SPDX-License-Identifier: Apache-2.0

"""Event functions for the recovery MDP.

Two paper-driven events live here:

- `reset_with_freefall` — paper §III-A initialisation: random orientation,
  1.1 m drop, random joint angles (clamped to soft limits), zero velocity.
  Also zeros the step counter and the action manager's `prev_action` so
  the first post-reset step is clean.

- `zero_action_freefall` — enforces "joint torques = 0" during the first
  2 s of every episode by zeroing per-env actuator stiffness/damping and
  pinning the PD target to the current joint_pos. Falls back to a
  rigid-at-default teleport if the actuator class does not expose
  mutable per-env gain tensors.

Both are wired into the env cfg as EventTerms. `zero_action_freefall` is
bound twice: once as mode="reset" so torques are already zero on the very
first physics step, and once as mode="interval" for each subsequent step
until the env exits free-fall.
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from ._utils import FREEFALL_STEPS, _ensure_step_counter, _get_joint_split

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ── Reset ──

def reset_with_freefall(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    drop_height: float = 1.1,
    leg_joint_pos_noise: float = 0.3,
):
    """Paper §III-A reset: random orientation, 1.1 m drop, random joint angles.

    Paper verbatim: "randomly initializing the robot's base orientation
    and joint angles, setting the joint torques to zero, and letting the
    robot free-fall from a height of 1.1 m for 2 seconds." We do the
    first two here; `zero_action_freefall` handles "torques to zero".

    Args:
      drop_height: spawn z in metres (paper: 1.1).
      leg_joint_pos_noise: leg joint angle noise, Uniform(±noise) rad.
        Wheels keep their default pose. The noise is clamped to soft
        joint limits so PhysX does not clamp at t=1 and emit an impulse.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if len(env_ids) == 0:
        return

    # Uniform random orientation on SO(3) via unit quaternion.
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
    noise = (torch.rand_like(joint_pos[:, leg_ids]) * 2.0 - 1.0) * leg_joint_pos_noise
    joint_pos[:, leg_ids] = joint_pos[:, leg_ids] + noise
    # Clamp to soft limits so PhysX doesn't impulse-clamp on step 1.
    soft_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clamp(joint_pos, min=soft_limits[..., 0], max=soft_limits[..., 1])
    asset.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)

    # Reset to -1 so first _advance_step_counter brings count to 0.
    _ensure_step_counter(env)
    env._recovery_step_count[env_ids] = -1

    # Zero the action manager's prev_action for these envs — otherwise
    # recovery_action_rate_legs on the first post-reset step would compute
    # (fresh_action − last_action_of_previous_episode)² and log a
    # systematic spike at every episode boundary.
    if hasattr(env, "action_manager"):
        prev_action = getattr(env.action_manager, "prev_action", None)
        if torch.is_tensor(prev_action):
            prev_action[env_ids] = 0.0

    # Reset the `previous_*` observation caches for these envs so the
    # first post-reset step's "previous" frame is the fresh fallen pose
    # rather than the last step of the previous episode.
    _, wheel_ids = _get_joint_split(env, asset)
    leg_pos_now = asset.data.joint_pos[env_ids][:, leg_ids]
    leg_vel_now = asset.data.joint_vel[env_ids][:, leg_ids]
    wheel_vel_now = asset.data.joint_vel[env_ids][:, wheel_ids]
    for key, current in (
        ("joint_pos_legs", leg_pos_now),
        ("joint_vel_legs", leg_vel_now),
        ("wheel_vel", wheel_vel_now),
    ):
        attr = f"_recovery_prev_{key}"
        if hasattr(env, attr):
            getattr(env, attr)[env_ids] = current


# ── Free-fall torque override ──

def _cache_actuator_gains(env: ManagerBasedRLEnv, asset: Articulation) -> bool:
    """Validate the actuator API and cache original stiffness/damping.

    Returns True if every actuator exposes (num_envs, num_joints) mutable
    tensors for `stiffness` and `damping`; returns False (caller should
    fall back to rigid-at-default teleport) otherwise. Cache is populated
    on the first successful call and reused; if the API check fails, the
    cache is set to None and subsequent calls also fail-over.
    """
    if hasattr(env, "_recovery_actuator_gain_cache"):
        return env._recovery_actuator_gain_cache is not None

    actuators = getattr(asset, "actuators", None)
    if not actuators:
        env._recovery_actuator_gain_cache = None
        return False

    cache = {}
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
    """Paper §III-A: enforce joint torques = 0 during the free-fall window.

    Primary path (preferred): for each env with step_count < FREEFALL_STEPS,
    zero the per-env actuator stiffness/damping and pin the PD target to
    the current joint_pos. The PD controller then produces ≈0 torque so
    joints are floppy and legs swing freely under gravity — matching the
    paper's fallen-state distribution. Cached original gains are restored
    when the env exits free-fall.

    Fallback path: rigid-at-default teleport (write `(default_pos, 0 vel)`
    every step). Used when the actuator class does not expose mutable
    (num_envs, num_joints) gain tensors.

    Bound twice in env cfg:
      - mode="reset":     zero gains at t=0 so the first physics step has
                          no PD impulse.
      - mode="interval": every control step to keep gains zero during the
                          window and restore them when envs exit.

    If Isaac Lab passes a specific `env_ids` subset, we restrict to it.
    """
    _ensure_step_counter(env)
    asset: Articulation = env.scene[asset_cfg.name]

    freefall_mask = env._recovery_step_count < FREEFALL_STEPS
    if env_ids is not None and not isinstance(env_ids, slice):
        subset_mask = torch.zeros_like(freefall_mask)
        subset_mask[env_ids] = True
        freefall_mask = freefall_mask & subset_mask

    gains_ok = _cache_actuator_gains(env, asset)

    if gains_ok:
        freefall_idx = torch.where(freefall_mask)[0]
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

    # Fallback: rigid-at-default teleport.
    freefall_idx = torch.where(freefall_mask)[0]
    if len(freefall_idx) == 0:
        return
    joint_pos = asset.data.default_joint_pos[freefall_idx]
    joint_vel = torch.zeros_like(joint_pos)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=freefall_idx)
