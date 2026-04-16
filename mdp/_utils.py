# Copyright (c) 2026 Qiongpei Technology
# SPDX-License-Identifier: Apache-2.0

"""Shared infrastructure for the recovery MDP (ED/CW shaping, step counter,
joint index split, dt resolution).

This module is private (leading underscore) — public re-exports of the
helpers users need (`RECOVERY_STEPS_PER_ITER`, `check_recovery_success`,
etc.) live in `mdp/__init__.py`.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ── Constants ──

FREEFALL_STEPS = 100
"""Duration of the torques=0 free-fall window in control steps (2 s @ 50 Hz)."""

RECOVERY_STEPS_PER_ITER = 48
"""Number of env.step() calls per PPO rollout iteration.

MUST match `num_steps_per_env` in recovery_ppo_cfg.py — CW decay timing
depends on the ratio. Isaac Lab does not expose the PPO rollout length to
the reward manager, so we mirror the constant here.
"""


# ── Per-env state ──

def _ensure_step_counter(env: ManagerBasedRLEnv) -> None:
    """Create the per-env step counter lazily (int64, exact)."""
    if not hasattr(env, "_recovery_step_count"):
        env._recovery_step_count = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )


def _advance_step_counter(env: ManagerBasedRLEnv) -> None:
    """Increment the per-env step counter exactly once per env.step().

    Driven by the zero-effect `recovery_step_counter` reward term.
    `reset_with_freefall` sets the counter to -1 so the first post-reset
    advance brings it to 0 — reward terms read count ∈ {0, 1, …, T-1}
    over an episode, which makes `_is_freefall` (< FREEFALL_STEPS) cover
    exactly 100 steps and ED reach exactly 1.0 at the final step.
    """
    _ensure_step_counter(env)
    if not hasattr(env, "_recovery_ed_last_step"):
        env._recovery_ed_last_step = -1

    current_step = env.common_step_counter if hasattr(env, "common_step_counter") else 0
    if current_step != env._recovery_ed_last_step:
        env._recovery_step_count += 1
        env._recovery_step_count.clamp_(max=int(env.max_episode_length) - 1)
        env._recovery_ed_last_step = current_step


def _is_freefall(env: ManagerBasedRLEnv) -> torch.Tensor:
    """True for envs still in the free-fall phase (step_count < FREEFALL_STEPS)."""
    _ensure_step_counter(env)
    return env._recovery_step_count < FREEFALL_STEPS


# ── Joint layout resolution ──

def _get_joint_split(
    env: ManagerBasedRLEnv,
    asset: Articulation,
    wheel_joint_regex: str = ".*foot.*",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (leg_ids, wheel_ids) as long tensors, cached on the env.

    On Thunder the wheel joints are named `<LEG>_foot_joint` (the foot
    link is rigidly attached to the wheel, so the joint driving it carries
    the `foot` suffix) — hence the default regex matches `.*foot.*`. Pass
    a different regex if your URDF uses a different convention (e.g.
    `.*wheel.*` for some Go2-W variants).

    Everything not matched by `wheel_joint_regex` is treated as a leg
    joint. Raises if either group is empty.
    """
    if not hasattr(env, "_recovery_joint_split"):
        wheel_ids, wheel_names = asset.find_joints(wheel_joint_regex)
        all_ids = list(range(asset.data.joint_pos.shape[1]))
        leg_ids = [i for i in all_ids if i not in wheel_ids]
        if not wheel_ids or not leg_ids:
            raise RuntimeError(
                f"recovery: joint split failed with regex '{wheel_joint_regex}'. "
                f"Got wheel_ids={wheel_ids} (names={wheel_names}), leg_ids={leg_ids}. "
                f"Asset joint_names={asset.data.joint_names}."
            )
        device = asset.data.joint_pos.device
        env._recovery_joint_split = (
            torch.tensor(leg_ids, dtype=torch.long, device=device),
            torch.tensor(wheel_ids, dtype=torch.long, device=device),
        )
    return env._recovery_joint_split


# ── Time resolution ──

def _env_dt(env: ManagerBasedRLEnv) -> float:
    """Control-step duration in seconds. Uses env.step_dt when available."""
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


# ── ED / CW (paper Eq. 1 / Eq. 3) ──

def _get_ed(env: ManagerBasedRLEnv, k: int = 3) -> torch.Tensor:
    """Episode-based Dynamic factor (paper Eq. 1, normalised).

    ED(t) = (t / T)^k ∈ [0, 1], k = 3.
    t = per-env step count in seconds, T = episode length in seconds.

    Normalising to [0, 1] lets paper Table I weights apply directly and
    makes the gating factors `ED` and `1-ED` self-documenting.
    """
    _ensure_step_counter(env)
    dt = _env_dt(env)
    t_sec = env._recovery_step_count.float() * dt
    T_sec = float(env.max_episode_length) * dt
    return (t_sec / T_sec).clamp_(0.0, 1.0) ** k


def _get_cw(env: ManagerBasedRLEnv, beta: float = 0.3, decay: float = 0.968) -> float:
    """Curriculum Weight (paper Eq. 3): CW(i) = β · decay^i.

    i = training iteration, approximated as
        common_step_counter / RECOVERY_STEPS_PER_ITER.
    At β=0.3, decay=0.968: CW=0.1 around iter 35, ~0.01 around iter 100.
    Behavior penalties are strong for a few dozen iterations, then fade as
    the policy stabilises and task/ED shaping takes over.
    """
    if hasattr(env, "common_step_counter"):
        iteration = env.common_step_counter / RECOVERY_STEPS_PER_ITER
    else:
        iteration = 0
    return beta * (decay ** iteration)
