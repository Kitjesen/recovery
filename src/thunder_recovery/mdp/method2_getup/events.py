# Copyright (c) 2026 Qiongpei Technology
# SPDX-License-Identifier: Apache-2.0

"""Getup (Method 2) reset event — mujoco_playground-style 60/40 drop.

Reference:
  github.com/google-deepmind/mujoco_playground
  → mujoco_playground/_src/locomotion/go1/getup.py::reset

Per-env branch on a Bernoulli draw:
  * with prob `drop_prob` (default 0.6): spawn at `drop_height` with a
    uniformly-random quaternion on SO(3) and random joint angles sampled
    uniformly inside the per-joint soft limits.
  * otherwise: start from the nominal default pose.

Root velocity is sampled uniform[-0.5, 0.5] on all 6 dof for every env.

There is NO free-fall torque-zeroing here (unlike Method 1): the policy
runs from t=0 with nominal PD gains. mujoco_playground relies on a short
physics-only settle before the policy starts; Isaac Lab's reset pipeline
does not expose a clean hook for that, so we skip it — the policy learns
to handle the first few steps' unsettled dynamics.
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_getup(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    drop_prob: float = 0.6,
    drop_height: float = 0.5,
    root_vel_range: float = 0.5,
):
    """60/40 drop reset.

    Args:
      drop_prob: Bernoulli prob. of choosing the dropped branch.
      drop_height: spawn z in metres when dropped (mujoco_playground: 0.5).
      root_vel_range: uniform[-r, +r] on all 6 root dof.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if len(env_ids) == 0:
        return
    n = len(env_ids)
    device = env.device

    drop_mask = torch.rand(n, device=device) < drop_prob

    root_state = asset.data.default_root_state[env_ids].clone()
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)

    # Root velocity perturbation for every env (matches mujoco_playground).
    root_state[:, 7:13] = (
        (torch.rand((n, 6), device=device) * 2.0 - 1.0) * root_vel_range
    )

    drop_idx = torch.where(drop_mask)[0]
    if drop_idx.numel() > 0:
        # Lift + random quaternion on SO(3) (Shoemake, 1992).
        root_state[drop_idx, 2] = drop_height
        u1 = torch.rand(drop_idx.numel(), device=device)
        u2 = torch.rand(drop_idx.numel(), device=device) * 2.0 * math.pi
        u3 = torch.rand(drop_idx.numel(), device=device) * 2.0 * math.pi
        qw = torch.sqrt(1.0 - u1) * torch.sin(u2)
        qx = torch.sqrt(1.0 - u1) * torch.cos(u2)
        qy = torch.sqrt(u1) * torch.sin(u3)
        qz = torch.sqrt(u1) * torch.cos(u3)
        root_state[drop_idx, 3:7] = torch.stack([qw, qx, qy, qz], dim=1)

        # Joint angles uniform inside per-joint soft limits.
        soft = asset.data.soft_joint_pos_limits[env_ids][drop_idx]
        rand = torch.rand_like(joint_pos[drop_idx])
        joint_pos[drop_idx] = soft[..., 0] + rand * (soft[..., 1] - soft[..., 0])

    asset.write_root_state_to_sim(root_state, env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # Clear action-rate state so the first post-reset step doesn't compute
    # (fresh_action − last_action_of_previous_episode)² and spike the cost.
    if hasattr(env, "action_manager"):
        pa = getattr(env.action_manager, "prev_action", None)
        if torch.is_tensor(pa):
            pa[env_ids] = 0.0
    if hasattr(env, "_getup_prev_prev_action"):
        env._getup_prev_prev_action[env_ids] = 0.0
