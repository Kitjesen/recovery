# Copyright (c) 2026 Qiongpei Technology
# SPDX-License-Identifier: Apache-2.0

"""Thunder fall recovery RL task — standalone Isaac Lab package.

Based on 'Learning to Recover: Dynamic Reward Shaping with Wheel-Leg
Coordination for Fallen Robots' (arXiv:2506.05516).

Registers a single gym task on import:
    RobotLab-Isaac-Velocity-Recovery-Thunder-v0

Dependencies: only isaaclab / isaaclab_rl / rsl-rl-lib (no robot_lab).
"""

import gymnasium as gym

from . import config  # noqa: F401 — trigger gym.register

gym.register(
    id="RobotLab-Isaac-Velocity-Recovery-Thunder-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "thunder_recovery.config.recovery_env_cfg:ThunderRecoveryEnvCfg",
        "rsl_rl_cfg_entry_point": "thunder_recovery.config.recovery_ppo_cfg:RecoveryPPORunnerCfg",
    },
)

__version__ = "0.1.0"
__all__ = ["__version__"]
