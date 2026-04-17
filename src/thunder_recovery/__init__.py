# Copyright (c) 2026 Qiongpei Technology
# SPDX-License-Identifier: Apache-2.0

"""Thunder fall recovery RL task — standalone Isaac Lab package.

Two methods are registered:

  * `RobotLab-Isaac-Velocity-Recovery-Thunder-v0`
    Method 1 — Deng et al., 'Learning to Recover: Dynamic Reward Shaping
    with Wheel-Leg Coordination for Fallen Robots' (arXiv:2506.05516).
    13 reward terms with ED/CW time-varying weights, 2 s free-fall.

  * `RobotLab-Isaac-Velocity-Recovery-Thunder-Getup-v0`
    Method 2 — mujoco_playground Go1 getup style
    (github.com/google-deepmind/mujoco_playground → go1/getup.py).
    9 plain-weight terms with upright-∧-at-height gating, 60/40 drop.

Both share the Scene / Actions / Observations / PPO cfg for direct
tensorboard comparability.

Dependencies: only isaaclab / isaaclab_rl / rsl-rl-lib (no robot_lab).
"""

import gymnasium as gym

from . import config  # noqa: F401 — trigger gym.register

gym.register(
    id="RobotLab-Isaac-Velocity-Recovery-Thunder-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "thunder_recovery.config.method1_deng.env_cfg:ThunderRecoveryEnvCfg",
        "rsl_rl_cfg_entry_point": "thunder_recovery.config.recovery_ppo_cfg:RecoveryPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Recovery-Thunder-Getup-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "thunder_recovery.config.method2_getup.env_cfg:ThunderRecoveryGetupEnvCfg",
        "rsl_rl_cfg_entry_point": "thunder_recovery.config.recovery_ppo_cfg:RecoveryPPORunnerCfg",
    },
)

__version__ = "0.1.0"
__all__ = ["__version__"]
