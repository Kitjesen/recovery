# Copyright (c) 2026 Qiongpei Technology
# SPDX-License-Identifier: Apache-2.0

"""PPO runner config for the Thunder fall recovery policy.

Network architecture and init_noise_std match the author's public
deployment repository (`boyuandeng/Recovery_go2w/simulate_python/test/
runtest.py`), which defines the `Actor` class that loads the paper's
`model_7999.pt` / `model_4999.pt` checkpoint weights. The checkpoint can
only be loaded when the class structure is identical to the one used
during training — so these sizes are authoritative:

  Actor  : MLP [128, 128, 128] with ReLU,   78 → 16
  Critic : MLP [128, 128, 128] with ReLU,  103 →  1
  std    : learned, init 1.0
"""

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass


@configclass
class RecoveryPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Standard asymmetric-obs ActorCritic, sized per the paper's
    published checkpoint."""

    seed = 42
    num_steps_per_env = 48  # matches RECOVERY_STEPS_PER_ITER in mdp/_utils.py
    max_iterations = 10000
    save_interval = 200
    experiment_name = "thunder_recovery"
    run_name = ""
    logger = "tensorboard"
    resume = False

    class_name = "OnPolicyRunner"

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="relu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=0.001,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=10.0,
    )
