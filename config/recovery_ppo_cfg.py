# Recovery PPO config — simple MLP, no HIM
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass


@configclass
class RecoveryPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Recovery: standard MLP actor-critic, single frame obs."""
    seed = 42
    num_steps_per_env = 48  # shorter rollout for 5s episodes
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
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=0.001,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=10.0,
    )
