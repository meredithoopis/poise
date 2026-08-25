from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class KnobAllegroPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # section 18
    num_steps_per_env = 32
    max_iterations = 10000
    save_interval = 250
    experiment_name = "physics_ot_knob_allegro"
    empirical_normalization = False
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.016,
        max_grad_norm=1.0,
    )
