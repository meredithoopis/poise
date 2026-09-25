import gymnasium as gym

gym.register(
    id="PoiseOT-Knob-Allegro-Direct-v0",
    entry_point=f"{__name__}.knob_allegro_poise_env:KnobAllegroPoiseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.knob_allegro_poise_env_cfg:KnobAllegroPoiseEnvCfg",
        # No rsl_rl_cfg_entry_point -- trained with poise_ot.two_critic_sac's
        # hand-rolled TwoCriticSAC (scripts/stage3/train_stage3.py), not rsl_rl.
    },
)
