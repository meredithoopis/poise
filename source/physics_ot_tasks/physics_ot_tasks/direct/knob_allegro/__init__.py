import gymnasium as gym

from . import agents

gym.register(
    id="PhysicsOT-Knob-Allegro-Direct-v0",
    entry_point=f"{__name__}.knob_allegro_env:KnobAllegroEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.knob_allegro_env_cfg:KnobAllegroEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KnobAllegroPPORunnerCfg",
    },
)
