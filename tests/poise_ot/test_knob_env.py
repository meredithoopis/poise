import torch

from physics_ot.dynamics.knob import KnobParams
from physics_ot.rewards import task_reward
from poise_ot.knob_env import KnobEnv, KnobEnvConfig


def make_cfg(**overrides) -> KnobEnvConfig:
    params = KnobParams(inertia=0.002, damping=0.02, stiffness=0.10, theta0=0.0, coulomb_torque=0.04, v_eps=0.05)
    defaults = dict(
        params=params,
        theta_goal=1.57,
        success_threshold=0.05,
        physics_dt=1.0 / 120.0,
        control_decimation=4,
        episode_seconds=1.0,
        tau_max=1.0,
    )
    defaults.update(overrides)
    return KnobEnvConfig(**defaults)


def test_reset_and_step_shapes():
    env = KnobEnv(make_cfg(), num_envs=4)
    obs, info = env.reset()
    assert obs.shape == (4, 3)
    assert info["theta"].shape == (4,)

    action = torch.zeros(4, 1)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (4, 3)
    assert reward.shape == (4,)
    assert terminated.shape == (4,)
    assert truncated.shape == (4,)
    assert torch.isfinite(obs).all()
    assert torch.isfinite(reward).all()


def test_zero_torque_relaxes_toward_theta0():
    """With no applied torque, the spring term should pull theta back toward
    theta0 if started away from it (passive_torque's sign convention)."""
    cfg = make_cfg(episode_seconds=2.0)
    env = KnobEnv(cfg, num_envs=1)
    env.reset()
    env.theta[:] = 0.5  # start away from theta0=0.0, no torque applied

    for _ in range(50):
        env.step(torch.zeros(1, 1))

    assert env.theta.item() < 0.5  # pulled back toward theta0


def test_reward_matches_task_reward_formula_on_zero_action():
    cfg = make_cfg()
    env = KnobEnv(cfg, num_envs=2)
    env.reset()
    _, reward, _, _, info = env.step(torch.zeros(2, 1))

    theta_goal_t = torch.full((2,), cfg.theta_goal)
    neg_error, is_success = task_reward(info["theta"], theta_goal_t, cfg.success_threshold)
    expected = neg_error + cfg.success_bonus * is_success.float() - cfg.action_weight * torch.zeros(2)
    assert torch.allclose(reward, expected, atol=1e-5)


def test_truncated_after_episode_length():
    cfg = make_cfg(episode_seconds=0.1, physics_dt=1.0 / 120.0, control_decimation=4)
    env = KnobEnv(cfg, num_envs=1)
    env.reset()
    max_steps = env._max_episode_steps
    for _ in range(max_steps - 1):
        _, _, _, truncated, _ = env.step(torch.zeros(1, 1))
        assert not truncated.item()
    _, _, _, truncated, _ = env.step(torch.zeros(1, 1))
    assert truncated.item()


def test_reset_only_clears_selected_envs():
    env = KnobEnv(make_cfg(), num_envs=3)
    env.reset()
    for _ in range(20):
        env.step(torch.ones(3, 1) * 0.5)
    theta_before = env.theta.clone()

    env.reset(env_ids=torch.tensor([1]))
    assert env.theta[1].item() == 0.0
    assert env.theta[0].item() == theta_before[0].item()
    assert env.theta[2].item() == theta_before[2].item()
