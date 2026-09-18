import torch

from poise_ot.two_critic_sac import GaussianPolicy, ReplayBuffer, SACConfig, TwoCriticSAC


def test_gaussian_policy_sample_shapes_and_bounded():
    policy = GaussianPolicy(obs_dim=5, action_dim=2)
    obs = torch.randn(7, 5)
    action, log_prob, mean_action = policy.sample(obs)
    assert action.shape == (7, 2)
    assert log_prob.shape == (7, 1)
    assert torch.all(action.abs() <= 1.0)
    assert torch.all(mean_action.abs() <= 1.0)
    assert torch.isfinite(log_prob).all()


def test_replay_buffer_add_and_sample_shapes():
    buf = ReplayBuffer(capacity=100, obs_dim=3, h_dim=10, action_dim=1)
    n = 16
    buf.add_batch(
        obs=torch.randn(n, 3),
        h=torch.randn(n, 10),
        action=torch.randn(n, 1),
        r_task=torch.randn(n),
        c_align=torch.rand(n),
        next_obs=torch.randn(n, 3),
        next_h=torch.randn(n, 10),
        done=torch.zeros(n),
    )
    assert len(buf) == n
    batch = buf.sample(8)
    assert batch["obs"].shape == (8, 3)
    assert batch["h"].shape == (8, 10)
    assert batch["action"].shape == (8, 1)
    assert batch["r_task"].shape == (8, 1)
    assert batch["c_align"].shape == (8, 1)


def test_replay_buffer_wraps_around_capacity():
    buf = ReplayBuffer(capacity=10, obs_dim=2, h_dim=2, action_dim=1)
    for _ in range(3):
        buf.add_batch(
            obs=torch.randn(4, 2),
            h=torch.randn(4, 2),
            action=torch.randn(4, 1),
            r_task=torch.randn(4),
            c_align=torch.rand(4),
            next_obs=torch.randn(4, 2),
            next_h=torch.randn(4, 2),
            done=torch.zeros(4),
        )
    assert len(buf) == 10  # capped, not 12


def _make_agent(obs_dim=4, h_dim=12, action_dim=2) -> TwoCriticSAC:
    cfg = SACConfig(obs_dim=obs_dim, h_dim=h_dim, action_dim=action_dim, hidden_dims=(32, 32))
    return TwoCriticSAC(cfg)


def _make_batch(batch_size, obs_dim, h_dim, action_dim) -> dict:
    return {
        "obs": torch.randn(batch_size, obs_dim),
        "h": torch.randn(batch_size, h_dim),
        "action": torch.rand(batch_size, action_dim) * 2 - 1,
        "r_task": torch.randn(batch_size, 1),
        "c_align": torch.rand(batch_size, 1),
        "next_obs": torch.randn(batch_size, obs_dim),
        "next_h": torch.randn(batch_size, h_dim),
        "done": torch.zeros(batch_size, 1),
    }


def test_update_runs_and_returns_finite_losses():
    agent = _make_agent()
    batch = _make_batch(32, 4, 12, 2)
    metrics = agent.update(batch)
    for key, value in metrics.items():
        assert value == value, f"{key} is NaN"  # NaN != NaN
        assert abs(value) < 1e6, f"{key} exploded: {value}"


def test_act_returns_bounded_actions():
    agent = _make_agent()
    obs = torch.randn(5, 4)
    action = agent.act(obs)
    assert action.shape == (5, 2)
    assert torch.all(action.abs() <= 1.0)


def test_critic_losses_trend_down_on_a_fixed_toy_batch():
    """Gradient sanity, as a fast unit test: repeatedly updating on a FIXED
    batch (a toy regression target both critics can actually fit) should
    drive both critic losses down, not diverge or stay flat."""
    torch.manual_seed(0)
    agent = _make_agent(obs_dim=4, h_dim=12, action_dim=2)
    batch = _make_batch(64, 4, 12, 2)

    losses_q_task = []
    losses_q_c = []
    for _ in range(300):
        metrics = agent.update(batch)
        losses_q_task.append(metrics["loss_q_task"])
        losses_q_c.append(metrics["loss_q_c"])

    early_task = sum(losses_q_task[:20]) / 20
    late_task = sum(losses_q_task[-20:]) / 20
    early_c = sum(losses_q_c[:20]) / 20
    late_c = sum(losses_q_c[-20:]) / 20

    assert late_task < early_task
    assert late_c < early_c
