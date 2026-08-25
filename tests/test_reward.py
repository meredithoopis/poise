import torch

from physics_ot.ot.temporal_ot import TemporalPhysicsOT
from physics_ot.rewards import RewardWeights, compose_reward, task_reward


def test_task_reward_negative_of_error_and_success_flag():
    theta = torch.tensor([0.0, 0.5, 1.0])
    goal = torch.tensor([1.0, 1.0, 1.0])
    reward, success = task_reward(theta, goal, success_threshold=0.1)
    assert torch.allclose(reward, torch.tensor([-1.0, -0.5, 0.0]))
    assert torch.equal(success, torch.tensor([False, False, True]))


def test_compose_reward_matches_manual_weighted_sum():
    task = torch.tensor([1.0, -2.0])
    success = torch.tensor([True, False])
    ot_shaping = torch.tensor([0.3, -0.1])
    action_penalty = torch.tensor([0.2, 0.4])
    weights = RewardWeights(task=1.0, physics_ot=0.5, action=0.001, success_bonus=10.0)

    total = compose_reward(task, success, ot_shaping, action_penalty, weights)
    expected = (
        weights.task * task
        + weights.physics_ot * ot_shaping
        - weights.action * action_penalty
        + weights.success_bonus * success.float()
    )
    assert torch.allclose(total, expected)


def test_delta_ot_reward_is_positive_when_alignment_improves():
    t = torch.linspace(0.0, 1.0, 64).unsqueeze(0)
    velocity = torch.ones_like(t)
    wrench = torch.zeros_like(t)
    pot = TemporalPhysicsOT(epsilon=0.01, num_iters=100)

    d_far = pot.distance(t, velocity, wrench, t, velocity, wrench + 5.0)
    d_close = pot.distance(t, velocity, wrench, t, velocity, wrench + 0.1)

    reward = TemporalPhysicsOT.delta_reward(d_far, d_close)
    assert reward.item() > 0.0
