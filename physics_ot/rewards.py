from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RewardWeights:
    """Reward composition weights (section 13)."""

    task: float = 1.0
    physics_ot: float = 0.5
    action: float = 0.001
    success_bonus: float = 10.0


def task_reward(theta: torch.Tensor, theta_goal: torch.Tensor, success_threshold: float) -> torch.Tensor:
    """r_task = -|theta - theta_goal| + R_success * 1[|theta - theta_goal| < eps_theta] (section 13)."""
    error = (theta - theta_goal).abs()
    return -error, error < success_threshold


def compose_reward(
    task_reward_value: torch.Tensor,
    is_success: torch.Tensor,
    ot_shaping_reward: torch.Tensor,
    action_penalty: torch.Tensor,
    weights: RewardWeights | None = None,
) -> torch.Tensor:
    """r_t = lambda_task * r_task + lambda_OT * r_OT - lambda_a * r_action (+ success bonus) (section 13)."""
    weights = weights or RewardWeights()
    bonus = weights.success_bonus * is_success.float()
    return (
        weights.task * task_reward_value
        + weights.physics_ot * ot_shaping_reward
        - weights.action * action_penalty
        + bonus
    )
