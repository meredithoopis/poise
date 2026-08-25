from __future__ import annotations

import torch

from .cost import CostWeights, normalize_state, physics_ot_cost
from .sinkhorn import batched_sinkhorn, transport_cost

"""Baseline distance/reward functions for the Experiment-0 2x2 ablation (sections 25-26).

                     Pointwise                 Optimal Transport
State only           pointwise_l2_distance      ot_state_distance
State + physics      pointwise_physics_distance physics_ot (TemporalPhysicsOT)

``dtw_distance`` is the extra DTW-Physics baseline (B2).
"""


def _resample_to_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Linearly resample the last dimension of ``x`` (shape [..., T]) to ``target_len``."""
    leading_shape = x.shape[:-1]
    flat = x.reshape(-1, 1, x.shape[-1])
    resampled = torch.nn.functional.interpolate(flat, size=target_len, mode="linear", align_corners=True)
    return resampled.reshape(*leading_shape, target_len)


def pointwise_l2_distance(
    progress_ref: torch.Tensor,
    velocity_ref: torch.Tensor,
    progress_rollout: torch.Tensor,
    velocity_rollout: torch.Tensor,
    sigma_v: float = 1.0,
) -> torch.Tensor:
    """B1: index-aligned pointwise L2 over state only (s, theta_dot)."""
    t_r = progress_rollout.shape[-1]
    s_ref = _resample_to_length(progress_ref, t_r)
    v_ref = _resample_to_length(velocity_ref, t_r) / sigma_v
    v_roll = velocity_rollout / sigma_v
    return ((s_ref - progress_rollout).pow(2) + (v_ref - v_roll).pow(2)).mean(dim=-1)


def pointwise_physics_distance(
    progress_ref: torch.Tensor,
    velocity_ref: torch.Tensor,
    wrench_ref: torch.Tensor,
    progress_rollout: torch.Tensor,
    velocity_rollout: torch.Tensor,
    wrench_rollout: torch.Tensor,
    sigma_v: float = 1.0,
    sigma_tau: float = 1.0,
) -> torch.Tensor:
    """B4: index-aligned pointwise L2 over state + physics (s, theta_dot, tau)."""
    t_r = progress_rollout.shape[-1]
    s_ref = _resample_to_length(progress_ref, t_r)
    v_ref = _resample_to_length(velocity_ref, t_r)
    tau_ref = _resample_to_length(wrench_ref, t_r)
    z_ref = normalize_state(s_ref, v_ref, tau_ref, sigma_v, sigma_tau)
    z_roll = normalize_state(progress_rollout, velocity_rollout, wrench_rollout, sigma_v, sigma_tau)
    return (z_ref - z_roll).pow(2).sum(dim=-1).mean(dim=-1)


def ot_state_distance(
    progress_ref: torch.Tensor,
    velocity_ref: torch.Tensor,
    progress_rollout: torch.Tensor,
    velocity_rollout: torch.Tensor,
    sigma_v: float = 1.0,
    epsilon: float = 0.05,
    num_iters: int = 20,
) -> torch.Tensor:
    """B3: temporal OT over state only (s, theta_dot); the wrench term is dropped."""
    zero_ref = torch.zeros_like(progress_ref)
    zero_roll = torch.zeros_like(progress_rollout)
    z_ref = normalize_state(progress_ref, velocity_ref, zero_ref, sigma_v, 1.0)
    z_roll = normalize_state(progress_rollout, velocity_rollout, zero_roll, sigma_v, 1.0)
    weights = CostWeights(progress=1.0, velocity=0.25, wrench=0.0, time=0.10)
    with torch.no_grad():
        cost = physics_ot_cost(z_ref, z_roll, weights)
        gamma = batched_sinkhorn(cost, epsilon=epsilon, num_iters=num_iters)
        return transport_cost(gamma, cost)


def dtw_distance(z_ref: torch.Tensor, z_rollout: torch.Tensor, weights: CostWeights | None = None) -> torch.Tensor:
    """B2: Dynamic Time Warping distance over the full physical trajectory (s, v_bar, tau_bar).

    Args:
        z_ref: normalized reference states, shape [B, T_H, 3].
        z_rollout: normalized rollout states, shape [B, T_R, 3].
        weights: state-term weights (the DTW alignment itself has no time-index term).

    Returns:
        Normalized DTW cost, shape [B].

    Note:
        Implemented as a sequential dynamic program over T_H + T_R - 1
        anti-diagonals, batched across the leading batch dimension. This is
        adequate for periodic evaluation / the Experiment-0 ablation, but is
        the slowest of the baselines and not intended as a dense per-step
        training reward at large ``num_envs``.
    """
    weights = weights or CostWeights()
    state_weights = z_ref.new_tensor([weights.progress, weights.velocity, weights.wrench])
    diff = z_ref.unsqueeze(-2) - z_rollout.unsqueeze(-3)  # [B, T_H, T_R, 3]
    cost = (diff.pow(2) * state_weights).sum(dim=-1)  # [B, T_H, T_R]

    batch, t_h, t_r = cost.shape
    dp = cost.new_full((batch, t_h, t_r), float("inf"))
    for i in range(t_h):
        for j in range(t_r):
            c = cost[:, i, j]
            if i == 0 and j == 0:
                dp[:, i, j] = c
            elif i == 0:
                dp[:, i, j] = dp[:, i, j - 1] + c
            elif j == 0:
                dp[:, i, j] = dp[:, i - 1, j] + c
            else:
                best_prev = torch.minimum(torch.minimum(dp[:, i - 1, j], dp[:, i, j - 1]), dp[:, i - 1, j - 1])
                dp[:, i, j] = best_prev + c
    return dp[:, t_h - 1, t_r - 1] / (t_h + t_r)
