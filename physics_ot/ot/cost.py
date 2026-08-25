from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CostWeights:
    """Weights for the Physics-OT ground cost (section 9)."""

    progress: float = 1.0
    velocity: float = 0.25
    wrench: float = 1.0
    time: float = 0.10


def normalize_state(
    progress: torch.Tensor, velocity: torch.Tensor, wrench: torch.Tensor, sigma_v: float, sigma_tau: float
) -> torch.Tensor:
    """Stack (s, v, tau) into a normalized state tensor z = [s, v/sigma_v, tau/sigma_tau].

    Args:
        progress: normalized task progress s, shape [..., T].
        velocity: raw task velocity, shape [..., T].
        wrench: raw required/measured generalized force, shape [..., T].
        sigma_v: velocity normalization scale.
        sigma_tau: wrench normalization scale.

    Returns:
        Tensor of shape [..., T, 3] with columns (s_bar, v_bar, tau_bar).
    """
    return torch.stack([progress, velocity / sigma_v, wrench / sigma_tau], dim=-1)


def physics_ot_cost(z_ref: torch.Tensor, z_rollout: torch.Tensor, weights: CostWeights | None = None) -> torch.Tensor:
    """Pairwise Physics-OT ground cost C_ij (section 9).

    C_ij = lambda_s (s*_i - s_j)^2 + lambda_v (v*_i - v_j)^2 + lambda_tau (tau*_i - tau_j)^2
           + lambda_t (i/T_H - j/T_R)^2

    Args:
        z_ref: normalized reference states, shape [..., T_H, 3] (columns: s, v_bar, tau_bar).
        z_rollout: normalized rollout states, shape [..., T_R, 3].
        weights: per-term weights; defaults to the values in section 9.

    Returns:
        Cost matrix of shape [..., T_H, T_R].
    """
    weights = weights or CostWeights()
    t_h = z_ref.shape[-2]
    t_r = z_rollout.shape[-2]

    diff = z_ref.unsqueeze(-2) - z_rollout.unsqueeze(-3)  # [..., T_H, T_R, 3]
    sq = diff.pow(2)
    state_weights = diff.new_tensor([weights.progress, weights.velocity, weights.wrench])
    cost = (sq * state_weights).sum(dim=-1)  # [..., T_H, T_R]

    idx_h = torch.arange(t_h, device=z_ref.device, dtype=z_ref.dtype) / max(t_h - 1, 1)
    idx_r = torch.arange(t_r, device=z_rollout.device, dtype=z_rollout.dtype) / max(t_r - 1, 1)
    time_cost = (idx_h.unsqueeze(-1) - idx_r.unsqueeze(-2)).pow(2)  # [T_H, T_R]
    cost = cost + weights.time * time_cost

    return cost
