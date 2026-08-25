from __future__ import annotations

import torch

from physics_ot.ot.baselines import dtw_distance, ot_state_distance, pointwise_l2_distance, pointwise_physics_distance
from physics_ot.ot.cost import CostWeights, normalize_state
from physics_ot.ot.temporal_ot import TemporalPhysicsOT

REWARD_MODES = ("sparse", "state_tracking", "dtw_physics", "ot_state", "pointwise_physics", "physics_ot")


def compute_distance(
    reward_mode: str,
    reference_progress: torch.Tensor,
    reference_velocity: torch.Tensor,
    reference_wrench: torch.Tensor,
    window_progress: torch.Tensor,
    window_velocity: torch.Tensor,
    window_wrench: torch.Tensor,
    sigma_v: float,
    sigma_tau: float,
    cost_weights: CostWeights,
    sinkhorn_epsilon: float,
    sinkhorn_iters: int,
) -> torch.Tensor:
    """Dispatch to the Physics-OT distance or one of the Experiment-0 baselines (sections 25-26).

    ``reference_*`` have shape [T_H]; ``window_*`` have shape [num_envs, T_R]
    (broadcast automatically against the reference).

    Returns a per-env distance tensor of shape [num_envs]. For ``reward_mode
    == "sparse"`` this returns zeros (no shaping signal is used).
    """
    batch = window_progress.shape[0]
    ref_progress = reference_progress.unsqueeze(0).expand(batch, -1)
    ref_velocity = reference_velocity.unsqueeze(0).expand(batch, -1)
    ref_wrench = reference_wrench.unsqueeze(0).expand(batch, -1)

    if reward_mode == "sparse":
        return torch.zeros(batch, device=window_progress.device)

    if reward_mode == "state_tracking":
        return pointwise_l2_distance(ref_progress, ref_velocity, window_progress, window_velocity, sigma_v)

    if reward_mode == "pointwise_physics":
        return pointwise_physics_distance(
            ref_progress, ref_velocity, ref_wrench, window_progress, window_velocity, window_wrench,
            sigma_v, sigma_tau,
        )

    if reward_mode == "ot_state":
        return ot_state_distance(
            ref_progress, ref_velocity, window_progress, window_velocity, sigma_v, sinkhorn_epsilon, sinkhorn_iters
        )

    if reward_mode == "dtw_physics":
        z_ref = normalize_state(ref_progress, ref_velocity, ref_wrench, sigma_v, sigma_tau)
        z_win = normalize_state(window_progress, window_velocity, window_wrench, sigma_v, sigma_tau)
        return dtw_distance(z_ref, z_win, cost_weights)

    if reward_mode == "physics_ot":
        pot = TemporalPhysicsOT(
            sigma_v=sigma_v, sigma_tau=sigma_tau, weights=cost_weights, epsilon=sinkhorn_epsilon,
            num_iters=sinkhorn_iters,
        )
        return pot.distance(
            ref_progress, ref_velocity, ref_wrench, window_progress, window_velocity, window_wrench
        )

    raise ValueError(f"Unknown reward_mode '{reward_mode}', expected one of {REWARD_MODES}")
