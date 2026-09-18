"""The mechanics-induced ground cost C_ij^mech (Eq 34, knob-specialized via
Eqs knob_kinetic/knob_dual/knob_power), entropic OT distance (reusing
``physics_ot.ot.sinkhorn`` unchanged -- the solver is cost-matrix-agnostic),
and the debiased Sinkhorn divergence the paper asks be reported alongside
D_POISE.

Also implements the two baseline ``cost_mode``s needed for Phase-I's decisive
comparisons and Stage 1's "ablation wiring sanity" check:

- ``euclidean_ot``  -- the *original* Physics-OT cost (standardized Euclidean
  state+wrench, entropic OT). Reuses ``physics_ot.ot.cost``/``sinkhorn``
  directly, unmodified: this mode exists specifically to reproduce the old
  formulation for comparison, not to reimplement it.
- ``pointwise_mechanics`` -- the same 5-channel mechanical terms as
  ``mechanics_ot`` but index-matched (resampled, elementwise), no Sinkhorn.

All three modes are reached through one dispatcher, :func:`alignment_distance`,
so swapping ``cost_mode`` is guaranteed to change what's actually computed
(not just a label) -- that guarantee is exactly what the ablation-wiring
check verifies.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from physics_ot.ot.baselines import _resample_to_length
from physics_ot.ot.cost import normalize_state, physics_ot_cost
from physics_ot.ot.sinkhorn import batched_sinkhorn, transport_cost

from .token_builder import Token

_TERM_LEVELS: dict[str, frozenset[str]] = {
    "K": frozenset({"velocity"}),
    "K+J": frozenset({"velocity", "impulse"}),
    "K+J+P": frozenset({"velocity", "impulse", "power"}),
    "K+J+P+D": frozenset({"velocity", "impulse", "power", "dissipation"}),
}


@dataclass
class MechCostWeights:
    """Weights for the mechanics-induced ground cost (Eq 34)."""

    progress: float = 1.0  # lambda_s -- always included, independent of cost_terms
    velocity: float = 0.25  # lambda_v (kinetic-geometry term K)
    impulse: float = 1.0  # lambda_J (dual interaction-impulse term)
    power: float = 0.1  # lambda_P
    dissipation: float = 0.1  # lambda_D
    time: float = 0.10  # lambda_t -- always included, independent of cost_terms


@dataclass
class MechCostScales:
    """Characteristic normalization scales (E0, P0, PD0) for Eq 34's process terms."""

    E0: float = 1.0
    P0: float = 1.0
    PD0: float = 1.0


def _unbind_z(z: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """z: [..., T, 5] -> (progress, velocity, impulse, power, dissipation), each [..., T]."""
    return tuple(z[..., i] for i in range(5))


def mechanics_pairwise_cost(
    z_ref: torch.Tensor,
    confidence_ref: torch.Tensor,
    z_rollout: torch.Tensor,
    inertia: float,
    weights: MechCostWeights | None = None,
    scales: MechCostScales | None = None,
    cost_terms: str = "K+J+P+D",
) -> torch.Tensor:
    """Pairwise C_ij^mech (Eq 34 / Eqs knob_kinetic, knob_dual, knob_power).

    Args:
        z_ref: reference tokens, shape [..., T_H, 5].
        confidence_ref: c_i* gating the impulse/power/dissipation terms
            (Sec 4.4) -- the progress and velocity terms stay ungated. Shape
            [..., T_H].
        z_rollout: rollout tokens, shape [..., T_R, 5].
        inertia: I_e, weights the velocity term and dual-weights the impulse
            term (Eqs knob_kinetic/knob_dual).
        cost_terms: one of "K", "K+J", "K+J+P", "K+J+P+D" -- which mechanical
            terms beyond progress/time are included (the paper's ablation
            ladder). Progress and time are always included.

    Returns:
        Cost matrix, shape [..., T_H, T_R].
    """
    weights = weights or MechCostWeights()
    scales = scales or MechCostScales()
    active = _TERM_LEVELS[cost_terms]

    s_ref, v_ref, j_ref, p_ref, d_ref = _unbind_z(z_ref)
    s_roll, v_roll, j_roll, p_roll, d_roll = _unbind_z(z_rollout)

    t_h, t_r = z_ref.shape[-2], z_rollout.shape[-2]
    conf = confidence_ref.unsqueeze(-1)  # [..., T_H, 1], broadcasts over T_R

    ds = s_ref.unsqueeze(-1) - s_roll.unsqueeze(-2)
    cost = weights.progress * ds.pow(2)

    if "velocity" in active:
        dv = v_ref.unsqueeze(-1) - v_roll.unsqueeze(-2)
        cost = cost + weights.velocity * inertia * dv.pow(2) / scales.E0

    if "impulse" in active:
        dj = j_ref.unsqueeze(-1) - j_roll.unsqueeze(-2)
        cost = cost + conf * weights.impulse * dj.pow(2) / inertia / scales.E0

    if "power" in active:
        dp = p_ref.unsqueeze(-1) - p_roll.unsqueeze(-2)
        cost = cost + conf * weights.power * (dp / scales.P0).pow(2)

    if "dissipation" in active:
        dd = d_ref.unsqueeze(-1) - d_roll.unsqueeze(-2)
        cost = cost + conf * weights.dissipation * (dd / scales.PD0).pow(2)

    idx_h = torch.arange(t_h, device=z_ref.device, dtype=z_ref.dtype) / max(t_h - 1, 1)
    idx_r = torch.arange(t_r, device=z_rollout.device, dtype=z_rollout.dtype) / max(t_r - 1, 1)
    time_cost = (idx_h.unsqueeze(-1) - idx_r.unsqueeze(-2)).pow(2)
    cost = cost + weights.time * time_cost

    return cost


def mechanics_ot_distance(
    z_ref: torch.Tensor,
    confidence_ref: torch.Tensor,
    z_rollout: torch.Tensor,
    inertia: float,
    weights: MechCostWeights | None = None,
    scales: MechCostScales | None = None,
    cost_terms: str = "K+J+P+D",
    epsilon: float = 0.05,
    num_iters: int = 20,
) -> torch.Tensor:
    """D_POISE = <Gamma*, C^mech> via entropic OT (Eq 39/40), reusing
    ``physics_ot.ot.sinkhorn`` unchanged. Returns shape [...] (batch only)."""
    with torch.no_grad():
        cost = mechanics_pairwise_cost(z_ref, confidence_ref, z_rollout, inertia, weights, scales, cost_terms)
        gamma = batched_sinkhorn(cost, epsilon=epsilon, num_iters=num_iters)
        return transport_cost(gamma, cost)


def debiased_sinkhorn_divergence(
    ref: Token,
    rollout: Token,
    inertia: float,
    weights: MechCostWeights | None = None,
    scales: MechCostScales | None = None,
    cost_terms: str = "K+J+P+D",
    epsilon: float = 0.05,
    num_iters: int = 20,
) -> torch.Tensor:
    """S_eps(rho*, rho) = D(rho*,rho) - 0.5*D(rho*,rho*) - 0.5*D(rho,rho).

    Removes the entropic self-distance bias that otherwise makes even
    ``mechanics_ot_distance(ref, ref)`` nonzero (see the paper's "Optimal
    Transport alignment" section, and the analogous known/documented gap in
    ``physics_ot`` -- ``tests/test_sinkhorn.py``'s comment on this exact
    bias). Reported *alongside* the raw distance, never as a silent
    replacement for it.
    """
    d_ref_roll = mechanics_ot_distance(ref.z, ref.confidence, rollout.z, inertia, weights, scales, cost_terms, epsilon, num_iters)
    d_ref_ref = mechanics_ot_distance(ref.z, ref.confidence, ref.z, inertia, weights, scales, cost_terms, epsilon, num_iters)
    d_roll_roll = mechanics_ot_distance(
        rollout.z, rollout.confidence, rollout.z, inertia, weights, scales, cost_terms, epsilon, num_iters
    )
    return d_ref_roll - 0.5 * d_ref_ref - 0.5 * d_roll_roll


def pointwise_mechanics_distance(
    z_ref: torch.Tensor,
    confidence_ref: torch.Tensor,
    z_rollout: torch.Tensor,
    inertia: float,
    weights: MechCostWeights | None = None,
    scales: MechCostScales | None = None,
    cost_terms: str = "K+J+P+D",
) -> torch.Tensor:
    """Index-matched (resampled, no Sinkhorn) mechanical distance -- the
    "Pointwise Mechanics" baseline: does mechanics help without sequence
    matching? Mirrors ``physics_ot.ot.baselines.pointwise_physics_distance``'s
    structure on the new 5-channel token. Returns shape [...] (batch only)."""
    weights = weights or MechCostWeights()
    scales = scales or MechCostScales()
    active = _TERM_LEVELS[cost_terms]

    t_r = z_rollout.shape[-2]
    s_ref, v_ref, j_ref, p_ref, d_ref = (_resample_to_length(z_ref[..., i], t_r) for i in range(5))
    conf_ref = _resample_to_length(confidence_ref, t_r)
    s_roll, v_roll, j_roll, p_roll, d_roll = _unbind_z(z_rollout)

    cost = weights.progress * (s_ref - s_roll).pow(2)

    if "velocity" in active:
        cost = cost + weights.velocity * inertia * (v_ref - v_roll).pow(2) / scales.E0
    if "impulse" in active:
        cost = cost + conf_ref * weights.impulse * (j_ref - j_roll).pow(2) / inertia / scales.E0
    if "power" in active:
        cost = cost + conf_ref * weights.power * ((p_ref - p_roll) / scales.P0).pow(2)
    if "dissipation" in active:
        cost = cost + conf_ref * weights.dissipation * ((d_ref - d_roll) / scales.PD0).pow(2)

    return cost.mean(dim=-1)


def alignment_distance(
    ref: Token,
    rollout: Token,
    inertia: float,
    cost_mode: str = "mechanics_ot",
    weights: MechCostWeights | None = None,
    scales: MechCostScales | None = None,
    cost_terms: str = "K+J+P+D",
    sigma_v: float = 1.0,
    sigma_tau: float = 1.0,
    epsilon: float = 0.05,
    num_iters: int = 20,
) -> torch.Tensor:
    """Dispatch across the three ablation cost modes -- the single call site
    ``alignment.py`` and the Stage-1 checks use, so swapping ``cost_mode`` is
    the only thing that changes between the primary method and its baselines.

    Args:
        cost_mode: "mechanics_ot" (default, Eq 34 + Sinkhorn), "pointwise_mechanics"
            (same terms, index-matched, no Sinkhorn), or "euclidean_ot" (the
            *original* Physics-OT cost -- standardized Euclidean state+wrench,
            entropic OT -- reused unmodified from ``physics_ot.ot.cost``).
    """
    if cost_mode == "mechanics_ot":
        return mechanics_ot_distance(ref.z, ref.confidence, rollout.z, inertia, weights, scales, cost_terms, epsilon, num_iters)
    if cost_mode == "pointwise_mechanics":
        return pointwise_mechanics_distance(ref.z, ref.confidence, rollout.z, inertia, weights, scales, cost_terms)
    if cost_mode == "euclidean_ot":
        s_ref, v_ref = ref.z[..., 0], ref.z[..., 1]
        s_roll, v_roll = rollout.z[..., 0], rollout.z[..., 1]
        z_ref = normalize_state(s_ref, v_ref, ref.tau, sigma_v, sigma_tau)
        z_roll = normalize_state(s_roll, v_roll, rollout.tau, sigma_v, sigma_tau)
        with torch.no_grad():
            cost = physics_ot_cost(z_ref, z_roll)
            gamma = batched_sinkhorn(cost, epsilon=epsilon, num_iters=num_iters)
            return transport_cost(gamma, cost)
    raise ValueError(f"unknown cost_mode: {cost_mode!r}")
