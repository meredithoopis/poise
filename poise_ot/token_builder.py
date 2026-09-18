"""The physical task token z_t (Eq 30), shared unchanged between Stage 1
(synthetic reference + simulated rollout) and Stage 2 (video-derived
reference) -- this is the entire point of factoring it out on its own: the
token doesn't care whether theta_H(t) came from a script or from video.

    z_t = [s_t, theta_dot_t, J_tau,t, P_t, P_diss,t]        (Eq 30)

using the short-window torque impulse (Eq 23-24) and the power/dissipation
observables (Eq 26). A scalar confidence c_t (Sec 4.4) travels alongside
z_t but is not one of its five channels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from physics_ot.dynamics.knob import KnobParams

# Column order of Token.z -- referenced by index elsewhere (mech_cost.py),
# kept here as the single source of truth.
TOKEN_COLUMNS = ("progress", "velocity", "impulse", "power", "dissipation")


@dataclass
class Token:
    """z_t (Eq 30) plus its confidence and the raw torque it was built from.

    Attributes:
        z: shape [..., T, 5], columns (s, theta_dot, J_tau, P, P_diss).
        confidence: c_t, shape [..., T]. All-ones unless explicitly gated
            (Stage 1's exact synthetic data; Stage 2's video/confidence_gate.py
            computes real values).
        tau: raw instantaneous torque, shape [..., T]. Not one of Eq 30's five
            channels -- kept only so mech_cost.py's "euclidean_ot" baseline
            mode (the original Physics-OT cost, reused for comparison) has a
            raw wrench signal to normalize, instead of the impulse.
    """

    z: torch.Tensor
    confidence: torch.Tensor
    tau: torch.Tensor


def windowed_impulse(tau: torch.Tensor, dt: float, window_h: int) -> torch.Tensor:
    """J_tau,t = integral_{t-h/2}^{t+h/2} tau dt (Eq 23), as a discrete centered
    rectangle-rule sum over ``window_h`` samples, clamped (replicated) at the
    trajectory's edges.

    Args:
        tau: shape [..., T].
        dt: sample spacing.
        window_h: window length in samples; <=1 reduces to a single-sample
            impulse ``tau * dt``.

    Returns:
        shape [..., T], same as ``tau``.
    """
    if window_h <= 1:
        return tau * dt
    t_len = tau.shape[-1]
    half = (window_h - 1) // 2
    offsets = torch.arange(-half, window_h - half, device=tau.device)
    idx = torch.arange(t_len, device=tau.device).unsqueeze(-1) + offsets.unsqueeze(0)  # [T, window_h]
    idx = idx.clamp(0, t_len - 1)
    windows = tau[..., idx]  # [..., T, window_h]
    return windows.sum(dim=-1) * dt


def build_token(
    progress: torch.Tensor,
    theta_dot: torch.Tensor,
    tau: torch.Tensor,
    params: KnobParams,
    dt: float,
    window_h: int = 1,
    confidence: torch.Tensor | None = None,
) -> Token:
    """Build z_t (Eq 30) from a trajectory's progress/velocity/torque.

    Args:
        progress: s_t = (theta_t - theta0) / (theta_goal - theta0), shape [..., T].
        theta_dot: shape [..., T].
        tau: applied/required torque -- tau* for a reference, tau_action for a
            rollout (the same function is deliberately run on both; that
            symmetry is what makes z_t comparable across reference and
            rollout at all). Shape [..., T].
        params: target knob physics (damping/coulomb_torque/v_eps feed the
            dissipation term; inertia is used by mech_cost.py, not here).
        dt: sample spacing, used by the impulse integral.
        window_h: impulse window length in samples (configs/poise_ot.yaml's
            token.impulse_window).
        confidence: c_t, shape [..., T]; defaults to all-ones (exact data,
            e.g. Stage 1's synthetic reference or any simulated rollout).

    Returns:
        A :class:`Token` with ``z`` of shape [..., T, 5].
    """
    j_tau = windowed_impulse(tau, dt, window_h)
    p = tau * theta_dot
    p_diss = params.damping * theta_dot.pow(2) + params.coulomb_torque * theta_dot * torch.tanh(
        theta_dot / params.v_eps
    )
    z = torch.stack([progress, theta_dot, j_tau, p, p_diss], dim=-1)
    if confidence is None:
        confidence = torch.ones_like(progress)
    return Token(z=z, confidence=confidence, tau=tau)
