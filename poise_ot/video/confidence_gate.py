"""Wrench-confidence gating c_t (Sec 4.4) -- the first place a video-derived
wrench/impulse/power estimate can actually be unreliable (occlusion, fast
motion, tracking failure), as opposed to Stage 1's exact synthetic data
where c_t is trivially 1.0 everywhere.

Combines three signals into a single c_t in [0, 1] per token (the paper's
Sec 4.4 lists smoothing-residual magnitude and local trajectory-estimation
uncertainty as separate sub-signals; they are merged into one
``residual_confidence`` signal here, since a genuinely separate local-
uncertainty estimate would need the tracker's own per-frame
confidence/covariance output, which isn't available from this repo's
synthetic/local pipeline -- documented as a simplification, not silently
dropped):

1. Smoothing-residual / local fit gap (large spline-fit residual near a
   token -> raw annotation was noisy there).
2. Inverse-dynamics conditioning: how close theta_dot is to the tanh kink at
   v_eps, where the Coulomb-friction term is least linear (and so most
   sensitive to velocity-estimation error).
3. Wrench-consistency: the residual between tau* and a short causal-window
   local median of tau* itself, under the assumed model -- large residual
   means the estimated tau* is locally inconsistent with its own recent
   history, a sign that either the assumption or the estimate is off at
   that point.

The combination is a simple product of per-signal scores in [0, 1] (each
signal can independently veto confidence to zero; this is the same
philosophy as an AND-gate over reliability signals, not an average that
lets one bad signal be washed out by three good ones).
"""

from __future__ import annotations

import numpy as np

from physics_ot.dynamics.knob import KnobParams

from .inverse_dynamics import InverseDynamicsResult
from .smoothing import SmoothingResult


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def residual_confidence(smoothing_result: SmoothingResult, theta: np.ndarray, scale: float) -> np.ndarray:
    """Score 1: higher residual -> lower confidence, per-token (broadcast the
    clip-level residual_rms uniformly is too coarse -- use the local spline
    fit gap at each token's own theta instead, so within-clip regions that
    were fit worse are distinguished)."""
    fitted_at_theta = smoothing_result.trajectory.position(theta)
    local_gap = np.abs(theta - fitted_at_theta)
    return _clip01(1.0 - local_gap / max(scale, 1e-9))


def kink_conditioning_confidence(theta_dot: np.ndarray, v_eps: float, margin: float = 2.0) -> np.ndarray:
    """Score 3: confidence dips near the tanh(theta_dot/v_eps) kink (|theta_dot|
    small relative to v_eps), where the Coulomb term is least linear and
    most sensitive to velocity-estimation error."""
    ratio = np.abs(theta_dot) / max(v_eps, 1e-9)
    return _clip01(ratio / margin)


def wrench_consistency_confidence(ind_result: InverseDynamicsResult, window: int = 3) -> np.ndarray:
    """Score 3: residual between tau* and a short causal-window rolling
    median of tau* itself, under the assumed model -- a crude but explicit
    self-consistency check (large local jumps in tau* are more likely
    estimation noise than genuine fast torque changes at HOI4D's frame
    rate)."""
    tau = ind_result.tau_star
    n = len(tau)
    if n < window:
        return np.ones(n)
    padded = np.pad(tau, (window // 2, window - 1 - window // 2), mode="edge")
    local_median = np.array([np.median(padded[i : i + window]) for i in range(n)])
    residual = np.abs(tau - local_median)
    scale = max(np.std(tau), 1e-6)
    return _clip01(1.0 - residual / (3.0 * scale))


def compute_confidence(
    smoothing_result: SmoothingResult,
    ind_result: InverseDynamicsResult,
    params: KnobParams,
    residual_scale: float,
) -> np.ndarray:
    """c_t (Sec 4.4): product of the three signals above, per token.

    Args:
        residual_scale: a clip-independent scale (e.g. the success_threshold
            or another characteristic angle) that ``residual_confidence``
            normalizes the local fit gap against, so c_t is comparable
            across clips of different raw-annotation noise levels.
    """
    c_residual = residual_confidence(smoothing_result, ind_result.theta, residual_scale)
    c_kink = kink_conditioning_confidence(ind_result.theta_dot, params.v_eps)
    c_wrench = wrench_consistency_confidence(ind_result)
    return c_residual * c_kink * c_wrench
