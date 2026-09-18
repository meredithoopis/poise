"""Thin adapter onto ``physics_ot.references.smooth`` for real video's
irregular frame timestamps and nonzero smoothing factor.

``smooth_trajectory`` already does the right thing for the paper's own
warning (Sec 4: "filter before differentiating, not after") -- it fits one
smooth spline to the raw samples and differentiates that spline
analytically, rather than finite-differencing the raw (noisy) samples
twice. Stage 1's synthetic data used ``smoothing=0.0`` (exact interpolation,
appropriate for noise-free scripted trajectories); real video needs a
nonzero smoothing factor and must tolerate non-uniform frame spacing
(dropped frames, variable capture rate) -- both are already handled by
``UnivariateSpline`` under the hood, so this module is mostly a place to
make the smoothing-factor choice explicit and inspectable rather than new
math.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics_ot.references.smooth import SmoothTrajectory, smooth_trajectory


@dataclass
class SmoothingResult:
    trajectory: SmoothTrajectory
    residual_rms: float  # RMS of (raw - fitted) at the raw sample times -- feeds confidence_gate.py


def smooth_clip(
    t_raw: np.ndarray, theta_raw: np.ndarray, smoothing: float, degree: int = 4
) -> SmoothingResult:
    """Smooth a real-video trajectory and report the fit residual.

    Args:
        t_raw: possibly non-uniformly-spaced frame timestamps, shape [N].
        theta_raw: raw annotated joint angle, shape [N].
        smoothing: UnivariateSpline's ``s`` -- 0 exactly interpolates (only
            appropriate for noise-free data); real video should pass a
            positive value (tuned per-dataset, not per-clip, to keep results
            comparable across clips).
        degree: spline degree, propagated to ``smooth_trajectory`` (must be
            >=3 so acceleration is itself smooth).

    Returns:
        The fitted :class:`~physics_ot.references.smooth.SmoothTrajectory`
        plus the RMS fit residual -- a coarse, first input to
        ``confidence_gate.py``'s c_t scoring (a large residual means the
        spline could not track the raw annotation closely, which is a signal
        the annotation itself may be noisy/unreliable at that point).
    """
    trajectory = smooth_trajectory(t_raw, theta_raw, smoothing=smoothing, degree=degree)
    fitted = trajectory.position(t_raw)
    residual_rms = float(np.sqrt(np.mean((theta_raw - fitted) ** 2)))
    return SmoothingResult(trajectory=trajectory, residual_rms=residual_rms)
