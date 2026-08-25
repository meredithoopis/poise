from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import UnivariateSpline


@dataclass
class SmoothTrajectory:
    """A smoothed 1-DoF trajectory theta(t), differentiated analytically from the spline.

    Fitting a smooth spline once and differentiating it (rather than finite-
    differencing raw samples twice) avoids amplifying measurement noise into
    the velocity/acceleration signals (section 3).
    """

    position_spline: UnivariateSpline
    velocity_spline: UnivariateSpline
    acceleration_spline: UnivariateSpline
    t_min: float
    t_max: float

    def position(self, t: np.ndarray) -> np.ndarray:
        return self.position_spline(t)

    def velocity(self, t: np.ndarray) -> np.ndarray:
        return self.velocity_spline(t)

    def acceleration(self, t: np.ndarray) -> np.ndarray:
        return self.acceleration_spline(t)


def smooth_trajectory(t: np.ndarray, theta: np.ndarray, smoothing: float = 0.0, degree: int = 4) -> SmoothTrajectory:
    """Fit a smooth spline to a raw 1-DoF trajectory and expose analytic derivatives.

    Args:
        t: sample times, strictly increasing, shape [N].
        theta: raw (possibly noisy) position samples, shape [N].
        smoothing: spline smoothing factor (``s`` in ``UnivariateSpline``); 0
            interpolates exactly (appropriate for noise-free synthetic/
            scripted demonstrations), positive values smooth noisy real data.
        degree: spline degree; must be >= 3 so the acceleration (2nd
            derivative) is itself a smooth curve rather than piecewise-constant.

    Returns:
        A :class:`SmoothTrajectory` exposing position/velocity/acceleration.
    """
    if degree < 3:
        raise ValueError("degree must be >= 3 so the 2nd derivative is smooth")
    position_spline = UnivariateSpline(t, theta, k=degree, s=smoothing)
    velocity_spline = position_spline.derivative(1)
    acceleration_spline = position_spline.derivative(2)
    return SmoothTrajectory(
        position_spline=position_spline,
        velocity_spline=velocity_spline,
        acceleration_spline=acceleration_spline,
        t_min=float(t[0]),
        t_max=float(t[-1]),
    )
