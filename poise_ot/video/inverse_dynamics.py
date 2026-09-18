"""Target-conditioned inverse dynamics for video-derived trajectories (Eq
10/20) -- reuses ``physics_ot.dynamics.knob.KnobDynamics.
required_generalized_force`` unchanged (it's the same formula regardless of
whether theta_H(t) came from a script, an integrated torque profile, or
smoothed video).

The only new thing here is that HOI4D provides no ground-truth object
dynamics parameters (mass/inertia/damping/stiffness/friction) -- an assumed
target ``I_e, b_e, k_e, tau_c,e`` must be picked (configs/poise_ot.yaml's
``stage2:`` section), and this module's whole job is making sure that
assumption is never silently lost: every call logs the exact values used,
both to stdout and into the returned result, so a downstream json/plot can
never be read without also seeing what physics parameters produced it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from physics_ot.dynamics.knob import KnobDynamics, KnobParams

from .smoothing import SmoothingResult


@dataclass
class InverseDynamicsResult:
    t: np.ndarray  # [T_H] resampled token times
    theta: np.ndarray
    theta_dot: np.ndarray
    theta_ddot: np.ndarray
    tau_star: np.ndarray  # target-conditioned required torque (Eq 10/20)
    assumed_params: dict  # I_e, b_e, k_e, tau_c,e, v_eps -- logged explicitly, see module docstring


def target_conditioned_torque(
    smoothing_result: SmoothingResult,
    assumed_params: KnobParams,
    num_tokens: int = 64,
    verbose: bool = True,
) -> InverseDynamicsResult:
    """Compute tau*_t (Eq 10/20) from a smoothed video trajectory under an
    ASSUMED target object model (HOI4D has no ground-truth dynamics params).

    Args:
        smoothing_result: output of ``smoothing.smooth_clip``.
        assumed_params: the assumed target I_e/b_e/k_e/tau_c,e/v_eps
            (configs/poise_ot.yaml's ``stage2:`` section) -- NOT measured
            from this clip or from HOI4D annotations.
        num_tokens: number of uniformly time-spaced tokens to resample to.
        verbose: print the assumed params being used (in addition to
            returning them in the result) -- this assumption should never be
            silently invisible in a log.
    """
    if verbose:
        print(f"[inverse_dynamics] ASSUMED target params (not measured): {assumed_params}")

    trajectory = smoothing_result.trajectory
    t = np.linspace(trajectory.t_min, trajectory.t_max, num_tokens)
    theta = trajectory.position(t)
    theta_dot = trajectory.velocity(t)
    theta_ddot = trajectory.acceleration(t)

    dynamics = KnobDynamics(assumed_params)
    tau_star = dynamics.required_generalized_force(
        torch.from_numpy(theta), torch.from_numpy(theta_dot), torch.from_numpy(theta_ddot)
    ).numpy()

    return InverseDynamicsResult(
        t=t,
        theta=theta,
        theta_dot=theta_dot,
        theta_ddot=theta_ddot,
        tau_star=tau_star,
        assumed_params=asdict(assumed_params),
    )
