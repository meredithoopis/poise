from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from physics_ot.dynamics.knob import KnobDynamics, KnobParams

from .smooth import smooth_trajectory


@dataclass
class PhysicsReference:
    """The Physics-OT state reference z* = [s*, theta_dot*, tau*] (section 7)."""

    time: np.ndarray
    progress: np.ndarray
    velocity: np.ndarray
    wrench: np.ndarray
    physics_params: dict
    theta_goal: float

    @property
    def num_tokens(self) -> int:
        return int(self.time.shape[0])


def build_reference(
    t_raw: np.ndarray,
    theta_raw: np.ndarray,
    knob_params: KnobParams,
    theta_goal: float,
    num_tokens: int = 64,
    smoothing: float = 0.0,
) -> PhysicsReference:
    """Build a Physics-OT reference from a raw object-trajectory demonstration (sections 3-7).

    Args:
        t_raw: demonstration sample times, shape [N].
        theta_raw: demonstrated knob angle theta_H(t), shape [N].
        knob_params: target knob physics used to compute the required wrench.
        theta_goal: goal knob angle, used to normalize progress s in [0, 1].
        num_tokens: number of reference tokens T_H to resample to (default 64).
        smoothing: spline smoothing factor passed to :func:`smooth_trajectory`.

    Returns:
        A :class:`PhysicsReference` with ``num_tokens`` uniformly time-spaced
        entries.
    """
    trajectory = smooth_trajectory(t_raw, theta_raw, smoothing=smoothing)
    t = np.linspace(trajectory.t_min, trajectory.t_max, num_tokens)

    theta = trajectory.position(t)
    theta_dot = trajectory.velocity(t)
    theta_ddot = trajectory.acceleration(t)

    dynamics = KnobDynamics(knob_params)
    wrench = dynamics.required_generalized_force(
        torch.from_numpy(theta), torch.from_numpy(theta_dot), torch.from_numpy(theta_ddot)
    ).numpy()

    progress = (theta - knob_params.theta0) / (theta_goal - knob_params.theta0)

    return PhysicsReference(
        time=t,
        progress=progress,
        velocity=theta_dot,
        wrench=wrench,
        physics_params=asdict(knob_params),
        theta_goal=theta_goal,
    )


def save_reference(path: str | Path, reference: PhysicsReference) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        time=reference.time,
        progress=reference.progress,
        velocity=reference.velocity,
        wrench=reference.wrench,
        theta_goal=reference.theta_goal,
        **{f"physics_params.{k}": v for k, v in reference.physics_params.items()},
    )


def load_reference(path: str | Path) -> PhysicsReference:
    data = np.load(path)
    physics_params = {
        key.split(".", 1)[1]: float(data[key]) for key in data.files if key.startswith("physics_params.")
    }
    return PhysicsReference(
        time=data["time"],
        progress=data["progress"],
        velocity=data["velocity"],
        wrench=data["wrench"],
        physics_params=physics_params,
        theta_goal=float(data["theta_goal"]),
    )
