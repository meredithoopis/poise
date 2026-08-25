from __future__ import annotations

from typing import Protocol

import torch


class InverseDynamics(Protocol):
    """Target-conditioned inverse-dynamics operator D_eta.

    Maps a task-space trajectory (position, velocity, acceleration) to the
    generalized force/torque required to produce it under a target object's
    physical model. The same operator is applied to a human-derived reference
    trajectory and to a robot rollout trajectory: it never sees who produced
    the motion, only the motion itself and the target physics parameters.
    """

    def required_generalized_force(
        self, position: torch.Tensor, velocity: torch.Tensor, acceleration: torch.Tensor
    ) -> torch.Tensor:
        """Compute the generalized force/torque required to realize a trajectory.

        Args:
            position: task-space position, arbitrary leading batch/time dims.
            velocity: task-space velocity, same shape as ``position``.
            acceleration: task-space acceleration, same shape as ``position``.

        Returns:
            Required generalized force/torque, same shape as ``position``.
        """
        ...
