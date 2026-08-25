from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class KnobParams:
    """Physical parameters of a 1-DoF resistive knob (section 5).

    Attributes:
        inertia: rotational inertia I [kg*m^2].
        damping: viscous damping b [N*m*s/rad].
        stiffness: return-spring stiffness k [N*m/rad].
        theta0: spring rest angle theta_0 [rad].
        coulomb_torque: Coulomb friction magnitude tau_c [N*m].
        v_eps: velocity smoothing scale for the tanh Coulomb model [rad/s].
    """

    inertia: float
    damping: float
    stiffness: float
    theta0: float
    coulomb_torque: float
    v_eps: float = 0.05

    def as_tensor_dict(self, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        return {
            "inertia": torch.tensor(self.inertia, device=device),
            "damping": torch.tensor(self.damping, device=device),
            "stiffness": torch.tensor(self.stiffness, device=device),
            "theta0": torch.tensor(self.theta0, device=device),
            "coulomb_torque": torch.tensor(self.coulomb_torque, device=device),
            "v_eps": torch.tensor(self.v_eps, device=device),
        }


class KnobDynamics:
    """Target-conditioned inverse dynamics for the resistive-knob object (section 5).

    tau* = I * theta_ddot + b * theta_dot + k * (theta - theta0) + tau_c * tanh(theta_dot / v_eps)

    This operator is applied both to the human-derived reference trajectory
    (producing tau*, the reference wrench) and to the robot rollout's measured
    trajectory (producing tau_R, section 7-8) -- it is one function of motion,
    not an estimate of any particular agent's applied force.

    Parameters may be scalars (shared across a batch) or tensors broadcastable
    against the trajectory batch dimension (e.g. per-environment physics
    randomization), matching the same physical model used to actuate the knob
    in simulation via :meth:`passive_torque`.
    """

    def __init__(self, params: KnobParams):
        self.params = params

    def required_generalized_force(
        self, position: torch.Tensor, velocity: torch.Tensor, acceleration: torch.Tensor
    ) -> torch.Tensor:
        p = self.params
        friction = p.coulomb_torque * torch.tanh(velocity / p.v_eps)
        return p.inertia * acceleration + p.damping * velocity + p.stiffness * (position - p.theta0) + friction

    def passive_torque(self, position: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        """Torque the knob's own (non-inertial) physics exerts against motion.

        This is the torque the simulation applies directly to the knob joint
        each physics step so that it behaves like a spring-damper-friction
        object on its own; any additional net torque comes from contact with
        the robot, and PhysX resolves the resulting acceleration.
        """
        p = self.params
        friction = p.coulomb_torque * torch.tanh(velocity / p.v_eps)
        return -(p.damping * velocity + p.stiffness * (position - p.theta0) + friction)
