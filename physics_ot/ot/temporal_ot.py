from __future__ import annotations

from dataclasses import dataclass

import torch

from .cost import CostWeights, normalize_state, physics_ot_cost
from .sinkhorn import batched_sinkhorn, transport_cost


@dataclass
class TemporalPhysicsOT:
    """Physics-OT temporal alignment (sections 7-10, 13).

    Wraps normalization, the ground cost, and batched Sinkhorn into a single
    distance D_POT between a physics reference and a robot rollout window, plus
    the incremental shaping reward r_OT,t = D_{t-1} - D_t.
    """

    sigma_v: float = 1.0
    sigma_tau: float = 1.0
    weights: CostWeights = None  # type: ignore[assignment]
    epsilon: float = 0.05
    num_iters: int = 20

    def __post_init__(self):
        if self.weights is None:
            self.weights = CostWeights()

    def distance(
        self,
        progress_ref: torch.Tensor,
        velocity_ref: torch.Tensor,
        wrench_ref: torch.Tensor,
        progress_rollout: torch.Tensor,
        velocity_rollout: torch.Tensor,
        wrench_rollout: torch.Tensor,
    ) -> torch.Tensor:
        """Compute D_POT between a reference trajectory and a rollout window.

        All ``*_ref`` tensors share shape [..., T_H]; all ``*_rollout`` tensors
        share shape [..., T_R]. Returns D_POT with shape [...].
        """
        with torch.no_grad():
            z_ref = normalize_state(progress_ref, velocity_ref, wrench_ref, self.sigma_v, self.sigma_tau)
            z_rollout = normalize_state(
                progress_rollout, velocity_rollout, wrench_rollout, self.sigma_v, self.sigma_tau
            )
            cost = physics_ot_cost(z_ref, z_rollout, self.weights)
            gamma = batched_sinkhorn(cost, epsilon=self.epsilon, num_iters=self.num_iters)
            return transport_cost(gamma, cost)

    @staticmethod
    def delta_reward(distance_prev: torch.Tensor, distance_curr: torch.Tensor) -> torch.Tensor:
        """Incremental shaping reward: positive when alignment improves (section 13)."""
        return distance_prev - distance_curr
