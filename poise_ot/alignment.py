"""Windowed robot-rollout occupancy (Eq window_rollout), the instantaneous
mechanical-alignment cost c_t^align (Eq alignment_cost), and the alignment
state h_t (Eq alignment_state) -- the online, per-step glue between
``knob_env.py``'s rollout and ``mech_cost.py``'s distance, consumed by
``two_critic_sac.py``'s Q_C.

c_t^align is the RAW mechanics-OT distance between the full reference and
the current rollout window -- not a potential difference. The potential-
difference shaping reward (Eq alignment_progress, ``D_{t-1}-D_t``) is kept
here only as a clearly separate, ablation-only static method: the paper is
explicit that summing that delta into the reward was "the earlier ambiguous
construction," now replaced by feeding the two critics separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from physics_ot.dynamics.knob import KnobParams

from .mech_cost import MechCostWeights, MechCostScales, alignment_distance
from .token_builder import Token, build_token


@dataclass
class AlignmentConfig:
    window_h: int = 32  # H (Eq window_rollout) -- configs/poise_ot.yaml's alignment.window_h
    epsilon: float = 0.05
    num_iters: int = 20
    impulse_window: int = 5  # h -- the token's own short-window impulse length, distinct from H
    cost_mode: str = "mechanics_ot"  # "mechanics_ot" (default) | "euclidean_ot" | "pointwise_mechanics"


class RollingAlignment:
    """Per-env ring buffer of the last H rollout tokens, plus c_t^align/h_t.

    One instance per training run (not per-episode): :meth:`reset_idx` clears
    the buffer only for the envs that just reset, matching the "expose
    pre-reset state via extras/info, not by re-deriving after reset" pattern
    already required by ``knob_allegro_env.py`` in the original pipeline.
    """

    def __init__(
        self,
        reference_token: Token,
        reference_time: torch.Tensor,
        params: KnobParams,
        num_envs: int,
        device: str | torch.device,
        cfg: AlignmentConfig | None = None,
        weights: MechCostWeights | None = None,
        scales: MechCostScales | None = None,
        cost_terms: str = "K+J+P+D",
    ):
        self.reference_token = reference_token
        self.reference_time = reference_time.to(device)
        self.params = params
        self.num_envs = num_envs
        self.device = device
        self.cfg = cfg or AlignmentConfig()
        self.weights = weights
        self.scales = scales
        self.cost_terms = cost_terms

        h = self.cfg.window_h
        self._buffer = torch.zeros(num_envs, h, 5, device=device)
        self._tau_buffer = torch.zeros(num_envs, h, device=device)
        self._filled = torch.zeros(num_envs, dtype=torch.long, device=device)

    @property
    def h_dim(self) -> int:
        """Dimensionality of h_t excluding obs -- (z_ref token) + (flattened window)."""
        return 5 + self.cfg.window_h * 5

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        self._buffer[env_ids] = 0.0
        self._tau_buffer[env_ids] = 0.0
        self._filled[env_ids] = 0

    def push(self, progress: torch.Tensor, theta_dot: torch.Tensor, tau: torch.Tensor, dt: float) -> Token:
        """Build this control step's rollout token and push it into the window."""
        token = build_token(progress, theta_dot, tau, self.params, dt, window_h=self.cfg.impulse_window)
        self._buffer = torch.cat([self._buffer[:, 1:], token.z.unsqueeze(1)], dim=1)
        self._tau_buffer = torch.cat([self._tau_buffer[:, 1:], token.tau.unsqueeze(1)], dim=1)
        self._filled = (self._filled + 1).clamp(max=self.cfg.window_h)
        return token

    def _nearest_reference_token(self, elapsed_time: torch.Tensor) -> torch.Tensor:
        clamped = elapsed_time.clamp(max=self.reference_time[-1].item())
        idx = torch.searchsorted(self.reference_time, clamped)
        idx = idx.clamp(0, self.reference_time.shape[0] - 1)
        return self.reference_token.z[idx]

    def step_alignment(self, obs: torch.Tensor, elapsed_time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (c_t_align, h_t) from the current window contents.

        c_t_align (Eq alignment_cost): D_t^mech between the full reference
        occupancy and the windowed rollout occupancy.
        h_t (Eq alignment_state): [obs, z_t^ref, psi(H_t)], with psi(H_t)
        taken as the flattened window itself -- the paper's stated simplest
        choice ("the simplest implementation uses the fixed window").
        """
        rollout_z = self._buffer
        ref_token_batched = Token(
            z=self.reference_token.z.unsqueeze(0).expand(self.num_envs, -1, -1),
            confidence=self.reference_token.confidence.unsqueeze(0).expand(self.num_envs, -1),
            tau=self.reference_token.tau.unsqueeze(0).expand(self.num_envs, -1),
        )
        rollout_token = Token(z=rollout_z, confidence=torch.ones_like(self._tau_buffer), tau=self._tau_buffer)

        c_align = alignment_distance(
            ref_token_batched,
            rollout_token,
            self.params.inertia,
            cost_mode=self.cfg.cost_mode,
            weights=self.weights,
            scales=self.scales,
            cost_terms=self.cost_terms,
            epsilon=self.cfg.epsilon,
            num_iters=self.cfg.num_iters,
        )
        z_ref_t = self._nearest_reference_token(elapsed_time)
        h_t = torch.cat([obs, z_ref_t, rollout_z.reshape(self.num_envs, -1)], dim=-1)
        return c_align, h_t

    @staticmethod
    def alignment_progress_reward(distance_prev: torch.Tensor, distance_curr: torch.Tensor) -> torch.Tensor:
        """Eq alignment_progress: D_{t-1} - D_t. ABLATION ONLY -- never the
        default Q_C training target (that's the raw c_t_align from
        :meth:`step_alignment`); the paper explicitly frames this delta as
        the *replaced* earlier construction, kept only for the dedicated
        alignment-progress-reward ablation."""
        return distance_prev - distance_curr
