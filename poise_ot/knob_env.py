"""A minimal, dependency-light Gym-style knob environment -- Eq 15's forward
dynamics as the ``step()`` integrator, no PhysX, no rendering:

    I*theta_ddot = tau_robot - b*theta_dot - k*(theta-theta0) - tau_c*tanh(theta_dot/v_eps)

Action = torque, per the Stage-1 spec literally (no joint-position-target
indirection like the IsaacLab Allegro env). Reuses
``physics_ot.dynamics.knob.KnobDynamics.passive_torque`` for the RHS (it is
exactly Eq 15's non-actuator terms, negated) and ``physics_ot.rewards.
task_reward`` for the ``-|theta-theta_goal|``/success-indicator core of Eq
``task_reward`` -- only the action-penalty/success-bonus scaling and the
(documented, zeroed) unsafe term are new here.

Vectorized over ``num_envs`` (a batch dimension, not a formal
``gymnasium.vector.VectorEnv``) so Stage 1's SAC training can use more than
one parallel env for sample throughput; ``num_envs=1`` behaves like a normal
single env for the sanity checks (tensors keep a leading size-1 dim rather
than being squeezed to python scalars -- callers index/`.item()` as needed).
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

from physics_ot.dynamics.knob import KnobDynamics, KnobParams
from physics_ot.rewards import task_reward


@dataclass
class KnobEnvConfig:
    params: KnobParams
    theta_goal: float
    success_threshold: float
    physics_dt: float = 1.0 / 120.0
    control_decimation: int = 4
    episode_seconds: float = 6.0
    tau_max: float = 1.0
    success_bonus: float = 10.0
    action_weight: float = 0.001
    unsafe_weight: float = 0.0  # lambda_u: always 0 here -- no unsafe states in a 1-DoF, unconstrained knob


class KnobEnv(gym.Env):
    """Vectorized, torch-native, no-simulator knob environment (Eq 15)."""

    metadata: dict = {}

    def __init__(self, cfg: KnobEnvConfig, num_envs: int = 1, device: str | torch.device = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.dynamics = KnobDynamics(cfg.params)
        self.num_envs = num_envs
        self.device = torch.device(device)

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        self._max_episode_steps = max(1, round(cfg.episode_seconds / (cfg.physics_dt * cfg.control_decimation)))
        self._success_threshold = cfg.success_threshold

        self.theta = torch.zeros(num_envs, device=self.device)
        self.theta_dot = torch.zeros(num_envs, device=self.device)
        self.theta_ddot = torch.zeros(num_envs, device=self.device)
        self.tau = torch.zeros(num_envs, device=self.device)
        self.elapsed_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    @property
    def dt(self) -> float:
        """Control-step spacing (physics_dt * control_decimation) -- what
        token_builder.py's ``dt`` argument should use for the rollout side."""
        return self.cfg.physics_dt * self.cfg.control_decimation

    def progress(self) -> torch.Tensor:
        return (self.theta - self.cfg.params.theta0) / (self.cfg.theta_goal - self.cfg.params.theta0)

    def _get_obs(self) -> torch.Tensor:
        s_goal = torch.ones_like(self.theta)  # progress target is always 1.0 by construction
        return torch.stack([self.progress(), self.theta_dot, s_goal], dim=-1)

    def _get_info(self) -> dict:
        return {
            "theta": self.theta.clone(),
            "theta_dot": self.theta_dot.clone(),
            "theta_ddot": self.theta_ddot.clone(),
            "tau": self.tau.clone(),
            "progress": self.progress(),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.theta[env_ids] = self.cfg.params.theta0
        self.theta_dot[env_ids] = 0.0
        self.theta_ddot[env_ids] = 0.0
        self.tau[env_ids] = 0.0
        self.elapsed_steps[env_ids] = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        tau_action = torch.as_tensor(action, dtype=torch.float32, device=self.device).reshape(self.num_envs)
        tau_action = tau_action.clamp(-1.0, 1.0) * self.cfg.tau_max

        for _ in range(self.cfg.control_decimation):
            passive = self.dynamics.passive_torque(self.theta, self.theta_dot)
            self.theta_ddot = (tau_action + passive) / self.cfg.params.inertia
            self.theta_dot = self.theta_dot + self.theta_ddot * self.cfg.physics_dt
            self.theta = self.theta + self.theta_dot * self.cfg.physics_dt
        self.tau = tau_action
        self.elapsed_steps += 1

        theta_goal_t = torch.full_like(self.theta, self.cfg.theta_goal)
        neg_error, is_success = task_reward(self.theta, theta_goal_t, self._success_threshold)
        action_penalty = tau_action.pow(2)
        # r_unsafe,t = 0 always (cfg.unsafe_weight defaults to 0.0) -- documented, not silently dropped.
        reward = (
            neg_error
            + self.cfg.success_bonus * is_success.float()
            - self.cfg.action_weight * action_penalty
            - self.cfg.unsafe_weight * torch.zeros_like(action_penalty)
        )

        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # no early termination
        truncated = self.elapsed_steps >= self._max_episode_steps

        info = self._get_info()
        info["is_success"] = is_success
        info["r_task"] = reward

        return self._get_obs(), reward, terminated, truncated, info
