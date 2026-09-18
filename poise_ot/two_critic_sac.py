"""Two-critic SAC (Eqs task_reward, cost_critic, alignment_state, sac_actor_obj).

Deliberately conventional continuous SAC internals -- squashed-Gaussian
actor, reparameterized sampling, twin-Q target networks with Polyak
averaging for *each* critic head, replay buffer -- per the paper's own
framing: "SAC is kept deliberately conventional so that the scientific
contribution can be attributed to the physical representation and alignment
signal rather than to a simultaneously redesigned RL algorithm." The only
non-standard piece is that there are two independently-trained critics
feeding one actor objective:

- ``Q_task(o, a)``: trained only on the task-and-safety reward (Eq
  task_reward) via a standard SAC Bellman target.
- ``Q_C(h, a)``: trained on the raw mechanical-alignment cost c_t^align (Eq
  alignment_cost) via an ordinary discounted-return TD target -- it is a
  ordinary value critic whose reward channel happens to be a cost, so no
  sign flip happens inside the critic itself.
- Actor objective (Eq sac_actor_obj):
  ``J_pi = E[Q_task(o,a) - xi_t*Q_C(h,a) - alpha*log pi(a|o)]``, maximized.
  Q_C is *subtracted* -- the paper's explicit sign-convention correction to
  "the earlier ambiguous construction" (a positive alignment-improvement
  reward that was learned and then subtracted).

No RL library dependency: SB3/rllib don't expose a two-critic actor
objective like this out of the box, and the Stage-1 checks need direct
access to each critic's loss/gradients every update, which needs full
control over the loop.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


def _mlp(in_dim: int, out_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(last, h), nn.ReLU()]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """Squashed-Gaussian actor with reparameterized sampling and the
    standard tanh log-prob correction."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...] = (256, 256)):
        super().__init__()
        self.body = _mlp(obs_dim, 2 * action_dim, hidden_dims)
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.body(obs).chunk(2, dim=-1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        log_prob = normal.log_prob(x) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


class TwinQ(nn.Module):
    """Twin Q-networks (min-Q trick) -- used for both Q_task and Q_C."""

    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...] = (256, 256)):
        super().__init__()
        self.q1 = _mlp(in_dim, 1, hidden_dims)
        self.q2 = _mlp(in_dim, 1, hidden_dims)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(x), self.q2(x)


@dataclass
class SACConfig:
    obs_dim: int
    h_dim: int
    action_dim: int
    hidden_dims: tuple[int, ...] = (256, 256)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau_polyak: float = 0.005
    alpha: float = 0.2
    xi: float = 0.5  # xi_t -- FIXED for Stage 1, no annealing
    grad_clip_norm: float = 10.0  # standard SAC stabilizer: Eq task_reward's per-step (uncapped)
    # success bonus makes the true return grow ~unboundedly as the policy improves (gamma=0.99),
    # so the critic is regressing against a fast-moving, growing target early in training --
    # gradient-norm clipping is conventional practice for exactly this, and changes optimization
    # dynamics only, not the objective itself.
    device: str = "cpu"


class ReplayBuffer:
    """Off-policy transition storage: (obs, h, action, r_task, c_align, next_obs, next_h, done)."""

    def __init__(self, capacity: int, obs_dim: int, h_dim: int, action_dim: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.obs = torch.zeros(capacity, obs_dim, device=device)
        self.h = torch.zeros(capacity, h_dim, device=device)
        self.action = torch.zeros(capacity, action_dim, device=device)
        self.r_task = torch.zeros(capacity, 1, device=device)
        self.c_align = torch.zeros(capacity, 1, device=device)
        self.next_obs = torch.zeros(capacity, obs_dim, device=device)
        self.next_h = torch.zeros(capacity, h_dim, device=device)
        self.done = torch.zeros(capacity, 1, device=device)
        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add_batch(
        self,
        obs: torch.Tensor,
        h: torch.Tensor,
        action: torch.Tensor,
        r_task: torch.Tensor,
        c_align: torch.Tensor,
        next_obs: torch.Tensor,
        next_h: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        n = obs.shape[0]
        idx = (self._ptr + torch.arange(n, device=self.device)) % self.capacity
        self.obs[idx] = obs
        self.h[idx] = h
        self.action[idx] = action
        self.r_task[idx] = r_task.reshape(n, 1)
        self.c_align[idx] = c_align.reshape(n, 1)
        self.next_obs[idx] = next_obs
        self.next_h[idx] = next_h
        self.done[idx] = done.reshape(n, 1).float()
        self._ptr = (self._ptr + n) % self.capacity
        self._size = min(self._size + n, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = torch.randint(0, self._size, (batch_size,), device=self.device)
        return {
            "obs": self.obs[idx],
            "h": self.h[idx],
            "action": self.action[idx],
            "r_task": self.r_task[idx],
            "c_align": self.c_align[idx],
            "next_obs": self.next_obs[idx],
            "next_h": self.next_h[idx],
            "done": self.done[idx],
        }


class TwoCriticSAC:
    def __init__(self, cfg: SACConfig):
        self.cfg = cfg
        device = cfg.device

        self.actor = GaussianPolicy(cfg.obs_dim, cfg.action_dim, cfg.hidden_dims).to(device)

        self.q_task = TwinQ(cfg.obs_dim + cfg.action_dim, cfg.hidden_dims).to(device)
        self.q_task_target = copy.deepcopy(self.q_task)
        self.q_c = TwinQ(cfg.h_dim + cfg.action_dim, cfg.hidden_dims).to(device)
        self.q_c_target = copy.deepcopy(self.q_c)
        for p in self.q_task_target.parameters():
            p.requires_grad_(False)
        for p in self.q_c_target.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q_task_opt = torch.optim.Adam(self.q_task.parameters(), lr=cfg.critic_lr)
        self.q_c_opt = torch.optim.Adam(self.q_c.parameters(), lr=cfg.critic_lr)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        with torch.no_grad():
            action, _, mean_action = self.actor.sample(obs)
            return mean_action if deterministic else action

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        obs, h, action = batch["obs"], batch["h"], batch["action"]
        r_task, c_align = batch["r_task"], batch["c_align"]
        next_obs, next_h, done = batch["next_obs"], batch["next_h"], batch["done"]

        # --- Q_task: standard SAC critic on the task-and-safety reward (Eq task_reward) ---
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_obs)
            q_task_next = torch.min(*self.q_task_target(torch.cat([next_obs, next_action], dim=-1)))
            y_task = r_task + self.cfg.gamma * (1 - done) * (q_task_next - self.cfg.alpha * next_log_prob)

        q_task_1, q_task_2 = self.q_task(torch.cat([obs, action], dim=-1))
        loss_q_task = F.mse_loss(q_task_1, y_task) + F.mse_loss(q_task_2, y_task)
        self.q_task_opt.zero_grad()
        loss_q_task.backward()
        nn.utils.clip_grad_norm_(self.q_task.parameters(), self.cfg.grad_clip_norm)
        self.q_task_opt.step()

        # --- Q_C: ordinary value critic, reward channel = c_align (Eq cost_critic) ---
        with torch.no_grad():
            q_c_next = torch.min(*self.q_c_target(torch.cat([next_h, next_action], dim=-1)))
            y_c = c_align + self.cfg.gamma * (1 - done) * q_c_next  # no entropy term: Q_C predicts a cost-to-go, not a value

        q_c_1, q_c_2 = self.q_c(torch.cat([h, action], dim=-1))
        loss_q_c = F.mse_loss(q_c_1, y_c) + F.mse_loss(q_c_2, y_c)
        self.q_c_opt.zero_grad()
        loss_q_c.backward()
        nn.utils.clip_grad_norm_(self.q_c.parameters(), self.cfg.grad_clip_norm)
        self.q_c_opt.step()

        # --- actor: J_pi = E[Q_task - xi*Q_C - alpha*log pi] (Eq sac_actor_obj), maximized ---
        new_action, log_prob, _ = self.actor.sample(obs)
        q_task_pi = torch.min(*self.q_task(torch.cat([obs, new_action], dim=-1)))
        q_c_pi = torch.min(*self.q_c(torch.cat([h, new_action], dim=-1)))
        actor_loss = (self.cfg.alpha * log_prob - q_task_pi + self.cfg.xi * q_c_pi).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip_norm)
        self.actor_opt.step()

        self._polyak_update(self.q_task, self.q_task_target)
        self._polyak_update(self.q_c, self.q_c_target)

        return {
            "loss_q_task": loss_q_task.item(),
            "loss_q_c": loss_q_c.item(),
            "loss_actor": actor_loss.item(),
            "q_task_mean": q_task_pi.mean().item(),
            "q_c_mean": q_c_pi.mean().item(),
        }

    def _polyak_update(self, net: nn.Module, target_net: nn.Module) -> None:
        with torch.no_grad():
            for p, tp in zip(net.parameters(), target_net.parameters()):
                tp.mul_(1.0 - self.cfg.tau_polyak).add_(self.cfg.tau_polyak * p)
