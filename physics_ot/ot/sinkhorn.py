from __future__ import annotations

import torch


def batched_sinkhorn(
    cost: torch.Tensor,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    epsilon: float = 0.05,
    num_iters: int = 20,
) -> torch.Tensor:
    """Batched entropic optimal transport via log-domain-stabilized Sinkhorn (section 10).

    Solves, independently per batch element:
        Gamma* = argmin_{Gamma in Pi(a,b)} <Gamma, C> - epsilon * H(Gamma)

    Args:
        cost: ground cost matrix, shape [..., T_H, T_R].
        a: source marginal, shape [..., T_H]; defaults to uniform (1/T_H).
        b: target marginal, shape [..., T_R]; defaults to uniform (1/T_R).
        epsilon: entropic regularization strength.
        num_iters: number of Sinkhorn iterations.

    Returns:
        Transport plan Gamma, shape [..., T_H, T_R]. Gradients are not required
        (this is used as a reward signal, not a loss) so callers should wrap
        calls in ``torch.no_grad()`` for efficiency.
    """
    t_h, t_r = cost.shape[-2], cost.shape[-1]
    batch_shape = cost.shape[:-2]

    if a is None:
        a = cost.new_full(batch_shape + (t_h,), 1.0 / t_h)
    if b is None:
        b = cost.new_full(batch_shape + (t_r,), 1.0 / t_r)

    log_a = torch.log(a.clamp_min(1e-30))
    log_b = torch.log(b.clamp_min(1e-30))

    f = torch.zeros_like(a)
    g = torch.zeros_like(b)

    for _ in range(num_iters):
        m_f = (g.unsqueeze(-2) - cost) / epsilon  # [..., T_H, T_R]
        f = epsilon * (log_a - torch.logsumexp(m_f, dim=-1))

        m_g = (f.unsqueeze(-1) - cost) / epsilon  # [..., T_H, T_R]
        g = epsilon * (log_b - torch.logsumexp(m_g, dim=-2))

    log_gamma = (f.unsqueeze(-1) + g.unsqueeze(-2) - cost) / epsilon
    return torch.exp(log_gamma)


def transport_cost(gamma: torch.Tensor, cost: torch.Tensor) -> torch.Tensor:
    """The Physics-OT distance D_POT = <Gamma*, C> (section 10), reduced over the last two dims."""
    return (gamma * cost).sum(dim=(-2, -1))
