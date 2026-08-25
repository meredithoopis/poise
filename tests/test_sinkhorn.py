import torch

from physics_ot.ot.cost import CostWeights, normalize_state, physics_ot_cost
from physics_ot.ot.sinkhorn import batched_sinkhorn, transport_cost
from physics_ot.ot.temporal_ot import TemporalPhysicsOT


def test_sinkhorn_transport_plan_matches_marginals():
    torch.manual_seed(0)
    batch, t_h, t_r = 3, 12, 20
    cost = torch.rand(batch, t_h, t_r)
    gamma = batched_sinkhorn(cost, epsilon=0.05, num_iters=200)
    row_marginal = gamma.sum(dim=-1)
    col_marginal = gamma.sum(dim=-2)
    assert torch.allclose(row_marginal, torch.full((batch, t_h), 1.0 / t_h), atol=1e-3)
    assert torch.allclose(col_marginal, torch.full((batch, t_r), 1.0 / t_r), atol=1e-3)


def test_identical_trajectories_have_near_zero_pot_distance():
    t = torch.linspace(0.0, 1.0, 64)
    progress = t.clone()
    velocity = torch.ones_like(t)
    wrench = torch.zeros_like(t)

    # Entropic OT has an inherent self-distance bias (Sinkhorn divergence is
    # D_eps(a,b) - 0.5*D_eps(a,a) - 0.5*D_eps(b,b), not D_eps(a,b) alone), so
    # even identical trajectories give a small but nonzero D_POT here.
    pot = TemporalPhysicsOT(sigma_v=1.0, sigma_tau=1.0, epsilon=0.01, num_iters=100)
    distance = pot.distance(
        progress.unsqueeze(0), velocity.unsqueeze(0), wrench.unsqueeze(0),
        progress.unsqueeze(0), velocity.unsqueeze(0), wrench.unsqueeze(0),
    )
    assert distance.item() < 0.02


def test_pot_distance_increases_with_mismatch():
    t = torch.linspace(0.0, 1.0, 64)
    progress = t.clone()
    velocity = torch.ones_like(t)
    wrench = torch.zeros_like(t)

    mismatched_wrench = wrench + 5.0

    pot = TemporalPhysicsOT(sigma_v=1.0, sigma_tau=1.0, epsilon=0.01, num_iters=100)
    d_matched = pot.distance(
        progress.unsqueeze(0), velocity.unsqueeze(0), wrench.unsqueeze(0),
        progress.unsqueeze(0), velocity.unsqueeze(0), wrench.unsqueeze(0),
    )
    d_mismatched = pot.distance(
        progress.unsqueeze(0), velocity.unsqueeze(0), wrench.unsqueeze(0),
        progress.unsqueeze(0), velocity.unsqueeze(0), mismatched_wrench.unsqueeze(0),
    )
    assert d_mismatched.item() > d_matched.item()


def test_cost_matrix_shape_and_symmetry_of_self_cost_diagonal():
    z = torch.randn(2, 5, 3)
    cost = physics_ot_cost(z, z, CostWeights(time=0.0))
    assert cost.shape == (2, 5, 5)
    diag = torch.diagonal(cost, dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-6)


def test_normalize_state_scales_columns():
    progress = torch.tensor([0.5])
    velocity = torch.tensor([2.0])
    wrench = torch.tensor([4.0])
    z = normalize_state(progress, velocity, wrench, sigma_v=2.0, sigma_tau=4.0)
    assert torch.allclose(z, torch.tensor([[0.5, 1.0, 1.0]]))


def test_transport_cost_matches_manual_reduction():
    gamma = torch.rand(2, 4, 5)
    cost = torch.rand(2, 4, 5)
    manual = (gamma * cost).sum(dim=(-2, -1))
    assert torch.allclose(transport_cost(gamma, cost), manual)
