import torch

from poise_ot.mech_cost import (
    alignment_distance,
    debiased_sinkhorn_divergence,
    mechanics_ot_distance,
    mechanics_pairwise_cost,
    pointwise_mechanics_distance,
)
from poise_ot.token_builder import Token


def make_token(z: torch.Tensor, confidence: torch.Tensor | None = None, tau: torch.Tensor | None = None) -> Token:
    if confidence is None:
        confidence = torch.ones(z.shape[:-1])
    if tau is None:
        tau = z[..., 2]  # reuse impulse column as a stand-in raw torque for tests that don't care about it
    return Token(z=z, confidence=confidence, tau=tau)


def make_z(batch=1, t=8):
    s = torch.linspace(0.0, 1.0, t).expand(batch, t)
    v = torch.ones(batch, t)
    j = torch.zeros(batch, t)
    p = torch.zeros(batch, t)
    d = torch.zeros(batch, t)
    return torch.stack([s, v, j, p, d], dim=-1)


def test_mechanics_pairwise_cost_shape_and_zero_diagonal_for_self_cost():
    z = make_z(batch=2, t=6)
    conf = torch.ones(2, 6)
    cost = mechanics_pairwise_cost(z, conf, z, inertia=0.002)
    assert cost.shape == (2, 6, 6)
    diag = torch.diagonal(cost, dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-6)


def test_mechanics_ot_distance_near_zero_for_identical_trajectories():
    z = make_z(batch=1, t=64)
    conf = torch.ones(1, 64)
    distance = mechanics_ot_distance(z, conf, z, inertia=0.002, epsilon=0.01, num_iters=100)
    assert distance.item() < 0.02


def test_mechanics_ot_distance_increases_with_mismatch():
    z_ref = make_z(batch=1, t=32)
    conf = torch.ones(1, 32)
    z_roll = z_ref.clone()
    z_roll[..., 3] = z_roll[..., 3] + 5.0  # mismatch power channel

    d_matched = mechanics_ot_distance(z_ref, conf, z_ref, inertia=0.002, epsilon=0.01, num_iters=100)
    d_mismatched = mechanics_ot_distance(z_ref, conf, z_roll, inertia=0.002, epsilon=0.01, num_iters=100)
    assert d_mismatched.item() > d_matched.item()


def test_cost_terms_ablation_changes_the_cost():
    z_ref = make_z(batch=1, t=16)
    conf = torch.ones(1, 16)
    z_roll = z_ref.clone()
    z_roll[..., 4] = z_roll[..., 4] + 3.0  # only the dissipation channel differs

    cost_k = mechanics_pairwise_cost(z_ref, conf, z_roll, inertia=0.002, cost_terms="K")
    cost_full = mechanics_pairwise_cost(z_ref, conf, z_roll, inertia=0.002, cost_terms="K+J+P+D")
    # "K" ignores dissipation entirely, so a dissipation-only mismatch must not appear there.
    assert not torch.allclose(cost_k, cost_full)


def test_confidence_gates_impulse_power_dissipation_not_progress_or_velocity():
    z_ref = make_z(batch=1, t=8)
    z_roll = z_ref.clone()
    z_roll[..., 2] = z_roll[..., 2] + 1.0  # mismatch impulse only

    conf_full = torch.ones(1, 8)
    conf_zero = torch.zeros(1, 8)

    cost_full_conf = mechanics_pairwise_cost(z_ref, conf_full, z_roll, inertia=0.002)
    cost_zero_conf = mechanics_pairwise_cost(z_ref, conf_zero, z_roll, inertia=0.002)
    # zero confidence must remove the impulse-mismatch contribution.
    assert not torch.allclose(cost_full_conf, cost_zero_conf)


def test_debiased_divergence_is_zero_for_literally_identical_tokens():
    z = make_z(batch=1, t=16)
    token = make_token(z)
    divergence = debiased_sinkhorn_divergence(token, token, inertia=0.002, epsilon=0.01, num_iters=100)
    assert abs(divergence.item()) < 1e-5


def test_pointwise_mechanics_distance_zero_for_identical():
    z = make_z(batch=1, t=10)
    conf = torch.ones(1, 10)
    distance = pointwise_mechanics_distance(z, conf, z, inertia=0.002)
    assert torch.allclose(distance, torch.zeros_like(distance), atol=1e-6)


def test_alignment_distance_dispatch_differs_across_cost_modes():
    """Ablation-wiring sanity, as a unit test: swapping cost_mode must change
    the actual computed value, not just relabel it."""
    z_ref = make_z(batch=1, t=32)
    z_roll = make_z(batch=1, t=16)
    z_roll[..., 1] = z_roll[..., 1] + 0.7  # mismatch velocity so all modes see *some* signal

    ref = make_token(z_ref)
    roll = make_token(z_roll)

    d_mech = alignment_distance(ref, roll, inertia=0.002, cost_mode="mechanics_ot")
    d_euclid = alignment_distance(ref, roll, inertia=0.002, cost_mode="euclidean_ot")
    d_point = alignment_distance(ref, roll, inertia=0.002, cost_mode="pointwise_mechanics")

    values = {d_mech.item(), d_euclid.item(), d_point.item()}
    assert len(values) == 3  # all three genuinely different, not accidental ties


def test_alignment_distance_rejects_unknown_mode():
    z = make_z(batch=1, t=4)
    token = make_token(z)
    try:
        alignment_distance(token, token, inertia=0.002, cost_mode="not_a_real_mode")
        assert False, "expected ValueError"
    except ValueError:
        pass
