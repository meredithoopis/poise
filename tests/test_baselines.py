import torch

from physics_ot.ot.baselines import dtw_distance, ot_state_distance, pointwise_l2_distance, pointwise_physics_distance
from physics_ot.ot.cost import normalize_state


def test_pointwise_l2_zero_for_identical_trajectories():
    t = torch.linspace(0.0, 1.0, 40).unsqueeze(0)
    velocity = torch.ones_like(t)
    d = pointwise_l2_distance(t, velocity, t, velocity)
    assert d.item() < 1e-6


def test_pointwise_physics_penalizes_wrench_mismatch():
    t = torch.linspace(0.0, 1.0, 40).unsqueeze(0)
    velocity = torch.ones_like(t)
    wrench = torch.zeros_like(t)
    d_matched = pointwise_physics_distance(t, velocity, wrench, t, velocity, wrench)
    d_mismatched = pointwise_physics_distance(t, velocity, wrench, t, velocity, wrench + 3.0)
    assert d_matched.item() < 1e-6
    assert d_mismatched.item() > d_matched.item()


def test_ot_state_distance_ignores_wrench_axis():
    t = torch.linspace(0.0, 1.0, 32).unsqueeze(0)
    velocity = torch.ones_like(t)
    d = ot_state_distance(t, velocity, t, velocity, epsilon=0.01, num_iters=100)
    assert d.item() < 0.02


def test_dtw_distance_zero_for_identical_trajectories():
    t = torch.linspace(0.0, 1.0, 10).unsqueeze(0)
    velocity = torch.ones_like(t)
    wrench = torch.zeros_like(t)
    z = normalize_state(t, velocity, wrench, sigma_v=1.0, sigma_tau=1.0)
    d = dtw_distance(z, z)
    assert d.item() < 1e-6


def test_dtw_distance_increases_with_mismatch():
    t = torch.linspace(0.0, 1.0, 10).unsqueeze(0)
    velocity = torch.ones_like(t)
    wrench = torch.zeros_like(t)
    z_ref = normalize_state(t, velocity, wrench, sigma_v=1.0, sigma_tau=1.0)
    z_roll = normalize_state(t, velocity, wrench + 4.0, sigma_v=1.0, sigma_tau=1.0)
    assert dtw_distance(z_roll, z_ref).item() > 0.0
