import torch

from physics_ot.dynamics.knob import KnobParams
from poise_ot.token_builder import build_token, windowed_impulse


def make_params() -> KnobParams:
    return KnobParams(inertia=0.002, damping=0.02, stiffness=0.10, theta0=0.0, coulomb_torque=0.04, v_eps=0.05)


def test_windowed_impulse_reduces_to_single_sample_for_window_1():
    tau = torch.tensor([1.0, 2.0, 3.0, 4.0])
    dt = 0.1
    assert torch.allclose(windowed_impulse(tau, dt, window_h=1), tau * dt)


def test_windowed_impulse_sums_neighbors_and_clamps_at_edges():
    tau = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    dt = 1.0
    # window_h=3, centered: interior points sum 3 neighbors; edges clamp (replicate) missing ones.
    result = windowed_impulse(tau, dt, window_h=3)
    assert torch.allclose(result, torch.full((5,), 3.0))


def test_build_token_shapes_and_finite():
    params = make_params()
    t = torch.linspace(0.0, 1.0, 64)
    progress = t.clone()
    theta_dot = torch.ones_like(t)
    tau = torch.zeros_like(t)

    token = build_token(progress, theta_dot, tau, params, dt=1.0 / 63, window_h=5)

    assert token.z.shape == (64, 5)
    assert token.confidence.shape == (64,)
    assert torch.isfinite(token.z).all()
    assert torch.allclose(token.confidence, torch.ones(64))


def test_build_token_matches_manual_power_and_dissipation():
    params = make_params()
    progress = torch.tensor([0.5])
    theta_dot = torch.tensor([0.3])
    tau = torch.tensor([0.02])

    token = build_token(progress, theta_dot, tau, params, dt=1.0, window_h=1)

    expected_power = tau * theta_dot
    expected_diss = params.damping * theta_dot**2 + params.coulomb_torque * theta_dot * torch.tanh(
        theta_dot / params.v_eps
    )
    assert torch.allclose(token.z[..., 3], expected_power)
    assert torch.allclose(token.z[..., 4], expected_diss)
    assert torch.allclose(token.z[..., 2], tau * 1.0)  # J_tau with window_h=1 == tau*dt
    assert torch.allclose(token.tau, tau)


def test_build_token_respects_explicit_confidence():
    params = make_params()
    progress = torch.linspace(0.0, 1.0, 8)
    theta_dot = torch.ones(8)
    tau = torch.zeros(8)
    confidence = torch.linspace(0.1, 1.0, 8)

    token = build_token(progress, theta_dot, tau, params, dt=0.1, confidence=confidence)
    assert torch.allclose(token.confidence, confidence)
