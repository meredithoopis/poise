import torch

from physics_ot.dynamics.knob import KnobDynamics, KnobParams


def make_params(**overrides):
    defaults = dict(inertia=0.002, damping=0.02, stiffness=0.10, theta0=0.0, coulomb_torque=0.04, v_eps=0.05)
    defaults.update(overrides)
    return KnobParams(**defaults)


def test_zero_velocity_and_acceleration_leaves_only_spring_term():
    params = make_params(theta0=0.3)
    dynamics = KnobDynamics(params)
    theta = torch.tensor([0.3, 0.8, -0.2])
    zero = torch.zeros_like(theta)
    tau = dynamics.required_generalized_force(theta, zero, zero)
    expected = params.stiffness * (theta - params.theta0)
    assert torch.allclose(tau, expected, atol=1e-6)


def test_required_force_matches_closed_form():
    params = make_params()
    dynamics = KnobDynamics(params)
    theta = torch.tensor([0.5])
    theta_dot = torch.tensor([1.2])
    theta_ddot = torch.tensor([-0.3])
    tau = dynamics.required_generalized_force(theta, theta_dot, theta_ddot)
    expected = (
        params.inertia * theta_ddot
        + params.damping * theta_dot
        + params.stiffness * (theta - params.theta0)
        + params.coulomb_torque * torch.tanh(theta_dot / params.v_eps)
    )
    assert torch.allclose(tau, expected, atol=1e-6)


def test_coulomb_friction_saturates_and_is_odd():
    params = make_params(inertia=0.0, damping=0.0, stiffness=0.0)
    dynamics = KnobDynamics(params)
    fast = torch.tensor([10.0])
    slow_negative = torch.tensor([-10.0])
    zero = torch.zeros(1)
    tau_fast = dynamics.required_generalized_force(zero, fast, zero)
    tau_slow_negative = dynamics.required_generalized_force(zero, slow_negative, zero)
    assert torch.allclose(tau_fast, torch.tensor([params.coulomb_torque]), atol=1e-3)
    assert torch.allclose(tau_slow_negative, -tau_fast, atol=1e-6)


def test_passive_torque_is_required_force_without_inertia_term():
    params = make_params()
    dynamics = KnobDynamics(params)
    theta = torch.tensor([0.2, -0.4])
    theta_dot = torch.tensor([0.5, -0.1])
    zero_accel = torch.zeros_like(theta)
    required = dynamics.required_generalized_force(theta, theta_dot, zero_accel)
    passive = dynamics.passive_torque(theta, theta_dot)
    assert torch.allclose(passive, -required, atol=1e-6)


def test_broadcasts_over_batch_and_time_dims():
    params = make_params()
    dynamics = KnobDynamics(params)
    theta = torch.randn(4, 8)
    theta_dot = torch.randn(4, 8)
    theta_ddot = torch.randn(4, 8)
    tau = dynamics.required_generalized_force(theta, theta_dot, theta_ddot)
    assert tau.shape == (4, 8)
