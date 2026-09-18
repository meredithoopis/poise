import torch

from physics_ot.dynamics.knob import KnobParams
from poise_ot.reference_generator import constant_ramp_torque, integrated_reference, scripted_reference


def make_params() -> KnobParams:
    return KnobParams(inertia=0.002, damping=0.02, stiffness=0.10, theta0=0.0, coulomb_torque=0.04, v_eps=0.05)


def test_scripted_reference_token_shape_and_finite():
    params = make_params()
    ref, token = scripted_reference(params, theta_goal=1.57, reach_duration=2.0, total_duration=6.0, num_tokens=64)

    assert ref.num_tokens == 64
    assert token.z.shape == (64, 5)
    assert torch.isfinite(token.z).all()


def test_scripted_reference_progress_holds_near_one_after_reach():
    params = make_params()
    ref, token = scripted_reference(params, theta_goal=1.57, reach_duration=2.0, total_duration=6.0, num_tokens=64)
    # tokens sampled after the reach phase should be at (near) full progress.
    late_progress = token.z[-5:, 0]
    assert torch.all(late_progress > 0.95)


def test_integrated_reference_matches_its_own_dynamics():
    """The reference is produced by literally the same forward integration
    knob_env.py uses -- token progress should be monotone-ish and finite,
    and running it twice with the same torque profile is deterministic."""
    params = make_params()
    torque_fn = constant_ramp_torque(magnitude=0.08, ramp_time=1.0)

    ref1, token1 = integrated_reference(params, theta_goal=1.57, torque_fn=torque_fn, total_duration=3.0, num_tokens=32)
    ref2, token2 = integrated_reference(params, theta_goal=1.57, torque_fn=torque_fn, total_duration=3.0, num_tokens=32)

    assert torch.isfinite(token1.z).all()
    assert torch.allclose(token1.z, token2.z)  # deterministic given the same torque profile


def test_integrated_reference_moves_toward_goal_under_positive_torque():
    params = make_params()
    torque_fn = constant_ramp_torque(magnitude=0.08, ramp_time=0.5)
    ref, token = integrated_reference(params, theta_goal=1.57, torque_fn=torque_fn, total_duration=4.0, num_tokens=32)
    assert token.z[-1, 0] > token.z[0, 0]  # progress increases over the trajectory
