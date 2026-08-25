import numpy as np

from physics_ot.dynamics.knob import KnobParams
from physics_ot.references.build_reference import build_reference, load_reference, save_reference


def min_jerk_theta(t: np.ndarray, theta0: float, theta_goal: float, duration: float) -> np.ndarray:
    tau = np.clip(t / duration, 0.0, 1.0)
    return theta0 + (theta_goal - theta0) * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)


def test_build_reference_token_count_and_monotonic_progress():
    theta0, theta_goal, duration = 0.0, 1.57, 2.0
    t_raw = np.linspace(0.0, duration, 200)
    theta_raw = min_jerk_theta(t_raw, theta0, theta_goal, duration)

    params = KnobParams(inertia=0.002, damping=0.02, stiffness=0.10, theta0=theta0, coulomb_torque=0.04)
    reference = build_reference(t_raw, theta_raw, params, theta_goal, num_tokens=64)

    assert reference.num_tokens == 64
    assert np.all(np.diff(reference.progress) >= -1e-6)
    assert abs(reference.progress[0]) < 1e-6
    assert abs(reference.progress[-1] - 1.0) < 1e-3


def test_build_reference_velocity_near_zero_at_endpoints():
    theta0, theta_goal, duration = 0.0, 1.57, 2.0
    t_raw = np.linspace(0.0, duration, 400)
    theta_raw = min_jerk_theta(t_raw, theta0, theta_goal, duration)

    params = KnobParams(inertia=0.002, damping=0.02, stiffness=0.10, theta0=theta0, coulomb_torque=0.04)
    reference = build_reference(t_raw, theta_raw, params, theta_goal, num_tokens=64)

    assert abs(reference.velocity[0]) < 0.05
    assert abs(reference.velocity[-1]) < 0.05
    peak_velocity = np.max(np.abs(reference.velocity))
    assert peak_velocity > 0.5


def test_reference_round_trips_through_npz(tmp_path):
    theta0, theta_goal, duration = 0.0, 1.57, 2.0
    t_raw = np.linspace(0.0, duration, 100)
    theta_raw = min_jerk_theta(t_raw, theta0, theta_goal, duration)

    params = KnobParams(inertia=0.002, damping=0.02, stiffness=0.10, theta0=theta0, coulomb_torque=0.04)
    reference = build_reference(t_raw, theta_raw, params, theta_goal, num_tokens=32)

    path = tmp_path / "ref.npz"
    save_reference(path, reference)
    loaded = load_reference(path)

    assert loaded.num_tokens == 32
    np.testing.assert_allclose(loaded.progress, reference.progress, atol=1e-6)
    np.testing.assert_allclose(loaded.wrench, reference.wrench, atol=1e-6)
    assert loaded.physics_params["inertia"] == params.inertia
