import torch

from physics_ot.dynamics.knob import KnobParams
from poise_ot.alignment import AlignmentConfig, RollingAlignment
from poise_ot.token_builder import Token


def make_reference_token(t_len=64) -> tuple[Token, torch.Tensor]:
    ref_time = torch.linspace(0.0, 6.0, t_len)
    s = torch.linspace(0.0, 1.0, t_len)
    v = torch.ones(t_len)
    j = torch.zeros(t_len)
    p = torch.zeros(t_len)
    d = torch.zeros(t_len)
    z = torch.stack([s, v, j, p, d], dim=-1)
    token = Token(z=z, confidence=torch.ones(t_len), tau=torch.zeros(t_len))
    return token, ref_time


def make_params() -> KnobParams:
    return KnobParams(inertia=0.002, damping=0.02, stiffness=0.10, theta0=0.0, coulomb_torque=0.04, v_eps=0.05)


def test_h_dim_matches_flattened_window_plus_ref_token():
    token, ref_time = make_reference_token()
    cfg = AlignmentConfig(window_h=8)
    alignment = RollingAlignment(token, ref_time, make_params(), num_envs=3, device="cpu", cfg=cfg)
    assert alignment.h_dim == 5 + 8 * 5


def test_step_alignment_shapes():
    token, ref_time = make_reference_token()
    cfg = AlignmentConfig(window_h=8)
    num_envs = 3
    alignment = RollingAlignment(token, ref_time, make_params(), num_envs=num_envs, device="cpu", cfg=cfg)

    for _ in range(8):
        progress = torch.full((num_envs,), 0.5)
        theta_dot = torch.ones(num_envs)
        tau = torch.zeros(num_envs)
        alignment.push(progress, theta_dot, tau, dt=0.1)

    obs = torch.zeros(num_envs, 3)
    elapsed_time = torch.full((num_envs,), 1.0)
    c_align, h_t = alignment.step_alignment(obs, elapsed_time)

    assert c_align.shape == (num_envs,)
    assert h_t.shape == (num_envs, alignment.h_dim + 3)  # +3 for the obs channels
    assert torch.isfinite(c_align).all()
    assert torch.isfinite(h_t).all()


def test_reset_idx_clears_only_selected_envs():
    token, ref_time = make_reference_token()
    cfg = AlignmentConfig(window_h=4)
    num_envs = 3
    alignment = RollingAlignment(token, ref_time, make_params(), num_envs=num_envs, device="cpu", cfg=cfg)

    for _ in range(4):
        alignment.push(torch.full((num_envs,), 0.7), torch.ones(num_envs), torch.zeros(num_envs), dt=0.1)

    assert torch.all(alignment._filled == 4)
    alignment.reset_idx(torch.tensor([1]))
    assert alignment._filled[1].item() == 0
    assert alignment._filled[0].item() == 4
    assert alignment._filled[2].item() == 4


def test_alignment_progress_reward_is_ablation_only_and_correct_sign():
    prev = torch.tensor([1.0, 0.5])
    curr = torch.tensor([0.8, 0.5])
    reward = RollingAlignment.alignment_progress_reward(prev, curr)
    assert torch.allclose(reward, torch.tensor([0.2, 0.0]))  # positive when alignment improves
