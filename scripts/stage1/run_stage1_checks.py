"""The five Stage-1 sanity checks (token/cost/gradient/learning-signal/
ablation-wiring) -- this script IS the "definition of done" artifact for
Stage 1, not just a training script. Each check prints PASS/FAIL with the
actual numbers behind the verdict (not just a boolean) and, when matplotlib
is available, saves a plot to ``logs/stage1/``.

Usage:
    python scripts/stage1/run_stage1_checks.py --check all
    python scripts/stage1/run_stage1_checks.py --check 3 --episodes 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
STAGE1_DIR = Path(__file__).resolve().parent
if str(STAGE1_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE1_DIR))

from train_stage1 import build_params, load_configs, train  # noqa: E402

from poise_ot.alignment import AlignmentConfig, RollingAlignment  # noqa: E402
from poise_ot.knob_env import KnobEnv, KnobEnvConfig  # noqa: E402
from poise_ot.mech_cost import MechCostScales, MechCostWeights, debiased_sinkhorn_divergence, mechanics_ot_distance  # noqa: E402
from poise_ot.reference_generator import scripted_reference  # noqa: E402
from poise_ot.token_builder import Token  # noqa: E402

LOG_DIR = REPO_ROOT / "logs" / "stage1"


def _maybe_plot(fig_fn, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  (matplotlib not installed -- skipping plot at {path})")
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fig = fig_fn(plt)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plot -> {path}")


def _build_reference(knob_cfg_path=None, poise_cfg_path=None):
    knob_cfg_path = knob_cfg_path or (REPO_ROOT / "configs" / "knob.yaml")
    poise_cfg_path = poise_cfg_path or (REPO_ROOT / "configs" / "poise_ot.yaml")
    knob_cfg, poise_cfg = load_configs(knob_cfg_path, poise_cfg_path)
    params = build_params(knob_cfg)
    theta_goal = knob_cfg["knob"]["theta_goal"]
    ref, token = scripted_reference(
        params, theta_goal,
        num_tokens=knob_cfg["reference"]["num_tokens"],
        smoothing=knob_cfg["reference"]["smoothing"],
        impulse_window=poise_cfg["token"]["impulse_window"],
    )
    return knob_cfg, poise_cfg, params, theta_goal, ref, token


def check_1_token_sanity(args) -> bool:
    """z_t finite and J_tau/P within an explicit order-of-magnitude bound
    derived from the knob's own physical parameters -- collected from a few
    early (random-action) rollouts, not a trained policy."""
    print("\n=== Check 1: token sanity ===")
    knob_cfg, poise_cfg, params, theta_goal, ref, ref_token = _build_reference()

    env_cfg = KnobEnvConfig(
        params=params, theta_goal=theta_goal, success_threshold=knob_cfg["knob"]["success_threshold"],
        physics_dt=knob_cfg["simulation"]["physics_dt"], control_decimation=knob_cfg["simulation"]["control_decimation"],
        episode_seconds=knob_cfg["simulation"]["episode_seconds"], tau_max=1.0,
    )
    num_envs = 8
    env = KnobEnv(env_cfg, num_envs=num_envs)
    align_cfg = AlignmentConfig(
        window_h=poise_cfg["alignment"]["window_h"], impulse_window=poise_cfg["token"]["impulse_window"],
    )
    alignment = RollingAlignment(ref_token, torch.from_numpy(ref.time.astype(np.float32)), params, num_envs, "cpu", align_cfg)

    obs, _ = env.reset()
    theta_dot_max_estimate = env_cfg.tau_max / params.inertia * env.dt * 20  # crude reachable-velocity bound over a short horizon
    power_bound = 5.0 * env_cfg.tau_max * max(theta_dot_max_estimate, 1.0)

    all_finite = True
    max_abs_j, max_abs_p = 0.0, 0.0
    for _ in range(30):
        action = torch.rand(num_envs, 1) * 2 - 1
        obs, reward, terminated, truncated, info = env.step(action)
        token = alignment.push(info["progress"], info["theta_dot"], info["tau"], dt=env.dt)
        all_finite &= bool(torch.isfinite(token.z).all())
        max_abs_j = max(max_abs_j, token.z[..., 2].abs().max().item())
        max_abs_p = max(max_abs_p, token.z[..., 3].abs().max().item())

    within_bound = max_abs_p < power_bound
    passed = all_finite and within_bound
    print(f"  all z_t finite: {all_finite}")
    print(f"  max |J_tau| observed: {max_abs_j:.6g}")
    print(f"  max |P| observed: {max_abs_p:.6g}  (bound: {power_bound:.6g})")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def check_2_cost_sanity(args) -> bool:
    """D_POISE(rho*, rho*) ~= 0 (self-distance near zero); reports the
    debiased Sinkhorn divergence alongside the raw distance so the known
    entropic self-distance bias is visible, not hidden."""
    print("\n=== Check 2: cost sanity ===")
    _, poise_cfg, params, _, _, ref_token = _build_reference()
    mc = poise_cfg["mech_cost"]
    weights = MechCostWeights(progress=mc["progress"], velocity=mc["velocity"], impulse=mc["impulse"], power=mc["power"], dissipation=mc["dissipation"], time=mc["time"])
    scales = MechCostScales(E0=mc["E0"], P0=mc["P0"], PD0=mc["PD0"])

    z = ref_token.z.unsqueeze(0)
    conf = ref_token.confidence.unsqueeze(0)
    d_self = mechanics_ot_distance(z, conf, z, params.inertia, weights, scales, mc["cost_terms"], poise_cfg["alignment"]["epsilon"], poise_cfg["alignment"]["num_iters"])
    debiased = debiased_sinkhorn_divergence(ref_token, ref_token, params.inertia, weights, scales, mc["cost_terms"], poise_cfg["alignment"]["epsilon"], poise_cfg["alignment"]["num_iters"])

    eps_self = args.self_distance_threshold
    passed = d_self.item() < eps_self
    print(f"  raw D_POISE(rho*, rho*): {d_self.item():.6g}")
    print(f"  debiased Sinkhorn divergence S_eps(rho*, rho*): {debiased.item():.6g}  (should be ~exactly 0)")
    print(f"  threshold: {eps_self}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def check_3_gradient_sanity(args) -> bool:
    """Q_task and Q_C critic losses both trend down (not diverging/flat)
    over the first few thousand update steps."""
    print("\n=== Check 3: gradient sanity ===")
    result = train(num_envs=args.num_envs, episodes=args.episodes, device=args.device, seed=args.seed, verbose=False)
    history = result["history"]

    def slope(values: list[float]) -> float:
        if len(values) < 2:
            return float("nan")
        x = np.arange(len(values))
        return float(np.polyfit(x, values, 1)[0])

    slope_q_task = slope(history["loss_q_task"])
    slope_q_c = slope(history["loss_q_c"])
    passed = slope_q_task < 0 and slope_q_c < 0
    print(f"  updates observed: {len(history['loss_q_task'])}")
    print(f"  Q_task loss trend (slope): {slope_q_task:.6g}")
    print(f"  Q_C loss trend (slope): {slope_q_c:.6g}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    _maybe_plot(
        lambda plt: _loss_figure(plt, history),
        LOG_DIR / "check3_gradient_sanity.png",
    )
    return passed


def _loss_figure(plt, history):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["step"], history["loss_q_task"])
    axes[0].set_title("Q_task loss")
    axes[0].set_xlabel("step")
    axes[1].plot(history["step"], history["loss_q_c"])
    axes[1].set_title("Q_C loss")
    axes[1].set_xlabel("step")
    fig.tight_layout()
    return fig


def check_4_learning_signal_sanity(args) -> bool:
    """With xi_t fixed at configs/poise_ot.yaml's mid-range value (never
    annealed -- TwoCriticSAC has no annealing code path at all, so this is
    true by construction), trained success rate should climb above a
    random-policy baseline within a modest episode budget."""
    print("\n=== Check 4: learning-signal sanity ===")
    trained = train(num_envs=args.num_envs, episodes=args.episodes, device=args.device, seed=args.seed, verbose=False)
    random_baseline = train(num_envs=args.num_envs, episodes=args.episodes, device=args.device, seed=args.seed, random_policy=True, verbose=False)

    def final_success_rate(result, tail=50):
        successes = result["history"]["episode_success"]
        return float(np.mean(successes[-tail:])) if successes else float("nan")

    trained_rate = final_success_rate(trained)
    random_rate = final_success_rate(random_baseline)
    passed = trained_rate > random_rate
    print(f"  trained success rate (last 50 episodes): {trained_rate:.3f}")
    print(f"  random-policy success rate (last 50 episodes): {random_rate:.3f}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    _maybe_plot(
        lambda plt: _success_curve_figure(plt, trained["history"]["episode_success"], random_baseline["history"]["episode_success"]),
        LOG_DIR / "check4_learning_signal_sanity.png",
    )
    return passed


def _success_curve_figure(plt, trained_success, random_success):
    def rolling_mean(values, window=10):
        values = np.asarray(values, dtype=float)
        if len(values) < window:
            return values
        return np.convolve(values, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rolling_mean(trained_success), label="trained (SAC)")
    ax.plot(rolling_mean(random_success), label="random policy")
    ax.set_xlabel("episode")
    ax.set_ylabel("rolling success rate")
    ax.legend()
    fig.tight_layout()
    return fig


def check_5_ablation_wiring_sanity(args) -> bool:
    """Swapping cost_mode (mechanics_ot / euclidean_ot / pointwise_mechanics)
    must produce genuinely different Q_C training curves, confirming the
    swap actually changes what Q_C is trained on."""
    print("\n=== Check 5: ablation wiring sanity ===")
    modes = ["mechanics_ot", "euclidean_ot", "pointwise_mechanics"]
    histories = {}
    for mode in modes:
        result = train(num_envs=args.num_envs, episodes=args.episodes, cost_mode=mode, device=args.device, seed=args.seed, verbose=False)
        histories[mode] = result["history"]["loss_q_c"]

    n = min(len(histories[m]) for m in modes)
    arrays = {m: np.asarray(histories[m][:n]) for m in modes}
    identical_pairs = []
    for i, a in enumerate(modes):
        for b in modes[i + 1 :]:
            if n > 0 and np.allclose(arrays[a], arrays[b]):
                identical_pairs.append((a, b))

    passed = n > 0 and len(identical_pairs) == 0
    for mode in modes:
        print(f"  {mode}: {n} Q_C-loss updates, final={histories[mode][-1] if histories[mode] else float('nan'):.6g}")
    if identical_pairs:
        print(f"  WARNING: identical curves for {identical_pairs} -- the swap is not actually changing anything")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    _maybe_plot(lambda plt: _ablation_figure(plt, histories), LOG_DIR / "check5_ablation_wiring_sanity.png")
    return passed


def _ablation_figure(plt, histories):
    fig, ax = plt.subplots(figsize=(6, 4))
    for mode, values in histories.items():
        ax.plot(values, label=mode)
    ax.set_xlabel("update step")
    ax.set_ylabel("Q_C loss")
    ax.legend()
    fig.tight_layout()
    return fig


CHECKS = {
    "1": check_1_token_sanity,
    "2": check_2_cost_sanity,
    "3": check_3_gradient_sanity,
    "4": check_4_learning_signal_sanity,
    "5": check_5_ablation_wiring_sanity,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", type=str, default="all", choices=list(CHECKS.keys()) + ["all"])
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self_distance_threshold", type=float, default=0.02)
    args = parser.parse_args()

    checks_to_run = list(CHECKS.keys()) if args.check == "all" else [args.check]
    results = {}
    for key in checks_to_run:
        results[key] = CHECKS[key](args)

    print("\n=== Stage 1 summary ===")
    for key, passed in results.items():
        print(f"  check {key}: {'PASS' if passed else 'FAIL'}")
    all_passed = all(results.values())
    print(f"  ALL: {'PASS' if all_passed else 'FAIL'}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "summary.json").write_text(json.dumps({k: bool(v) for k, v in results.items()}, indent=2))

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
