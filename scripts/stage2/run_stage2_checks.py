"""The four Stage-2 sanity checks (trajectory/differentiation/token/
confidence-gate) -- run against either real HOI4D clips (once available) or
pre-processed npz clips in the simple intermediate format (for smoke-testing
this script itself without the dataset; see video_loader.py's module
docstring on the two loading paths).

Unlike Stage 1's checks, #1 is explicitly NOT auto-pass/fail (the spec asks
for visual inspection); #2-4 do have an automatic verdict, reported with the
actual numbers behind it.

Usage (against real HOI4D, once POISE_OT_HOI4D_ROOT is set):
    python scripts/stage2/run_stage2_checks.py --category Safe --num_clips 5

Usage (against pre-processed npz clips, e.g. for a dry run / this script's
own test coverage):
    python scripts/stage2/run_stage2_checks.py --npz_clips clip1.npz clip2.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_stage2_pipeline import load_assumed_params  # noqa: E402

from poise_ot.video.confidence_gate import compute_confidence  # noqa: E402
from poise_ot.video.inverse_dynamics import target_conditioned_torque  # noqa: E402
from poise_ot.video.smoothing import smooth_clip  # noqa: E402
from poise_ot.video.video_loader import load_clip_npz, load_clip_raw, list_clips  # noqa: E402

LOG_DIR = REPO_ROOT / "logs" / "stage2"


def _maybe_plot(fig_fn, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"    (matplotlib not installed -- skipping plot at {path})")
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fig = fig_fn(plt)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved plot -> {path}")


def load_clips(args):
    if args.npz_clips:
        return [load_clip_npz(p, category=args.category) for p in args.npz_clips]
    clip_dirs = list_clips(args.category)[: args.num_clips]
    clips = []
    for clip_dir in clip_dirs:
        try:
            clips.append(load_clip_raw(clip_dir, category=args.category))
        except (FileNotFoundError, KeyError) as exc:
            print(f"  SKIPPED {clip_dir.name}: {exc}")
    return clips


def run_checks_for_clip(clip, params, args) -> dict:
    smoothing_result = smooth_clip(clip.t, clip.theta, smoothing=args.smoothing)
    ind_result = target_conditioned_torque(smoothing_result, params, num_tokens=args.num_tokens, verbose=False)
    confidence = compute_confidence(smoothing_result, ind_result, params, residual_scale=args.residual_scale)

    # --- Check 1: trajectory sanity (visual, not auto-pass/fail) ---
    fitted_at_raw = smoothing_result.trajectory.position(clip.t)
    print(f"    check1 raw-vs-smoothed RMS gap: {np.sqrt(np.mean((clip.theta - fitted_at_raw) ** 2)):.5g}")
    _maybe_plot(
        lambda plt: _trajectory_figure(plt, clip.t, clip.theta, ind_result.t, ind_result.theta),
        LOG_DIR / f"{clip.clip_id}_check1_trajectory.png",
    )

    # --- Check 2: differentiation sanity ---
    theta_noise_floor = float(np.std(np.diff(clip.theta))) if len(clip.theta) > 1 else 0.0
    theta_ddot_variance = float(np.var(ind_result.theta_ddot))
    threshold = args.theta_ddot_noise_multiplier * max(theta_noise_floor, 1e-9)
    diff_flagged = theta_ddot_variance > threshold
    print(f"    check2 theta_ddot variance={theta_ddot_variance:.5g}, flag threshold={threshold:.5g}, "
          f"{'FLAGGED (likely under-smoothed)' if diff_flagged else 'ok'}")
    _maybe_plot(
        lambda plt: _differentiation_figure(plt, ind_result.t, ind_result.theta_dot, ind_result.theta_ddot),
        LOG_DIR / f"{clip.clip_id}_check2_differentiation.png",
    )

    # --- Check 3: token plausibility sanity ---
    j_tau_max = float(np.abs(ind_result.tau_star).max())  # proxy for J_tau's order of magnitude
    p_max = float(np.abs(ind_result.tau_star * ind_result.theta_dot).max())
    j_lo, j_hi = args.plausible_impulse_range
    p_lo, p_hi = args.plausible_power_range
    token_plausible = (j_lo <= j_tau_max <= j_hi) and (p_lo <= p_max <= p_hi)
    print(f"    check3 max|tau*|={j_tau_max:.4g} (range [{j_lo},{j_hi}]), max|P|={p_max:.4g} "
          f"(range [{p_lo},{p_hi})) -> {'PASS' if token_plausible else 'FLAGGED (implausible magnitude)'}")

    # --- Check 4: confidence-gate sanity (only meaningful if this clip was
    # manually identified as having a tracking failure -- see --flagged_clips) ---
    c_min, c_mean = float(confidence.min()), float(confidence.mean())
    print(f"    check4 confidence: min={c_min:.3f} mean={c_mean:.3f}")

    return {
        "clip_id": clip.clip_id,
        "raw_smoothed_rms_gap": float(np.sqrt(np.mean((clip.theta - fitted_at_raw) ** 2))),
        "theta_ddot_variance": theta_ddot_variance,
        "differentiation_flagged": diff_flagged,
        "token_plausible": token_plausible,
        "confidence_min": c_min,
        "confidence_mean": c_mean,
    }


def _trajectory_figure(plt, t_raw, theta_raw, t_smooth, theta_smooth):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_raw, theta_raw, "o", markersize=3, alpha=0.5, label="raw annotation")
    ax.plot(t_smooth, theta_smooth, "-", label="smoothed")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("theta [rad]")
    ax.legend()
    fig.tight_layout()
    return fig


def _differentiation_figure(plt, t, theta_dot, theta_ddot):
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    axes[0].plot(t, theta_dot)
    axes[0].set_ylabel("theta_dot")
    axes[1].plot(t, theta_ddot)
    axes[1].set_ylabel("theta_ddot")
    axes[1].set_xlabel("t [s]")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", type=str, default="Safe")
    parser.add_argument("--num_clips", type=int, default=5)
    parser.add_argument("--npz_clips", type=str, nargs="*", default=None, help="Pre-processed (t, theta) npz clips, bypassing HOI4D entirely.")
    parser.add_argument("--num_tokens", type=int, default=64)
    parser.add_argument("--smoothing", type=float, default=1e-3)
    parser.add_argument("--residual_scale", type=float, default=0.05)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "poise_ot.yaml")
    parser.add_argument("--flagged_clips", type=str, nargs="*", default=[], help="clip_ids manually identified as having a visible tracking failure (check 4).")
    args = parser.parse_args()

    with open(args.config) as f:
        poise_cfg = yaml.safe_load(f)
    s2 = poise_cfg["stage2"]
    args.plausible_impulse_range = s2["plausible_impulse_range"]
    args.plausible_power_range = s2["plausible_power_range"]
    args.theta_ddot_noise_multiplier = s2["theta_ddot_noise_multiplier"]

    params = load_assumed_params(args.config)
    print(f"[stage2_checks] ASSUMED target params (not measured): {params}")

    clips = load_clips(args)
    if not clips:
        raise SystemExit("no clips loaded -- pass --npz_clips or set POISE_OT_HOI4D_ROOT and use --category/--num_clips")

    results = []
    for clip in clips:
        print(f"\n=== clip: {clip.clip_id} ===")
        results.append(run_checks_for_clip(clip, params, args))

    if args.flagged_clips:
        print("\n=== Check 4 summary: manually-flagged (tracking-failure) clips ===")
        flagged_results = [r for r in results if r["clip_id"] in args.flagged_clips]
        other_results = [r for r in results if r["clip_id"] not in args.flagged_clips]
        if flagged_results and other_results:
            flagged_conf = np.mean([r["confidence_mean"] for r in flagged_results])
            other_conf = np.mean([r["confidence_mean"] for r in other_results])
            print(f"  flagged-clip mean confidence: {flagged_conf:.3f}")
            print(f"  other-clip mean confidence: {other_conf:.3f}")
            print(f"  RESULT: {'PASS' if flagged_conf < other_conf else 'FAIL'} (flagged clips should score lower)")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"\n[stage2_checks] wrote {LOG_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
