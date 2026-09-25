"""Step 11's "Regression check #1" gate: compare train_stage3.py's
learning curve / critic-loss shape against Stage 1's toy-env result.

Deliberately a *separate*, non-IsaacLab script rather than one that runs
both trainings itself: IsaacLab's ``AppLauncher`` can only be constructed
once per process, so "run Stage 1, run Stage 3, compare" can't happen
in-process here the way ``run_stage1_checks.py`` calls ``train_stage1.train``
directly -- instead this takes the two already-produced history jsons and
does the comparison the gate actually asks for: do the critic-loss curves
have a *comparable shape* (both trending the same direction, similar order
of magnitude), not identical numbers (the environments are different --
synthetic ODE vs. contact-rich PhysX -- so exact numeric match is not the
bar; "does this look like the same algorithm learning, not a differently-
broken one" is).

Usage:
    python scripts/stage1/train_stage1.py --episodes 30 --output logs/stage1_history.json
    $ISAACLAB scripts/stage3/train_stage3.py --episodes 20 --output logs/stage3_history.json
    python scripts/stage3/run_stage3_regression_check.py \\
        --stage1_history logs/stage1_history.json --stage3_history logs/stage3_history.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "logs" / "stage3"


def _fit_slope(y: list[float]) -> float:
    if len(y) < 2:
        return float("nan")
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def _maybe_plot(stage1: dict, stage3: dict, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"    (matplotlib not installed -- skipping plot at {path})")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(stage1["step"], stage1["loss_q_task"], label="stage1")
    axes[0].plot(stage3["step"], stage3["loss_q_task"], label="stage3")
    axes[0].set_title("Q_task loss")
    axes[0].legend()
    axes[1].plot(stage1["step"], stage1["loss_q_c"], label="stage1")
    axes[1].plot(stage3["step"], stage3["loss_q_c"], label="stage3")
    axes[1].set_title("Q_C loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved plot -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1_history", type=Path, required=True)
    parser.add_argument("--stage3_history", type=Path, required=True)
    # A generous default: the gate is "comparable order of magnitude", not
    # "within X%" -- exact match isn't expected given the different envs.
    parser.add_argument("--max_magnitude_ratio", type=float, default=20.0)
    args = parser.parse_args()

    stage1 = json.loads(args.stage1_history.read_text())
    stage3 = json.loads(args.stage3_history.read_text())

    for name in ("loss_q_task", "loss_q_c"):
        if not stage1.get(name) or not stage3.get(name):
            raise SystemExit(f"{name} missing/empty in one of the two histories -- did SAC updates ever run "
                              f"(warmup_steps reached, replay buffer filled)?")

    print("=== Step 11: regression check #1 (synthetic reference, small scale) ===")
    results = {}
    for name in ("loss_q_task", "loss_q_c"):
        s1_slope = _fit_slope(stage1[name])
        s3_slope = _fit_slope(stage3[name])
        s1_final = float(np.mean(stage1[name][-10:]))
        s3_final = float(np.mean(stage3[name][-10:]))
        ratio = max(s1_final, s3_final) / max(min(s1_final, s3_final), 1e-9)
        same_direction = (s1_slope >= 0) == (s3_slope >= 0)
        magnitude_ok = ratio <= args.max_magnitude_ratio
        ok = same_direction and magnitude_ok
        print(f"  {name}: stage1 slope={s1_slope:.5g} final={s1_final:.5g} | "
              f"stage3 slope={s3_slope:.5g} final={s3_final:.5g} | "
              f"magnitude ratio={ratio:.2g} (max {args.max_magnitude_ratio}) | "
              f"{'PASS' if ok else 'FAIL'}")
        results[name] = {"stage1_slope": s1_slope, "stage3_slope": s3_slope,
                          "stage1_final": s1_final, "stage3_final": s3_final,
                          "magnitude_ratio": ratio, "pass": ok}

    s1_success = float(np.mean(stage1["episode_success"][-50:])) if stage1.get("episode_success") else float("nan")
    s3_success = float(np.mean(stage3["episode_success"][-50:])) if stage3.get("episode_success") else float("nan")
    print(f"  success rate (last 50 episodes): stage1={s1_success:.3f} stage3={s3_success:.3f}")

    all_pass = all(r["pass"] for r in results.values())
    print(f"\n=== RESULT: {'PASS' if all_pass else 'FAIL'} ===")
    if not all_pass:
        print("If this fails, per step 11: the bug is in the simulator/action/observation wiring "
              "(knob_allegro_poise_env.py / knob_allegro_poise_env_cfg.py), NOT in mech_cost.py, "
              "alignment.py, or two_critic_sac.py, which are byte-for-byte the same code Stage 1 "
              "already validated.")

    _maybe_plot(stage1, stage3, LOG_DIR / "step11_regression_check.png")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "step11_summary.json").write_text(json.dumps(results, indent=2))

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
