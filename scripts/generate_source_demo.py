"""Generate a scripted source object trajectory for the resistive-knob task.

Stands in for a real human/Shadow-hand demonstration (section 25: "use only
source object trajectory"). Produces a min-jerk theta(t) profile from theta0
to theta_goal over --reach_duration, then holds at theta_goal for the rest of
--total_duration, sampled like an egocentric-video extraction pipeline would
(section 21: ~30 Hz, position only -- velocity/acceleration are derived later
by the smoothing pipeline in build_physics_reference.py, never baked in here).

--total_duration defaults to the environment's episode_seconds (configs/
knob.yaml, simulation:): the reference must span the same duration as a
training episode, not just the reach itself, since the window compared
against it (knob_allegro_env.py's robot_window) is resampled from the whole
episode. A reference that only covers the first few seconds of a longer
episode is comparing "reach" motion against "reach, then idle" motion for
every step after the reach completes -- a structural mismatch that the
Sinkhorn OT distance (used by the physics_ot reward mode) is far more
sensitive to than the simpler baseline distance metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def min_jerk_theta(t: np.ndarray, theta0: float, theta_goal: float, reach_duration: float) -> np.ndarray:
    tau = np.clip(t / reach_duration, 0.0, 1.0)
    return theta0 + (theta_goal - theta0) * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "knob.yaml")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data" / "demonstrations" / "knob_source_001.npz")
    parser.add_argument("--reach_duration", type=float, default=2.0, help="Min-jerk reach time [s].")
    parser.add_argument(
        "--total_duration",
        type=float,
        default=None,
        help="Total demo span [s]; defaults to configs/knob.yaml's simulation.episode_seconds. "
        "The knob holds at theta_goal for t > reach_duration.",
    )
    parser.add_argument("--sample_rate", type=float, default=30.0, help="Sample rate [Hz], mimicking video capture.")
    args = parser.parse_args()

    with open(args.config) as f:
        full_cfg = yaml.safe_load(f)
    cfg = full_cfg["knob"]
    total_duration = args.total_duration
    if total_duration is None:
        total_duration = full_cfg["simulation"]["episode_seconds"]

    num_samples = int(round(total_duration * args.sample_rate)) + 1
    t = np.linspace(0.0, total_duration, num_samples)
    theta = min_jerk_theta(t, cfg["theta0"], cfg["theta_goal"], args.reach_duration)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, time=t, task_position=theta)
    print(
        f"[generate_source_demo] wrote {num_samples} samples over {total_duration}s "
        f"(reach: {args.reach_duration}s) to {args.output}"
    )


if __name__ == "__main__":
    main()
