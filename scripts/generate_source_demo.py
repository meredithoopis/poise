"""Generate a scripted source object trajectory for the resistive-knob task.

Stands in for a real human/Shadow-hand demonstration (section 25: "use only
source object trajectory"). Produces a min-jerk theta(t) profile from theta0
to theta_goal, sampled like an egocentric-video extraction pipeline would
(section 21: ~30 Hz, position only -- velocity/acceleration are derived later
by the smoothing pipeline in build_physics_reference.py, never baked in here).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def min_jerk_theta(t: np.ndarray, theta0: float, theta_goal: float, duration: float) -> np.ndarray:
    tau = np.clip(t / duration, 0.0, 1.0)
    return theta0 + (theta_goal - theta0) * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "knob.yaml")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data" / "demonstrations" / "knob_source_001.npz")
    parser.add_argument("--duration", type=float, default=2.0, help="Demonstration duration [s].")
    parser.add_argument("--sample_rate", type=float, default=30.0, help="Sample rate [Hz], mimicking video capture.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["knob"]

    num_samples = int(round(args.duration * args.sample_rate)) + 1
    t = np.linspace(0.0, args.duration, num_samples)
    theta = min_jerk_theta(t, cfg["theta0"], cfg["theta_goal"], args.duration)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, time=t, task_position=theta)
    print(f"[generate_source_demo] wrote {num_samples} samples to {args.output}")


if __name__ == "__main__":
    main()
