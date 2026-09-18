"""Combine logs/sweep/seed<N>/eval_<mode>.json across seeds into a mean +/-
std table (section 26: the ablation is reported over 3 seeds, not one).

Pure Python, no IsaacLab dependency -- run with plain `python`:
    python scripts/aggregate_seeds.py --log_dir logs/sweep --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

MODES = ["sparse", "state_tracking", "dtw_physics", "ot_state", "pointwise_physics", "physics_ot"]
LABELS = {
    "sparse": "B0 Sparse",
    "state_tracking": "B1 State Tracking",
    "dtw_physics": "B2 DTW-Physics",
    "ot_state": "B3 OT-State",
    "pointwise_physics": "B4 Pointwise Physics",
    "physics_ot": "Ours: Physics-OT",
}
METRICS = [
    ("success_rate", 3),
    ("mean_final_pot_distance", 4),
    ("mean_completion_time_s", 2),
    ("mean_actuation_energy", 3),
]


def load(log_dir: Path, seeds: list[int]) -> dict[str, dict[str, list[float]]]:
    """mode -> metric -> list of per-seed values (None entries skipped, missing files warned)."""
    per_mode: dict[str, dict[str, list[float]]] = {mode: {key: [] for key, _ in METRICS} for mode in MODES}
    for seed in seeds:
        seed_dir = log_dir / f"seed{seed}"
        for mode in MODES:
            path = seed_dir / f"eval_{mode}.json"
            if not path.exists():
                print(f"[aggregate] warning: missing {path}")
                continue
            data = json.loads(path.read_text())
            for key, _ in METRICS:
                value = data.get(key)
                if value is not None:
                    per_mode[mode][key].append(value)
    return per_mode


def fmt_stat(values: list[float], digits: int) -> str:
    if not values:
        return "--"
    mean = statistics.mean(values)
    if len(values) > 1:
        std = statistics.stdev(values)
        return f"{mean:.{digits}f} +/- {std:.{digits}f} (n={len(values)})"
    return f"{mean:.{digits}f} (n=1)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_dir", type=Path, default=Path("logs/sweep"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=None, help="Optional combined JSON path.")
    args = parser.parse_args()

    per_mode = load(args.log_dir, args.seeds)

    print(f"Experiment-0 ablation, seeds={args.seeds}\n")
    for mode in MODES:
        print(f"{LABELS[mode]}:")
        for key, digits in METRICS:
            print(f"  {key:28s} {fmt_stat(per_mode[mode][key], digits)}")
        print()

    if args.output:
        combined = {
            mode: {
                key: {
                    "mean": statistics.mean(vals) if vals else None,
                    "std": statistics.stdev(vals) if len(vals) > 1 else 0.0 if vals else None,
                    "n": len(vals),
                    "values": vals,
                }
                for key, vals in per_mode[mode].items()
            }
            for mode in MODES
        }
        args.output.write_text(json.dumps(combined, indent=2))
        print(f"[aggregate] wrote {args.output}")


if __name__ == "__main__":
    main()
