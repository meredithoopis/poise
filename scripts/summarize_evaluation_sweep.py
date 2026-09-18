"""Print the Experiment-0 ablation table (sections 25-26, 32) from the JSON
results written by scripts/evaluate.py --output <path>.

Pure Python, no IsaacLab dependency -- run with plain `python`:
    python scripts/summarize_evaluation_sweep.py --log_dir logs/sweep
"""

from __future__ import annotations

import argparse
import json
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


def fmt(value: float | None, digits: int = 3) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_dir", type=Path, default=Path("logs/sweep"))
    args = parser.parse_args()

    rows = []
    for mode in MODES:
        path = args.log_dir / f"eval_{mode}.json"
        if not path.exists():
            rows.append((LABELS[mode], "(missing)", "", "", "", ""))
            continue
        data = json.loads(path.read_text())
        rows.append((
            LABELS[mode],
            fmt(data.get("success_rate"), 3),
            fmt(data.get("mean_final_pot_distance"), 4),
            fmt(data.get("mean_completion_time_s"), 2),
            fmt(data.get("mean_actuation_energy"), 3),
            str(data.get("num_episodes", "")),
        ))

    header = ("Method", "Success rate", "D_POT", "Time-to-success [s]", "Actuation energy", "Episodes")
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]

    def print_row(cells: tuple[str, ...]) -> None:
        print("  ".join(c.ljust(w) for c, w in zip(cells, widths)))

    print_row(header)
    print_row(tuple("-" * w for w in widths))
    for row in rows:
        print_row(row)


if __name__ == "__main__":
    main()
