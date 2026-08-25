"""Convert a raw object-trajectory demonstration into a Physics-OT reference.

Applies target-conditioned inverse dynamics (section 4-7): the demonstrated
theta_H(t) is smoothed once, differentiated, and combined with the *target*
knob's physics parameters (not any estimate of human force) to produce
tau*(t), then resampled to T_H reference tokens.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from physics_ot.dynamics.knob import KnobParams
from physics_ot.references.build_reference import build_reference, save_reference

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "knob.yaml")
    parser.add_argument(
        "--demo", type=Path, default=REPO_ROOT / "data" / "demonstrations" / "knob_source_001.npz"
    )
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "data" / "references" / "knob_target_nominal.npz"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    knob_cfg = cfg["knob"]
    ref_cfg = cfg["reference"]
    params = KnobParams(
        inertia=knob_cfg["inertia"],
        damping=knob_cfg["damping"],
        stiffness=knob_cfg["stiffness"],
        theta0=knob_cfg["theta0"],
        coulomb_torque=knob_cfg["coulomb_torque"],
        v_eps=knob_cfg["v_eps"],
    )

    demo = np.load(args.demo)
    reference = build_reference(
        t_raw=demo["time"],
        theta_raw=demo["task_position"],
        knob_params=params,
        theta_goal=knob_cfg["theta_goal"],
        num_tokens=ref_cfg["num_tokens"],
        smoothing=ref_cfg["smoothing"],
    )

    save_reference(args.output, reference)
    print(f"[build_physics_reference] wrote {reference.num_tokens}-token reference to {args.output}")


if __name__ == "__main__":
    main()
