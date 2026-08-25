"""Train a Physics-OT policy via IsaacLab's unified training CLI.

IsaacLab's per-library ``scripts/reinforcement_learning/<lib>/train.py``
entry points are deprecated in favor of ``isaaclab.sh train``; this is a
thin wrapper around that command so the Experiment-0 ``--reward_mode`` sweep
(sections 25-26) has a single, stable entry point here regardless of which
exact IsaacLab CLI surface is current on the training machine.

Requires the ``ISAACLAB_PATH`` environment variable (or ``--isaaclab_path``)
to point at the IsaacLab checkout, and both ``physics-ot`` and
``physics_ot_tasks`` to already be ``pip install -e``'d into that Python
environment.

Example:
    python scripts/train.py --task PhysicsOT-Knob-Allegro-Direct-v0 \\
        --reward_mode physics_ot --num_envs 512
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=str, default="PhysicsOT-Knob-Allegro-Direct-v0")
    parser.add_argument("--rl_library", type=str, default="rsl_rl")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument(
        "--reward_mode",
        type=str,
        default=None,
        choices=["sparse", "state_tracking", "dtw_physics", "ot_state", "pointwise_physics", "physics_ot"],
        help="Experiment-0 ablation arm (sections 25-26); passed through as a Hydra env override.",
    )
    parser.add_argument("--isaaclab_path", type=str, default=os.environ.get("ISAACLAB_PATH"))
    args, extra = parser.parse_known_args()

    if not args.isaaclab_path:
        raise SystemExit("Set ISAACLAB_PATH or pass --isaaclab_path to locate the IsaacLab checkout.")
    isaaclab_path = Path(args.isaaclab_path)
    launcher = isaaclab_path / ("isaaclab.bat" if platform.system() == "Windows" else "isaaclab.sh")
    if not launcher.exists():
        raise SystemExit(f"Could not find {launcher}; check --isaaclab_path/ISAACLAB_PATH.")

    cmd = [str(launcher), "train", "--rl_library", args.rl_library, "--task", args.task]
    if args.num_envs is not None:
        cmd += ["--num_envs", str(args.num_envs)]
    if args.max_iterations is not None:
        cmd += ["--max_iterations", str(args.max_iterations)]
    if args.reward_mode is not None:
        cmd += [f"env.reward_mode={args.reward_mode}"]
    cmd += extra

    print("[train] " + " ".join(cmd))
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
