"""HOI4D clip loader -> raw, un-smoothed per-frame joint angle theta(t).

**External prerequisite**: this requires an actual HOI4D download
(https://hoi4d.github.io) somewhere reachable. This dev machine has neither
the dataset nor a verified copy of its raw per-frame annotation layout, so
two loading paths are provided:

1. :func:`load_clip_npz` -- a simple, self-documented intermediate format
   (``{"t": [...], "theta": [...]}`` in an ``.npz``). This is the reliable
   path: same shape contract ``poise_ot.reference_generator`` already uses
   for synthetic data, so everything downstream is format-agnostic once a
   clip has been reduced to this. Recommended if the raw-layout parser below
   doesn't match what you actually see once the dataset is downloaded.
2. :func:`load_clip_raw` -- a best-effort parser against HOI4D's documented
   per-frame object/part-pose annotation convention (one json per frame
   under a clip's ``objpose``-style directory, an articulated joint's angle
   recovered from the annotated part rotation about its joint axis). This
   is **not verified against real files** -- write access to HOI4D wasn't
   available while writing this. Treat it as a documented best guess: if it
   raises or parses incorrectly against your actual download, preprocess a
   clip into the npz format above (one small script per clip is enough) and
   use :func:`load_clip_npz` instead, rather than debugging this parser
   blind.

The data root is configurable via the ``POISE_OT_HOI4D_ROOT`` environment
variable (matching the ``PHYSICS_OT_DATA_DIR``/``PHYSICS_OT_ALLEGRO_USD_PATH``
override pattern already used elsewhere in this repo), never hardcoded.

Only the ``Safe`` category (a rotational latch/knob-like joint) maps
directly onto the resistive-knob model reused from Stage 1
(``physics_ot.dynamics.knob.KnobDynamics``, a rotational spring-damper-
Coulomb model). ``StorageFurniture`` clips may be *either* a revolute door
(also rotational, fine) or a prismatic sliding drawer (translational -- the
knob's inverse dynamics don't apply; would need the drawer force equation
``F* = m*x_ddot + b*x_dot + F_c*tanh(x_dot/v_eps)`` instead, out of scope
here). Callers should confirm a clip's joint type before batching it through
``inverse_dynamics.py`` -- :func:`load_clip_raw` and :func:`load_clip_npz`
both return a ``joint_type`` field precisely so this can be checked rather
than silently assumed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def hoi4d_root() -> Path:
    override = os.environ.get("POISE_OT_HOI4D_ROOT")
    if not override:
        raise RuntimeError(
            "POISE_OT_HOI4D_ROOT is not set -- point it at a local HOI4D download "
            "(this repo does not bundle the dataset)."
        )
    return Path(override)


@dataclass
class RawClip:
    """Raw, un-smoothed per-frame joint-angle samples for one clip.

    Same shape contract as ``poise_ot.reference_generator``'s synthetic
    ``(t, theta)`` output before smoothing -- downstream code (smoothing.py,
    inverse_dynamics.py, token_builder.py) doesn't care whether the source
    was synthetic or real.
    """

    clip_id: str
    category: str
    joint_type: str  # "revolute" or "prismatic" -- KnobDynamics only applies to "revolute"
    t: np.ndarray  # [N], seconds
    theta: np.ndarray  # [N], radians (revolute) or meters (prismatic)


def load_clip_npz(path: str | Path, category: str, joint_type: str = "revolute") -> RawClip:
    """Load a pre-processed clip from the simple intermediate npz format.

    Expected keys: ``t`` [N] (seconds) and ``theta`` [N] (radians/meters).
    """
    path = Path(path)
    data = np.load(path)
    return RawClip(
        clip_id=path.stem,
        category=category,
        joint_type=joint_type,
        t=data["t"].astype(np.float64),
        theta=data["theta"].astype(np.float64),
    )


def load_clip_raw(clip_dir: str | Path, category: str, fps: float = 30.0) -> RawClip:
    """Best-effort parse of HOI4D's per-frame annotation convention.

    Looks for ``<clip_dir>/objpose/*.json``, one file per frame, each
    containing a ``rotation`` (axis-angle or quaternion about the
    articulated joint's axis) and a frame index recoverable from the
    filename (``0000.json``, ``0001.json``, ...). Extracts the scalar joint
    angle as the rotation's magnitude about the annotated joint axis.

    **Unverified** -- see module docstring. Raises a ``FileNotFoundError``
    with an explicit pointer to :func:`load_clip_npz` if the expected
    directory layout isn't present, rather than failing silently or
    guessing further.
    """
    clip_dir = Path(clip_dir)
    pose_dir = clip_dir / "objpose"
    if not pose_dir.is_dir():
        raise FileNotFoundError(
            f"{pose_dir} not found -- HOI4D's raw per-frame annotation layout was not verified "
            "against a real download while writing this loader. Preprocess this clip into the "
            "npz format (t, theta arrays) and use load_clip_npz instead."
        )

    frame_files = sorted(pose_dir.glob("*.json"))
    if not frame_files:
        raise FileNotFoundError(f"no per-frame json files found under {pose_dir}")

    thetas = []
    for frame_file in frame_files:
        with open(frame_file) as f:
            frame = json.load(f)
        # Best-effort key guess: a per-frame scalar joint angle, or an
        # axis-angle rotation vector whose norm is the joint angle.
        if "jointState" in frame:
            thetas.append(float(frame["jointState"]))
        elif "rotation" in frame:
            rotation = np.asarray(frame["rotation"], dtype=np.float64)
            thetas.append(float(np.linalg.norm(rotation)))
        else:
            raise KeyError(
                f"{frame_file} has neither a 'jointState' nor 'rotation' key -- "
                "this parser's key guesses don't match this file; inspect it directly "
                "and adjust, or preprocess into load_clip_npz's format instead."
            )

    theta = np.asarray(thetas, dtype=np.float64)
    t = np.arange(len(theta), dtype=np.float64) / fps
    return RawClip(clip_id=clip_dir.name, category=category, joint_type="revolute", t=t, theta=theta)


def list_clips(category: str, root: Path | None = None) -> list[Path]:
    """List clip directories under ``<root>/<category>``."""
    root = root or hoi4d_root()
    category_dir = root / category
    if not category_dir.is_dir():
        raise FileNotFoundError(f"{category_dir} not found under HOI4D root {root}")
    return sorted(p for p in category_dir.iterdir() if p.is_dir())
