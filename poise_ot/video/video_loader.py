"""HOI4D clip loader -> raw, un-smoothed per-frame joint angle theta(t).

**Verified against real HOI4D data** (category Safe, clip
``ZY20210800001/H1/C6/N03/S247/s01/T1``, fetched via
``scripts/stage2/select_hoi4d_clips.py``). Two loading paths:

1. :func:`load_clip_raw` -- parses HOI4D's real per-frame object-pose
   annotations (``HOI4D_annotations/<clip_id>/objpose/*.json``, one file per
   frame) together with the CAD model's joint annotation
   (``HOI4D_CAD_Model_for_release/articulated/<CategoryName>/<instance>/
   mobility_v2.json``) to recover the scalar joint angle. See the function
   docstring for why this needs *both* files (camera egomotion) and what's
   verified vs. assumed.
2. :func:`load_clip_npz` -- a simple, self-documented intermediate format
   (``{"t": [...], "theta": [...]}`` in an ``.npz``), for clips you've
   preprocessed by hand or from a category whose label/joint convention
   :func:`load_clip_raw` doesn't match.

The data root is configurable via the ``POISE_OT_HOI4D_ROOT`` environment
variable (matching the ``PHYSICS_OT_DATA_DIR``/``PHYSICS_OT_ALLEGRO_USD_PATH``
override pattern already used elsewhere in this repo), never hardcoded --
point it at wherever ``select_hoi4d_clips.py --fetch``'s ``--out_dir``
landed (i.e. the directory directly containing ``HOI4D_annotations/`` and
``HOI4D_CAD_Model_for_release/``).

Only the ``Safe`` category (a rotational door latch) has been verified end
to end. ``StorageFurniture`` clips may be *either* a revolute door (should
work, unverified) or a prismatic sliding drawer (translational -- the
knob's inverse dynamics don't apply; would need the drawer force equation
``F* = m*x_ddot + b*x_dot + F_c*tanh(x_dot/v_eps)`` instead, out of scope
here -- this is exactly why ``select_hoi4d_clips.py`` defaults to C6/T1,
not C4/T1). Callers should confirm a clip's joint type before batching it
through ``inverse_dynamics.py`` -- both loaders return a ``joint_type``
field precisely so this can be checked rather than silently assumed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rt

# HOI4D-Instructions README's C* -> category-name mapping. The CAD-model zip
# is organized by spelled-out category name, not the C6/C4-style category ID
# used in release.txt/clip paths -- confirmed against a real
# HOI4D_CAD_Model_for_release.zip (e.g. "articulated/Safe/003/mobility_v2.json").
# Kept in sync by hand with the identical copy in
# scripts/stage2/select_hoi4d_clips.py (duplicated deliberately -- that
# script doesn't otherwise depend on poise_ot being importable).
CATEGORY_ID_TO_NAME = {
    "C1": "ToyCar", "C2": "Mug", "C3": "Laptop", "C4": "StorageFurniture",
    "C5": "Bottle", "C6": "Safe", "C7": "Bowl", "C8": "Bucket", "C9": "Scissors",
    "C11": "Pliers", "C12": "Kettle", "C13": "Knife", "C14": "TrashCan",
    "C17": "Lamp", "C18": "Stapler", "C20": "Chair",
}


def hoi4d_root() -> Path:
    override = os.environ.get("POISE_OT_HOI4D_ROOT")
    if not override:
        raise RuntimeError(
            "POISE_OT_HOI4D_ROOT is not set -- point it at the directory containing "
            "HOI4D_annotations/ and HOI4D_CAD_Model_for_release/ (see "
            "scripts/stage2/select_hoi4d_clips.py --fetch's --out_dir)."
        )
    return Path(override)


def resolve_clip_paths(root: Path, clip_id: str) -> tuple[Path, Path, str]:
    """``clip_id`` is release.txt-style
    (``'ZY.../H../C6/N03/S.../s../T1'``, as produced by
    ``scripts/stage2/select_hoi4d_clips.py``). Returns
    ``(objpose_dir, mobility_json_path, category_name)`` under the real
    extracted layout: ``HOI4D_annotations/<clip_id>/objpose/`` and
    ``HOI4D_CAD_Model_for_release/articulated/<CategoryName>/<NNN>/
    mobility_v2.json``."""
    parts = clip_id.split("/")
    if len(parts) != 7:
        raise ValueError(f"clip_id {clip_id!r} doesn't look like a release.txt-style path "
                          f"({{camera}}/{{human}}/{{category}}/{{instance}}/{{room}}/{{layout}}/{{task}}).")
    category_id, instance_id = parts[2], parts[3]
    category_name = CATEGORY_ID_TO_NAME.get(category_id, category_id)
    instance_num = int(instance_id.lstrip("N"))
    objpose_dir = root / "HOI4D_annotations" / clip_id / "objpose"
    mobility_json = root / "HOI4D_CAD_Model_for_release" / "articulated" / category_name / f"{instance_num:03d}" / "mobility_v2.json"
    return objpose_dir, mobility_json, category_name


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


def _part_rotation(part: dict) -> Rt:
    r = part["rotation"]
    return Rt.from_euler("XYZ", [r["x"], r["y"], r["z"]])


def _find_part(data_list: list[dict], suffix: str, frame_file: Path) -> dict:
    matches = [d for d in data_list if str(d.get("label", "")).lower().endswith(suffix)]
    if len(matches) != 1:
        labels = [d.get("label") for d in data_list]
        raise KeyError(
            f"{frame_file}: expected exactly one annotated part with label ending in "
            f"{suffix!r}, found {len(matches)} (labels present: {labels}). The 'box'/'door' "
            "label suffix convention was only verified for category Safe -- pass "
            "moving_suffix/static_suffix matching this category's real labels."
        )
    return matches[0]


def _load_joint_axis(mobility_json: Path) -> np.ndarray:
    """Parses mobility_v2.json's joint tree, requires exactly one non-root
    ("自由"/free) joint (matching KnobDynamics' single-DOF model), and
    requires it to be a revolute hinge ("铰链（旋转）") -- verified against a
    real Safe instance. Raises clearly rather than guessing on any other
    joint-tree shape (multi-joint objects, prismatic sliders, ...)."""
    with open(mobility_json, encoding="utf-8") as f:
        joints = json.load(f)
    articulated = [j for j in joints if j.get("joint") not in (None, "自由")]
    if len(articulated) != 1:
        raise ValueError(
            f"{mobility_json} has {len(articulated)} non-free joint(s) -- this parser only "
            f"supports a single-DOF object (matching KnobDynamics' 1-DoF model); found joint "
            f"types: {[j.get('joint') for j in articulated]}"
        )
    joint = articulated[0]
    joint_label = joint.get("joint", "")
    if "铰链" not in joint_label:  # "hinge" -- revolute; a prismatic slider uses a different label
        raise ValueError(
            f"{mobility_json}'s joint type {joint_label!r} is not a recognized revolute hinge "
            "-- KnobDynamics only models revolute joints (see video_loader.py's module docstring)."
        )
    axis = np.asarray(joint["jointData"]["axis"]["direction"], dtype=np.float64)
    return axis / np.linalg.norm(axis)


def load_clip_raw(
    clip_id: str,
    root: Path | None = None,
    fps: float = 15.0,
    moving_suffix: str = "door",
    static_suffix: str = "box",
) -> RawClip:
    """Recover theta(t) for one real HOI4D clip.

    HOI4D's ``objpose/*.json`` (one file per frame, filename stem = frame
    index -- the JSON's own internal ``frameId`` field was NOT usable as a
    time index when checked against a real clip, it did not vary per file)
    annotates each tracked rigid part's pose *in camera frame*. Because
    HOI4D is egocentric video, BOTH the static and moving parts' absolute
    rotation drift frame-to-frame from camera motion alone (confirmed: in a
    real clip, the static part's rotation shifted by about as much as the
    moving part's between adjacent early frames, before any real opening
    motion) -- so the joint angle cannot be read off the moving part's
    rotation directly. Instead this computes the *relative* rotation between
    the moving and static parts (camera egomotion cancels out of a
    relative-rotation) and projects it onto the joint's axis direction from
    ``mobility_v2.json`` (the CAD model's canonical-frame joint annotation).

    Part matching is by annotation label suffix (``"...door"``/``"...box"``
    for category Safe, e.g. "Safedoor"/"Safebox") -- verified for Safe only;
    other categories may use a different convention and would need
    ``moving_suffix``/``static_suffix`` adjusted (inspect one frame file
    directly first).

    fps=15.0 matches HOI4D-Instructions' own ``utils/decode.py`` (RGB/depth
    decoded at ``fps=15``); this assumes objpose annotations are at the same
    per-frame rate as the decoded video, which has not been independently
    confirmed beyond "the filename indices are consecutive integers."
    """
    root = root or hoi4d_root()
    objpose_dir, mobility_json, category_name = resolve_clip_paths(root, clip_id)
    if not objpose_dir.is_dir():
        raise FileNotFoundError(
            f"{objpose_dir} not found -- check clip_id/root, or that "
            "select_hoi4d_clips.py --fetch actually extracted this clip's objpose files."
        )
    if not mobility_json.is_file():
        raise FileNotFoundError(
            f"{mobility_json} not found -- check root, or that select_hoi4d_clips.py --fetch "
            "actually extracted this instance's CAD model."
        )

    axis = _load_joint_axis(mobility_json)

    frame_files = sorted(objpose_dir.glob("*.json"), key=lambda p: int(p.stem))
    if not frame_files:
        raise FileNotFoundError(f"no per-frame json files found under {objpose_dir}")

    thetas: list[float] = []
    frame_indices: list[int] = []
    for frame_file in frame_files:
        with open(frame_file, encoding="utf-8") as f:
            frame = json.load(f)
        data_list = frame["dataList"]
        moving = _find_part(data_list, moving_suffix, frame_file)
        static = _find_part(data_list, static_suffix, frame_file)
        r_rel = _part_rotation(static).inv() * _part_rotation(moving)
        theta = float(np.dot(r_rel.as_rotvec(), axis))
        thetas.append(theta)
        frame_indices.append(int(frame_file.stem))

    # Each frame's angle is recovered independently via Rotation.as_rotvec(),
    # whose magnitude is bounded to [0, pi] -- near a near-zero relative
    # rotation (e.g. this clip's start, before the door has moved), the
    # rotation's axis direction is numerically unstable, so consecutive
    # frames can land on opposite sides of that branch cut and produce a
    # spurious near-2*theta jump that then blows up under differentiation
    # (the "three orders of magnitude" amplification the paper warns about,
    # here from a tracking-representation artifact rather than raw pose
    # noise). np.unwrap corrects any inter-frame jump >= pi, which this
    # branch-cut artifact always produces but genuine motion within the
    # joint's own +-104.76 deg range never does.
    theta = np.unwrap(np.asarray(thetas, dtype=np.float64))
    t = np.asarray(frame_indices, dtype=np.float64) / fps
    flat_id = clip_id.replace("/", "_")
    return RawClip(clip_id=flat_id, category=category_name, joint_type="revolute", t=t, theta=theta)
