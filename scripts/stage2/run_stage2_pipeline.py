"""Batch HOI4D clips through video_loader -> smoothing -> inverse_dynamics
-> confidence_gate -> token_builder (the last step literally unchanged from
Stage 1 -- that's the entire point of factoring it out on its own) and write
one token sequence + npz per clip to data/stage2_tokens/.

**External prerequisite**: requires POISE_OT_HOI4D_ROOT to point at the
directory containing HOI4D_annotations/ and HOI4D_CAD_Model_for_release/
(i.e. select_hoi4d_clips.py --fetch's --out_dir), and a clip-list file (its
--output) naming which release.txt-style clip_ids to process. This script
does not fetch the dataset -- see scripts/stage2/select_hoi4d_clips.py and
docs/HOI4D_DOWNLOAD.md.

Usage:
    python scripts/stage2/select_hoi4d_clips.py --release_txt ... \\
        --category C6 --task T1 --num_clips 12 --output data/clip_list.txt \\
        --annotations_source ... --cad_source ... --out_dir data/hoi4d_raw --fetch

    POISE_OT_HOI4D_ROOT=data/hoi4d_raw python scripts/stage2/run_stage2_pipeline.py \\
        --clip_list data/clip_list.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics_ot.dynamics.knob import KnobParams  # noqa: E402

from poise_ot.token_builder import build_token  # noqa: E402
from poise_ot.video.confidence_gate import compute_confidence  # noqa: E402
from poise_ot.video.inverse_dynamics import target_conditioned_torque  # noqa: E402
from poise_ot.video.smoothing import smooth_clip  # noqa: E402
from poise_ot.video.video_loader import RawClip, hoi4d_root, load_clip_raw  # noqa: E402


def load_assumed_params(poise_cfg_path: Path) -> KnobParams:
    """theta0 is deliberately NOT set here -- see process_clip: HOI4D's raw
    annotated angle is not guaranteed to be zeroed at the object's rest
    position (unlike the synthetic Stage-1 task, which defines theta0 by
    construction), so it is derived per-clip instead of assumed globally."""
    with open(poise_cfg_path) as f:
        poise_cfg = yaml.safe_load(f)
    s2 = poise_cfg["stage2"]
    return KnobParams(
        inertia=s2["assumed_inertia"], damping=s2["assumed_damping"], stiffness=s2["assumed_stiffness"],
        theta0=0.0, coulomb_torque=s2["assumed_coulomb_torque"], v_eps=s2["assumed_v_eps"],
    )


def process_clip(clip: RawClip, params: KnobParams, num_tokens: int, smoothing: float, impulse_window: int, residual_scale: float) -> dict:
    if clip.joint_type != "revolute":
        raise ValueError(
            f"clip {clip.clip_id} has joint_type={clip.joint_type!r} -- KnobDynamics is a rotational "
            "spring-damper-Coulomb model and does not apply to a prismatic joint (see video_loader.py's "
            "module docstring). Skip this clip or add the drawer force equation before processing it."
        )

    # theta0 (the spring rest angle Eq 10/20's tau* is computed relative to)
    # is assumed to be this clip's own first raw sample -- HOI4D clips
    # record an interaction starting from the object at rest, so this is a
    # documented, clip-specific assumption, not the arbitrary global 0.0 a
    # naive reuse of load_assumed_params' KnobParams would silently apply.
    # theta_goal is similarly taken as this clip's own final sample (the
    # human's achieved end state), only to normalize progress for the
    # token's s_t channel -- HOI4D annotations carry no separate "goal".
    theta0 = float(clip.theta[0])
    theta_goal = float(clip.theta[-1])
    clip_params = replace(params, theta0=theta0)

    smoothing_result = smooth_clip(clip.t, clip.theta, smoothing=smoothing)
    ind_result = target_conditioned_torque(smoothing_result, clip_params, num_tokens=num_tokens)
    confidence = compute_confidence(smoothing_result, ind_result, clip_params, residual_scale=residual_scale)

    progress = torch.from_numpy(
        ((ind_result.theta - theta0) / max(theta_goal - theta0, 1e-6)).astype(np.float32)
    )
    theta_dot = torch.from_numpy(ind_result.theta_dot.astype(np.float32))
    tau = torch.from_numpy(ind_result.tau_star.astype(np.float32))
    conf_t = torch.from_numpy(confidence.astype(np.float32))
    dt = float(ind_result.t[1] - ind_result.t[0]) if len(ind_result.t) > 1 else 1.0

    token = build_token(progress, theta_dot, tau, clip_params, dt, window_h=impulse_window, confidence=conf_t)

    return {
        "clip_id": clip.clip_id,
        "category": clip.category,
        "t": ind_result.t,
        "z": token.z.numpy(),
        "confidence": token.confidence.numpy(),
        "tau": token.tau.numpy(),
        "residual_rms": smoothing_result.residual_rms,
        "assumed_params": ind_result.assumed_params,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip_list", type=Path, required=True,
                         help="One release.txt-style clip_id per line (select_hoi4d_clips.py's --output).")
    parser.add_argument("--hoi4d_root", type=Path, default=None,
                         help="Directory containing HOI4D_annotations/ and HOI4D_CAD_Model_for_release/ "
                              "(defaults to POISE_OT_HOI4D_ROOT).")
    parser.add_argument("--num_tokens", type=int, default=64)
    parser.add_argument("--smoothing", type=float, default=1e-3, help="Nonzero -- real video needs actual smoothing, unlike Stage 1's exact synthetic data.")
    parser.add_argument("--residual_scale", type=float, default=0.05, help="Normalizes the residual-confidence signal; defaults to configs/knob.yaml's success_threshold scale.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "poise_ot.yaml")
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "data" / "stage2_tokens")
    args = parser.parse_args()

    root = args.hoi4d_root or hoi4d_root()
    params = load_assumed_params(args.config)
    print(f"[stage2_pipeline] ASSUMED target params (not measured from HOI4D): {asdict(params)}")

    clip_ids = [line.strip() for line in args.clip_list.read_text().splitlines() if line.strip()]
    if not clip_ids:
        raise SystemExit(f"no clip_ids found in {args.clip_list}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for clip_id in clip_ids:
        try:
            clip = load_clip_raw(clip_id, root=root)
            result = process_clip(clip, params, args.num_tokens, args.smoothing, impulse_window=5, residual_scale=args.residual_scale)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"[stage2_pipeline] SKIPPED {clip_id}: {exc}")
            continue

        out_path = args.output_dir / f"{result['clip_id']}.npz"
        np.savez(out_path, t=result["t"], z=result["z"], confidence=result["confidence"], tau=result["tau"])
        manifest.append({
            "clip_id": result["clip_id"], "category": result["category"], "path": str(out_path),
            "residual_rms": result["residual_rms"], "assumed_params": result["assumed_params"],
        })
        print(f"[stage2_pipeline] wrote {out_path} (residual_rms={result['residual_rms']:.4g})")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[stage2_pipeline] done -- {len(manifest)}/{len(clip_ids)} clips processed, manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
