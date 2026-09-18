"""Dump the prim hierarchy, applied API schemas, and joint body targets of the
generated knob USD asset, to diagnose why IsaacLab reports zero joints found.

    $ISAACLAB scripts/tools/inspect_knob_asset.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def inspect(path: Path) -> None:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    print(f"default prim: {stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else None}")
    print()

    for prim in stage.Traverse():
        applied = list(prim.GetAppliedSchemas())
        print(f"{prim.GetPath()}  (type={prim.GetTypeName()})")
        if applied:
            print(f"    applied schemas: {applied}")

        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            body0 = joint.GetBody0Rel().GetTargets()
            body1 = joint.GetBody1Rel().GetTargets()
            print(f"    joint body0: {body0}")
            print(f"    joint body1: {body1}")
            excluded = joint.GetExcludeFromArticulationAttr().Get() if joint.GetExcludeFromArticulationAttr() else None
            print(f"    excludeFromArticulation: {excluded}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--path",
        type=Path,
        default=Path(os.environ.get("PHYSICS_OT_DATA_DIR", REPO_ROOT / "data")) / "objects" / "knob.usd",
    )
    args = parser.parse_args()
    inspect(args.path)


if __name__ == "__main__":
    main()
