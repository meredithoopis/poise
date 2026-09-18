"""Procedurally author a minimal resistive-knob USD asset.

Run this once on the machine with Isaac Sim installed (``pxr`` is only
available there, not on the Windows dev machine this repo was written on):

    python scripts/tools/generate_knob_asset.py

Builds a fixed base with a single revolute-jointed cylinder ("knob_joint",
Z axis) using the stable ``pxr.UsdPhysics`` schema, with no drive authored on
the joint -- ``KnobAllegroEnvCfg``'s ``ArticulationCfg`` uses
``actuators={}`` and the environment applies the knob's spring/damper/
Coulomb-friction torque explicitly each physics step (section 5), so the
joint here only needs to be free to rotate.

This is a best-effort authoring script written without access to Isaac Sim;
smoke-test the resulting asset with ``scripts/zero_agent.py`` (or open it in
the Isaac Sim USD viewer) before relying on it for training.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_knob_usd(output_path: Path, radius: float, height: float) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    # Fixed base ("Mount") and articulation root are the SAME prim: IsaacLab's
    # fix_root_link handling (schemas.modify_articulation_root_properties)
    # looks for RigidBodyAPI on the ArticulationRootAPI prim itself, not on a
    # descendant, so a separate Xform-with-a-rigid-body-child doesn't work.
    # It must NOT be kinematic -- PhysX articulations reject kinematic links
    # entirely ("Articulations with kinematic bodies are not supported").
    # The base is fixed instead via
    # ArticulationRootPropertiesCfg(fix_root_link=True) in
    # knob_allegro_env_cfg.py, PhysX's proper mechanism for a fixed-base
    # articulation (the same pattern as any robot arm bolted to a table).
    base_path = "/World/KnobBase"
    base_geom = UsdGeom.Cylinder.Define(stage, base_path)
    base_geom.CreateRadiusAttr(radius * 1.5)
    base_geom.CreateHeightAttr(height * 0.3)
    base_geom.CreateAxisAttr(UsdGeom.Tokens.z)
    UsdPhysics.ArticulationRootAPI.Apply(base_geom.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(base_geom.GetPrim())
    UsdPhysics.CollisionAPI.Apply(base_geom.GetPrim())

    # Moving link: the knob itself.
    knob_path = base_path + "/Knob"
    knob_geom = UsdGeom.Cylinder.Define(stage, knob_path)
    knob_geom.CreateRadiusAttr(radius)
    knob_geom.CreateHeightAttr(height)
    knob_geom.CreateAxisAttr(UsdGeom.Tokens.z)
    UsdGeom.XformCommonAPI(knob_geom).SetTranslate(Gf.Vec3d(0.0, 0.0, height))
    # Required whenever a RigidBodyAPI prim is nested under another
    # RigidBodyAPI prim (KnobBase): PhysX otherwise doesn't know whether to
    # treat the child's transform as parent-relative or independent, and
    # warns "missing xformstack reset ... will cause unpredicted results".
    # This makes the translate above an independent world-space offset
    # rather than one chained through KnobBase's (identity, but ambiguous)
    # transform.
    knob_geom.SetResetXformStack(True)
    UsdPhysics.RigidBodyAPI.Apply(knob_geom.GetPrim())
    UsdPhysics.CollisionAPI.Apply(knob_geom.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(knob_geom.GetPrim())
    mass_api.CreateMassAttr(0.05)

    # Revolute joint about Z, free (no drive -- torque is applied from Python
    # each step via KnobDynamics.passive_torque() + contact from the hand).
    # body0 = the base link itself, body1 = Knob (the dynamic link) -- both
    # real rigid bodies, as PhysX articulations require.
    joint = UsdPhysics.RevoluteJoint.Define(stage, base_path + "/knob_joint")
    joint.CreateAxisAttr("Z")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(base_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(knob_path)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, height))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))

    stage.GetRootLayer().Save()
    print(f"[generate_knob_asset] wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("PHYSICS_OT_DATA_DIR", REPO_ROOT / "data")) / "objects" / "knob.usd",
    )
    parser.add_argument("--radius", type=float, default=0.04, help="Knob radius [m] (matches configs/knob.yaml).")
    parser.add_argument("--height", type=float, default=0.03, help="Knob height [m].")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_knob_usd(args.output, args.radius, args.height)


if __name__ == "__main__":
    main()
