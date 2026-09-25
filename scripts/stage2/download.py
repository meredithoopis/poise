"""Select which HOI4D clips/instances to use for Stage 2, and (optionally)
pull *only* their needed files out of the annotations/CAD zips -- either
streamed via HTTP Range requests from a URL, or extracted from an
already-downloaded local zip, without ever fully unpacking the ~20GB/~1.4GB
archives. See docs/HOI4D_DOWNLOAD.md for why C6/T1 (Safe, revolute door) is
the recommended default and C4/T1 (StorageFurniture drawer, prismatic) is
explicitly excluded, and for what each zip is expected to contain.

Three independent steps:

1. **Select** -- filter release.txt down to N clip paths for a
   category/task (always runs; this part needs no network access).
2. **Fetch** (``--fetch``) -- for each selected clip, extract just its
   ``objpose/*.json`` out of the annotations zip; for each distinct object
   instance those clips use, extract just its ``mobility_v2.json`` out of
   the CAD-model zip. ``--annotations_source``/``--cad_url`` accept either
   an http(s) URL (streamed via the ``remotezip`` package, only the needed
   bytes are downloaded -- requires Range-request support and an
   unauthenticated link, e.g. fails behind a Microsoft/OneDrive login
   wall) or a local path to a zip you already downloaded (through a
   browser, if the link needs a login a plain HTTP client can't complete --
   opened directly with the stdlib ``zipfile``, no extra dependency).
3. **Probe** (``--probe``) -- inspect a zip's *real* internal namelist
   before trusting the matcher; the matching logic
   (``_match_objpose_members``/``_match_mobility_members``) was written
   against the documented release.txt-style path, not a real HOI4D zip, so
   always probe first.

Usage (selection only, no network):
    python scripts/stage2/select_hoi4d_clips.py \\
        --release_txt /path/to/HOI4D-Instructions/release.txt \\
        --category C6 --task T1 --num_clips 12

Usage (probe + fetch from a local zip you already downloaded):
    python scripts/stage2/select_hoi4d_clips.py \\
        --release_txt ... --category C6 --task T1 --num_clips 12 \\
        --annotations_source /path/to/HOI4D_annotations.zip \\
        --cad_source /path/to/HOI4D_CAD_Model_for_release.zip \\
        --out_dir data/hoi4d_raw --probe --fetch

Usage (stream straight from an unauthenticated direct-download URL):
    python scripts/stage2/select_hoi4d_clips.py \\
        --release_txt ... --category C6 --task T1 --num_clips 12 \\
        --annotations_source "https://.../HOI4D_annotations.zip" \\
        --cad_source "https://.../HOI4D_CAD_Model_for_release.zip" \\
        --out_dir data/hoi4d_raw --fetch
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def filter_clips(release_txt: Path, category: str, task: str) -> list[str]:
    lines = [line.strip() for line in release_txt.read_text().splitlines() if line.strip()]
    return [line for line in lines if f"/{category}/" in line and line.endswith(f"/{task}")]


def distinct_instances(clips: list[str]) -> list[str]:
    """clip = 'ZY.../H../C6/N03/S.../s../T1' -> instance key 'C6/N03', per
    HOI4D-Instructions' README (CAD models are indexed by category+instance
    only, not by the full clip/session path)."""
    seen: list[str] = []
    for clip in clips:
        parts = clip.split("/")
        key = f"{parts[2]}/{parts[3]}"  # category, instance
        if key not in seen:
            seen.append(key)
    return seen


def _open_zip(source: str):
    """``source`` is either an http(s) URL (streamed via remotezip, only the
    needed member files are downloaded -- requires Range support and an
    unauthenticated link) or a local path to an already-downloaded zip
    (opened directly with zipfile -- use this after pulling the file through
    a browser login wall that no HTTP client can get past)."""
    if source.startswith("http://") or source.startswith("https://"):
        try:
            from remotezip import RemoteZip
        except ImportError as exc:
            raise SystemExit(
                "remotezip is required for a URL source (pip install remotezip). "
                "It reads a zip's central directory via HTTP Range requests so only "
                "the needed member files are downloaded, not the whole archive. "
                "This needs the server behind the URL to support Range requests and "
                "the link to need no login -- if it raises immediately or redirects "
                "to a sign-in page, download the zip through a browser instead and "
                "pass its local path here."
            ) from exc
        return RemoteZip(source)

    import zipfile

    path = Path(source)
    if not path.exists():
        raise SystemExit(f"local zip not found: {path}")
    return zipfile.ZipFile(path, "r")


def _match_objpose_members(namelist: list[str], clip: str) -> list[str]:
    needle = f"{clip}/objpose/"
    return [n for n in namelist if needle in n]


# HOI4D-Instructions README's C* -> category-name mapping. The CAD-model zip
# is organized by spelled-out category name (confirmed against a real
# HOI4D_CAD_Model_for_release.zip: e.g. "articulated/Safe/068/mobility_v2.json"),
# not by the C6/C4-style category ID used in release.txt/clip paths.
CATEGORY_ID_TO_NAME = {
    "C1": "ToyCar", "C2": "Mug", "C3": "Laptop", "C4": "StorageFurniture",
    "C5": "Bottle", "C6": "Safe", "C7": "Bowl", "C8": "Bucket", "C9": "Scissors",
    "C11": "Pliers", "C12": "Kettle", "C13": "Knife", "C14": "TrashCan",
    "C17": "Lamp", "C18": "Stapler", "C20": "Chair",
}


def _cad_instance_path(instance: str) -> str:
    """'C6/N03' (release.txt-style category-id/instance-id, as produced by
    distinct_instances()) -> 'Safe/003' (the CAD zip's real category-name/
    zero-padded-3-digit-instance-id layout)."""
    cat_id, inst_id = instance.split("/")
    name = CATEGORY_ID_TO_NAME.get(cat_id, cat_id)
    num = int(inst_id.lstrip("N"))
    return f"{name}/{num:03d}"


def _match_mobility_members(namelist: list[str], instance: str) -> list[str]:
    needle = f"{_cad_instance_path(instance)}/mobility_v2.json"
    return [n for n in namelist if n.endswith(needle)]


def probe(annotations_source: str | None, cad_source: str | None, clips: list[str], instances: list[str]) -> None:
    if annotations_source:
        print(f"\n[probe] opening annotations zip ({annotations_source})...")
        with _open_zip(annotations_source) as z:
            namelist = z.namelist()
            print(f"[probe] {len(namelist)} entries total. First 10:")
            for n in namelist[:10]:
                print(f"    {n}")
            clip = clips[0]
            matches = _match_objpose_members(namelist, clip)
            print(f"[probe] matches for clip {clip}: {len(matches)}")
            for n in matches[:5]:
                print(f"    {n}")
            if not matches:
                print(f"[probe] WARNING: no entries contain '{clip}/objpose/' -- the zip's internal "
                      f"layout doesn't match the assumed release.txt-style path. Inspect the 'First 10' "
                      f"list above and adjust _match_objpose_members().")
    if cad_source:
        print(f"\n[probe] opening CAD-model zip ({cad_source})...")
        with _open_zip(cad_source) as z:
            namelist = z.namelist()
            print(f"[probe] {len(namelist)} entries total. First 10:")
            for n in namelist[:10]:
                print(f"    {n}")
            inst = instances[0]
            matches = _match_mobility_members(namelist, inst)
            print(f"[probe] matches for instance {inst}: {len(matches)}")
            for n in matches[:5]:
                print(f"    {n}")
            if not matches:
                print(f"[probe] WARNING: no entries end with '{inst}/mobility_v2.json' -- inspect the "
                      f"'First 10' list above and adjust _match_mobility_members().")


def fetch(annotations_source: str | None, cad_source: str | None, clips: list[str], instances: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if annotations_source:
        print(f"\n[fetch] extracting objpose/*.json for {len(clips)} clips from {annotations_source}...")
        with _open_zip(annotations_source) as z:
            namelist = z.namelist()
            wanted: list[str] = []
            for clip in clips:
                matches = _match_objpose_members(namelist, clip)
                if not matches:
                    print(f"  WARNING: no objpose files found for {clip} -- skipped (run --probe to debug)")
                wanted.extend(matches)
            print(f"[fetch] extracting {len(wanted)} objpose files -> {out_dir}")
            z.extractall(members=wanted, path=out_dir)

    if cad_source:
        print(f"\n[fetch] extracting mobility_v2.json for {len(instances)} instances from {cad_source}...")
        with _open_zip(cad_source) as z:
            namelist = z.namelist()
            wanted = []
            for inst in instances:
                matches = _match_mobility_members(namelist, inst)
                if not matches:
                    print(f"  WARNING: no mobility_v2.json found for {inst} -- skipped (run --probe to debug)")
                wanted.extend(matches)
            print(f"[fetch] extracting {len(wanted)} CAD files -> {out_dir}")
            z.extractall(members=wanted, path=out_dir)

    print(f"\n[fetch] done. Set POISE_OT_HOI4D_ROOT={out_dir} (or point video_loader.py at it) for the "
          f"Stage-2 pipeline. Remember to fix load_clip_raw()'s parser against these real files before "
          f"running it (see docs/HOI4D_DOWNLOAD.md's 'Known follow-up' section).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release_txt", type=Path, required=True)
    parser.add_argument("--category", type=str, default="C6", help="C6=Safe, C4=StorageFurniture (see HOI4D README's mapping).")
    parser.add_argument("--task", type=str, default="T1", help="C6/T1='open and close the door' (revolute); avoid C4/T1 (drawer, prismatic).")
    parser.add_argument("--num_clips", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write the selected clip list, one per line.")

    parser.add_argument("--annotations_source", type=str, default=None,
                         help="Either an unauthenticated, range-request-capable URL to HOI4D_annotations.zip, "
                              "or a local path to it after downloading through a browser.")
    parser.add_argument("--cad_source", type=str, default=None,
                         help="Either an unauthenticated, range-request-capable URL to "
                              "HOI4D_CAD_Model_for_release.zip, or a local path to it after downloading "
                              "through a browser.")
    parser.add_argument("--out_dir", type=Path, default=Path("data/hoi4d_raw"), help="Where --fetch extracts files to.")
    parser.add_argument("--probe", action="store_true", help="Inspect the zips' real internal namelist/matches without extracting anything.")
    parser.add_argument("--fetch", action="store_true", help="Actually extract the matched files.")
    args = parser.parse_args()

    matches = filter_clips(args.release_txt, args.category, args.task)
    if not matches:
        raise SystemExit(f"no clips found for category={args.category} task={args.task} in {args.release_txt}")

    random.Random(args.seed).shuffle(matches)
    selected = sorted(matches[: args.num_clips])
    instances = distinct_instances(selected)

    print(f"[select_hoi4d_clips] {len(matches)} clips available for {args.category}/{args.task}, "
          f"selected {len(selected)} across {len(instances)} distinct object instances:")
    for clip in selected:
        print(f"  {clip}")

    if args.output:
        args.output.write_text("\n".join(selected) + "\n")
        print(f"[select_hoi4d_clips] wrote {args.output}")

    if args.probe:
        probe(args.annotations_source, args.cad_source, selected, instances)
    if args.fetch:
        if not (args.annotations_source or args.cad_source):
            raise SystemExit("--fetch needs at least one of --annotations_source / --cad_source")
        fetch(args.annotations_source, args.cad_source, selected, instances, args.out_dir)
    if not args.probe and not args.fetch:
        print("\nNext: pass --annotations_source/--cad_source (a URL or a local zip path) with --probe "
              "first to confirm the zips' real internal layout, then --fetch to extract just these "
              "files. See docs/HOI4D_DOWNLOAD.md.")


if __name__ == "__main__":
    main()
