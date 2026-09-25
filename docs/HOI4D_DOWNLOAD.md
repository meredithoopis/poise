# Getting HOI4D clips for Stage 2

Source: [`HOI4D-Instructions`](https://github.com/leolyliu/HOI4D-Instructions)
(docs + processing scripts only, no bundled data), project page
https://hoi4d.github.io/, latest info/challenge at https://www.hoi4d.top.

There is no scriptable download in that repo -- getting the actual RGB-D
clips requires going through HOI4D's own access process on those sites
(registration/license agreement is typical for this kind of dataset). This
doc tells you exactly *what* to request and *which* clips to pick once you
have access; it can't do the request for you.

## What to request

Category/task combo, and why:

| Category | Task | Joint type | Released clips | Use |
|---|---|---|---|---|
| **C6 Safe** | **T1** "Open and close the door" | revolute | 69 | **primary** |
| C4 StorageFurniture | **T2** "Open and close the door" | revolute | 62 | secondary/diversity |
| C4 StorageFurniture | T1 "Open and close the drawer" | **prismatic** | 51 | **skip** -- `poise_ot` only models a single revolute DOF (Eq 15); `run_stage2_pipeline.py` rejects non-revolute clips outright |

`C6`/`C4` and `T1`/`T2` are HOI4D's own category/task IDs (see
`HOI4D-Instructions/README.md`'s category mapping and
`definitions/task/task_definitions.csv`). C6/T2 and C6/T3 ("put/take
something") and C4/T3-T4 ("put the drink in ...") mix manipulation with the
joint motion and are noisier matches for a clean single-DOF trace -- skip
those for now too.

## How many clips

**~10-15 total is enough**, mostly `C6/T1` (Safe). Split:
- ~7-10 clean-looking clips, for checks 1-3 (trajectory/differentiation/token
  sanity) and to build the token dataset itself
  (`run_stage2_pipeline.py --num_clips 10`).
- **2-3 clips you can see have a tracking problem** (hand occluding the
  door/dial, fast motion, blur) -- required for check 4 (confidence-gate
  sanity), which needs a `--flagged_clips` list to contrast against the
  clean ones. You pick these by eye after running check 1's trajectory plots.

Add a few `C4/T2` clips later only if you want cross-category diversity --
not required to pass any of the four checks.

## Which files you actually need, per clip

Per HOI4D-Instructions' `README.md`, a clip lives at
`{camera}/{human}/{category}/{instance}/{room}/{layout}/{task}/` and contains
several annotation streams. For Stage 2 you need:

- **`objpose/*.json`** -- per-frame part pose (the raw material `theta(t)`
  is derived from). Required.
- **`{category}/{instance}/mobility_v2.json`** -- the CAD model's joint
  annotation (axis origin/direction, rotation limits for the revolute
  joint). This comes from HOI4D's *object CAD model* release, indexed by
  category + instance ID (`N*`), not from the per-clip session folder --
  fetch it once per distinct object instance your chosen clips use, not
  once per clip. Required to turn `objpose` rotations into a scalar joint
  angle.
- **`align_rgb/image.mp4`** -- only needed if you want to eyeball a clip
  before deciding it's "clean" vs. "flagged" for check 4. Decode with
  `HOI4D-Instructions/utils/decode.py` (needs `ffmpeg`; note it decodes at
  **15 fps**, not 30 -- `video_loader.py`'s current `fps=30.0` default is
  wrong and needs updating once real clips are in hand).
- Not needed for Stage 2: `align_depth`, `3Dseg`, `2Dseg`, hand-pose
  `.pickle` files.

## Fetching, once you have the zips (or a working URL)

`select_hoi4d_clips.py` (added alongside this doc) filters `release.txt`
down to the clips to use, and can extract just the needed files straight
out of `HOI4D_annotations.zip`/`HOI4D_CAD_Model_for_release.zip` -- either
streamed from an unauthenticated, range-request-capable URL, or from a zip
you already downloaded locally (the more reliable path in practice -- see
the script's own docstring for both). Save the selected clip list with
`--output`, you'll need it for the pipeline step below:

```bash
python scripts/stage2/select_hoi4d_clips.py \
    --release_txt /path/to/HOI4D-Instructions/release.txt \
    --category C6 --task T1 --num_clips 12 \
    --output data/clip_list.txt \
    --annotations_source /path/to/HOI4D_annotations.zip \
    --cad_source /path/to/HOI4D_CAD_Model_for_release.zip \
    --out_dir data/hoi4d_raw --probe --fetch
```

`--out_dir` (e.g. `data/hoi4d_raw`) is what `POISE_OT_HOI4D_ROOT` should
point to -- it contains `HOI4D_annotations/` and
`HOI4D_CAD_Model_for_release/` after `--fetch`. Then:

```bash
POISE_OT_HOI4D_ROOT=data/hoi4d_raw python scripts/stage2/run_stage2_pipeline.py \
    --clip_list data/clip_list.txt
POISE_OT_HOI4D_ROOT=data/hoi4d_raw python scripts/stage2/run_stage2_checks.py \
    --clip_list data/clip_list.txt
```

## Resolved: video_loader.py now parses the real format

`load_clip_raw()` was originally written against HOI4D's *documented*
schema (a guess), not a real clip -- since fixed and verified against real
Safe-category data (clip `ZY20210800001/H1/C6/N03/S247/s01/T1`, instance
`003`). Two things the real data revealed that changed the design:

- HOI4D is **egocentric video**, so a moving part's raw per-frame
  `rotation` is contaminated by camera motion -- in the real sample, the
  *static* part's rotation drifted by about as much as the *moving* part's
  between adjacent frames, before the door had actually started opening.
  `load_clip_raw()` now computes the door's rotation **relative to the
  static base part** (which cancels the shared camera egomotion) and
  projects that onto the joint's axis from `mobility_v2.json`, rather than
  reading the moving part's absolute rotation directly.
- The CAD zip's real layout is `articulated/{CategoryName}/{zero-padded-
  3-digit-id}/mobility_v2.json` (e.g. `articulated/Safe/003/`) -- spelled-
  out category name, not `C6`, and no `N` prefix on the instance id. Both
  `video_loader.py` and `select_hoi4d_clips.py` now convert between
  release.txt's `C6/N03` ids and this format (`CATEGORY_ID_TO_NAME` +
  `_cad_instance_path()`/`resolve_clip_paths()`).

Part matching (`Safebox`/`Safedoor` -> static/moving) is by annotation
label suffix, verified for **Safe only**. If you later add `StorageFurniture`
clips, inspect one real `objpose/*.json` first -- the label convention
there is unverified and may need `load_clip_raw(..., moving_suffix=...,
static_suffix=...)` adjusted.
