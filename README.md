# Physics-OT: Stage A (resistive knob)

Cross-embodiment manipulation from egocentric human video via dynamic
physical task transport. This is the Stage-A slice of the full research
program: the resistive knob task, Allegro hand, a scripted source
trajectory standing in for a real human/Shadow-hand demonstration, and the
Experiment-0 2x2 ablation (pointwise/OT x state-only/state+physics, plus
sparse and DTW baselines). Drawer/pushing/6-DoF tasks, real egocentric
video, Franka transfer, and unbalanced OT are deferred until this slice
proves the method works (see the design doc this repo implements).

## Layout

- `physics_ot/` -- pure PyTorch/NumPy package: target-conditioned inverse
  dynamics, the Physics-OT ground cost + batched Sinkhorn, the temporal-OT
  reward, the Experiment-0 baseline distance functions, reward composition,
  and reference-trajectory building. No IsaacLab dependency; fully testable
  without Isaac Sim.
- `source/physics_ot_tasks/` -- an IsaacLab "external" extension package
  (see `IsaacLab/tools/template/templates/external/`) that registers
  `PhysicsOT-Knob-Allegro-Direct-v0`. Only importable/runnable where
  IsaacLab + Isaac Sim are installed.
- `scripts/` -- data generation (`generate_source_demo.py`,
  `build_physics_reference.py`, both pure-Python), IsaacLab-dependent
  sanity/training/eval scripts, and `tools/generate_knob_asset.py`.
- `configs/knob.yaml`, `configs/ppo.yaml` -- physics, OT, reward, and PPO
  hyperparameters.
- `data/` -- demonstrations, the generated knob USD asset, built
  references, and rollout logs.
- `tests/` -- pure-torch unit tests for every piece of the novel math.

## Now, without Isaac Sim (this machine)

```bash
pip install -e .
pip install -e ".[dev]"
pytest tests/

python scripts/generate_source_demo.py
python scripts/build_physics_reference.py
```

The first two commands validate the inverse dynamics, Sinkhorn OT, reward
composition, and reference-building pipeline in isolation. The last two
produce `data/demonstrations/knob_source_001.npz` and
`data/references/knob_target_nominal.npz`, which the environment loads at
reset.

## Later, on the remote IsaacLab machine

Every IsaacLab-dependent script below must run through IsaacLab's own Python
(`isaaclab.sh -p`), not a bare `python` -- that's what resolves `import
isaaclab`, `pxr`, etc. If you already have an IsaacLab conda env (created by
IsaacLab's own installer) activated, bare `python` works too and `isaaclab.sh
-p` is equivalent to it; when in doubt, use the wrapper.

Environment variables set with `export` only last for that one shell session.
Put the ones below in a file you `source` at the start of every session (or
append to `~/.bashrc`) rather than retyping them:

```bash
cat > ~/poise_env.sh << 'EOF'
export ISAACLAB_PATH=/path/to/IsaacLab
export ISAACLAB="${ISAACLAB_PATH}/isaaclab.sh -p"   # use isaaclab.bat on Windows
# Only needed if this machine has no route to NVIDIA's cloud asset server
# (see "Known issues on airgapped/restricted-network machines" below).
export PHYSICS_OT_ALLEGRO_USD_PATH=/path/to/mirrored/allegro_hand_instanceable.usd
EOF
```

```bash
source ~/poise_env.sh

# once: author the knob USD asset (needs pxr / Isaac Sim)
$ISAACLAB scripts/tools/generate_knob_asset.py

# install both packages into the IsaacLab Python environment
$ISAACLAB -m pip install -e .
$ISAACLAB -m pip install -e source/physics_ot_tasks

# rsl-rl-lib is an IsaacLab *extra*, not part of the base install -- skip if
# you've already trained another IsaacLab task with rsl_rl on this machine
$ISAACLAB -m pip install "isaaclab_rl[rsl_rl]"

# sanity checks -- add --headless on a display-less remote box: without it,
# AppLauncher loads the full (non-headless) Kit experience, which pulls in
# extra extensions that need to sync against Omniverse's cloud extension
# registry and fail there if the machine has no route to it
$ISAACLAB scripts/list_envs.py
$ISAACLAB scripts/zero_agent.py --task PhysicsOT-Knob-Allegro-Direct-v0 --num_envs 16 --headless

# reproduce the data pipeline (or scp data/ over from this machine) -- pure
# Python, no pxr needed, so plain `python` is fine here too
python scripts/generate_source_demo.py
python scripts/build_physics_reference.py

# debug-scale training run first (section 18), then scale up
$ISAACLAB scripts/train.py --task PhysicsOT-Knob-Allegro-Direct-v0 \
    --reward_mode physics_ot --num_envs 512 --headless

# Experiment-0 ablation sweep (sections 25-26) -- each run gets its own log
# dir (logs/rsl_rl/<experiment_name>_<reward_mode>/...) via --reward_mode
for mode in sparse state_tracking dtw_physics ot_state pointwise_physics physics_ot; do
    $ISAACLAB scripts/train.py --task PhysicsOT-Knob-Allegro-Direct-v0 --reward_mode $mode --headless
done

$ISAACLAB scripts/evaluate.py --task PhysicsOT-Knob-Allegro-Direct-v0 --num_episodes 100 --headless
```

`scripts/train.py` is a standalone script (`AppLauncher` + RSL-RL's
`OnPolicyRunner`, same pattern as `zero_agent.py`/`evaluate.py`) -- run it
through `$ISAACLAB`, not plain `python`. It does *not* shell out to
`isaaclab.sh train`: that subcommand doesn't exist on every IsaacLab
checkout (some only support `-i/-f/-p/-s/-t/-o/-v/-d/-n/-c/-u`), and the
older per-library `scripts/reinforcement_learning/rsl_rl/train.py` never
imports `physics_ot_tasks`, so our task would never get registered there.
`--reward_mode` is applied by setting `env_cfg.reward_mode` directly in
Python, not via a Hydra CLI override.

## What's deliberately not implemented yet

- Domain randomization (section 19) -- knob physics params are per-env
  tensors already (ready for it) but constant for now.
- Fingertip contact sensing in the observation (left as zeros with a
  `TODO(remote)` in `knob_allegro_env.py`).
- Drawer/planar-pushing/6-DoF stages, Franka target embodiment, unbalanced
  OT, and the real egocentric-video pipeline (Experiment 4).

## Status

**Verified working** (2026-08-27, on a remote H100 node, headless): `pip
install -e .` / `pip install -e source/physics_ot_tasks`,
`generate_knob_asset.py`, `list_envs.py`, and a full `zero_agent.py` run on
both `--device cpu` and `--device cuda:0` (config load -> Allegro hand +
knob spawn -> 200 zero-action steps, including the Physics-OT Sinkhorn
reward path -> clean shutdown). Observation space came back as `Box(-inf,
inf, (16, 42), float32)` and action space as `Box(-inf, inf, (16, 16),
float32)`, matching the design.

**Verified** (2026-09-05): `scripts/train.py`, `scripts/evaluate.py`, and the
full Experiment-0 ablation sweep (all six `reward_mode` arms, 3 seeds each)
ran end-to-end on the remote H100 node. See "Experiment-0 results" below.

## Experiment-0 results (2026-09-05, 3 seeds, resistive-knob + Allegro)

Six `reward_mode` arms (`sparse`, `state_tracking`, `dtw_physics`, `ot_state`,
`pointwise_physics`, `physics_ot`), each trained for 3 seeds and evaluated
over >=100 episodes per seed with `evaluate.py` (which always scores D_POT
with the physics_ot distance regardless of which reward trained the policy,
so the metric is apples-to-apples across arms). Reported as mean +/- std
across seeds (`scripts/aggregate_seeds.py`).

| Arm                  | Success rate    | D_POT             | Time-to-success [s] | Actuation energy      |
|-----------------------|-----------------|--------------------|----------------------|------------------------|
| B0 Sparse             | 0.990 +/- 0.018 | **0.266 +/- 0.031** | 1.42 +/- 0.11        | 2888 +/- 216           |
| B1 State Tracking     | 0.988 +/- 0.017 | 0.534 +/- 0.230    | 1.50 +/- 0.19        | 2449 +/- 642           |
| B2 DTW-Physics        | 0.982 +/- 0.011 | 0.443 +/- 0.226    | 1.68 +/- 0.47        | 2898 +/- 2314          |
| B3 OT-State           | 0.982 +/- 0.022 | 0.423 +/- 0.132    | 1.37 +/- 0.18        | 2821 +/- 624           |
| B4 Pointwise Physics  | 0.961 +/- 0.061 | 0.482 +/- 0.410    | 1.75 +/- 0.47        | 2556 +/- 293           |
| Ours: Physics-OT      | 0.960 +/- 0.030 | 0.473 +/- 0.119    | 1.62 +/- 0.17        | 3090 +/- 1032          |

**Finding: negative for the core hypothesis, on this task as currently
implemented.** Success rate and completion time do not differentiate the
arms (all overlap within 1 std). On D_POT -- the metric Physics-OT is meant
to win on -- the sparse baseline (B0), which never optimizes any physics/OT
term, has the *lowest* mean D_POT with the tightest spread of any arm.
Physics-OT (ours) is second-to-last, and its interval does not overlap B0's.
Several arms (notably B2, B4, Ours) also show high seed-to-seed variance in
actuation energy (std comparable to or exceeding the mean), suggesting
training itself is not fully stable across seeds for this task/reward
combination, independent of the ranking question.

This does not mean the Physics-OT formulation is wrong in general -- only
that, on this single resistive-knob task with these reward weights
(`configs/knob.yaml`: `reward.physics_ot=0.5` against `reward.success_bonus
=10.0`), it is not producing better OT-aligned behavior than a plain sparse
task reward. Plausible untested explanations (not yet investigated, in case
this gets picked back up): the OT shaping term may be too small relative to
the success bonus to meaningfully shape behavior; per-seed training
instability (see the energy variance above) may be masking any real effect;
or the task itself (short, low-DoF, single-joint) may be too easy for sparse
reward to already find a near-reference-optimal solution on its own, leaving
little room for shaping to help. None of these have been checked -- this is
recorded as Experiment-0's result as observed, not a diagnosed root cause.

## Known issues on airgapped/restricted-network machines

If your IsaacLab machine has no outbound route to NVIDIA's cloud asset
server (`omniverse-content-production.s3-us-west-2.amazonaws.com`) or to the
Omniverse extension registry (`ovextensionsprod.blob.core.windows.net`),
expect these:

- **`AppLauncher` without `--headless`** loads the full (non-headless) Kit
  experience, which tries to sync extensions against the cloud registry and
  hangs for minutes before failing. Always pass `--headless`.
- **`ALLEGRO_HAND_CFG`'s default USD path is cloud-hosted** and will hang for
  ~300s per attempt before raising `FileNotFoundError`. Mirror it locally
  with `aws s3 sync --no-sign-request` (the bucket is public-read) and set
  `PHYSICS_OT_ALLEGRO_USD_PATH` -- see `knob_allegro_env_cfg.py` for the
  override mechanism. The same applies to any other Nucleus-hosted asset you
  might reference later; `knob_allegro_env.py`'s `_setup_scene` deliberately
  has no ground plane for this reason (`GroundPlaneCfg`'s default is also
  cloud-hosted, and we don't need one -- the hand and knob are both mounted,
  not resting on a floor).
- **`omni.kit.app` (and anything importing `isaaclab.envs`) is a Kit
  extension**, not a regular package -- it's only importable *after*
  `AppLauncher` is constructed. `zero_agent.py`/`evaluate.py` construct
  `AppLauncher` first and only then import `physics_ot_tasks`/`gymnasium`,
  mirroring IsaacLab's own `scripts/environments/list_envs.py`. Keep that
  ordering if you touch these files -- getting it backwards raises
  `ModuleNotFoundError: No module named 'omni'`.

## Notes on `generate_knob_asset.py`'s USD authoring

Getting a hand-authored (not CAD-exported) PhysX articulation right from raw
`pxr.UsdPhysics` calls took several iterations against a real error message
each time, worth knowing if you need to author another one:

- The `ArticulationRootAPI` prim must **also** carry `RigidBodyAPI` itself --
  IsaacLab's `fix_root_link` handling looks for a rigid body on that exact
  prim, not on a descendant.
- That root/base rigid body must **not** be kinematic -- PhysX articulations
  reject kinematic links outright ("Articulations with kinematic bodies are
  not supported"). Fix the base via
  `ArticulationRootPropertiesCfg(fix_root_link=True)` in the env cfg instead.
- A `RigidBodyAPI` prim nested directly under another `RigidBodyAPI` prim
  needs `xformable.SetResetXformStack(True)`, or PhysX warns about
  "unpredicted results" and fails to build the articulation.

## POISE-OT: Stage 1 (algorithm correctness) and Stage 2 (perception correctness)

A separate, newer package from the `physics_ot`/Stage-A work above -- the
revised POISE-OT paper changes the physical token (5-dim, adds a
torque-impulse/power/dissipation triple), the ground cost (mechanics-induced
primal/dual kinetic-energy geometry, not fixed-`sigma` Euclidean), and the RL
algorithm (an explicit two-critic SAC, `Q_task` + `Q_C`, not PPO with a
single summed reward). `physics_ot`'s Experiment-0 results above are
untouched and still stand as the record of that earlier slice of work.

Stage 1 validates the new mechanism -- OT + mechanics cost + two-critic SAC
-- on data this repo fully controls: a torch-only knob environment with no
PhysX/Isaac Sim and no video. Stage 2 validates the perception pipeline
(HOI4D video -> smoothed `theta(t)` -> target-conditioned inverse dynamics
-> the same token builder Stage 1 uses, unchanged) against HOI4D's own
annotated ground truth, independent of any RL loop.

## Layout (additions)

- `poise_ot/` -- pure PyTorch package (no IsaacLab, no video dependency for
  Stage 1): `knob_env.py` (Eq 15 forward dynamics), `reference_generator.py`,
  `token_builder.py` (Eq 30's 5-channel token), `mech_cost.py` (the
  mechanics-induced ground cost + Sinkhorn + debiased divergence, plus the
  `euclidean_ot`/`pointwise_mechanics` ablation baselines), `alignment.py`
  (the windowed rollout occupancy and `Q_C`'s alignment state `h_t`),
  `two_critic_sac.py` (the actual two-critic SAC agent).
- `poise_ot/video/` -- Stage 2's perception pipeline: `video_loader.py`
  (HOI4D clip loading), `smoothing.py`, `inverse_dynamics.py`,
  `confidence_gate.py`.
- `configs/poise_ot.yaml` -- POISE-OT-specific hyperparameters (mechanics
  cost weights/scales, SAC hyperparameters including the fixed `xi`, and
  Stage 2's assumed target object-dynamics parameters). Physical knob
  parameters themselves (inertia, damping, stiffness, theta0/theta_goal,
  success_threshold) still come from `configs/knob.yaml`.
- `scripts/stage1/` -- `train_stage1.py` (the training loop, also
  importable as a library by the checks script) and
  `run_stage1_checks.py` (the five sanity checks).
- `scripts/stage2/` -- `run_stage2_pipeline.py` (batches HOI4D clips into
  token sequences) and `run_stage2_checks.py` (the four sanity checks).
- `tests/poise_ot/` -- pure-torch unit tests (33 tests, all passing
  locally, no Isaac Sim needed) mirroring `tests/`'s existing style.

## Running Stage 1 (no Isaac Sim needed -- runs on this machine)

```bash
pip install -e .            # now also pulls in gymnasium
pytest tests/poise_ot/      # 33 tests, torch-only

# quick smoke run (no meaningful learning yet, just confirms the pipeline runs)
python scripts/stage1/train_stage1.py --num_envs 8 --episodes 5

# a real training run, sweeping the cost_mode ablation
python scripts/stage1/train_stage1.py --num_envs 16 --episodes 40 --cost_mode mechanics_ot
python scripts/stage1/train_stage1.py --num_envs 16 --episodes 40 --cost_mode euclidean_ot
python scripts/stage1/train_stage1.py --num_envs 16 --episodes 40 --cost_mode pointwise_mechanics

# the five sanity checks -- this is Stage 1's actual "definition of done"
python scripts/stage1/run_stage1_checks.py --check all --num_envs 8 --episodes 15
# or one at a time while debugging:
python scripts/stage1/run_stage1_checks.py --check 3 --num_envs 8 --episodes 15
```

Each check prints PASS/FAIL with the real numbers behind the verdict (not
just a boolean) and, if `matplotlib` is installed, saves a plot to
`logs/stage1/`. `--check 1` (token sanity) and `--check 2` (cost sanity)
need no training and run in seconds; `--check 3/4/5` train real SAC agents
and take longer -- shrink `--num_envs`/`--episodes` while iterating.

**Known nuance, not a bug**: check 3 (gradient sanity) can show *growing*
(not decreasing) raw critic loss even while the policy is learning
correctly. `Eq task_reward`'s success bonus is awarded every single step
the knob is at the goal (uncapped), so as the policy improves from failing
to reliably solving the task, the true return -- and so the critic's
regression target -- grows roughly 40x (observed: ~-400 to ~+1750 over one
run, against a ~1800 theoretical ceiling). Raw MSE loss reflects that
growing target scale, not instability; gradient clipping (already applied,
`configs/poise_ot.yaml`'s `sac.grad_clip_norm`) doesn't change this, which
is itself evidence it's a scale effect rather than exploding gradients.
Cross-check against check 4's success-rate curve (and the episode-return
history in `train_stage1.py`'s returned metrics) before concluding a real
training problem exists.

## Running Stage 2 (needs an actual HOI4D download)

**External prerequisite**: this repo does not bundle HOI4D
(https://hoi4d.github.io). See `docs/HOI4D_DOWNLOAD.md` for what to
request (category `Safe`/task `T1`, why) and how to fetch just the needed
files (`objpose/*.json` + `mobility_v2.json`) out of the annotations/CAD
zips without unpacking the full ~20GB archives, via
`scripts/stage2/select_hoi4d_clips.py`. That script's `--output` produces a
clip-list file the pipeline below consumes; `--out_dir` is what
`POISE_OT_HOI4D_ROOT` should point at:

```bash
python scripts/stage2/select_hoi4d_clips.py \
    --release_txt /path/to/HOI4D-Instructions/release.txt \
    --category C6 --task T1 --num_clips 12 --output data/clip_list.txt \
    --annotations_source /path/to/HOI4D_annotations.zip \
    --cad_source /path/to/HOI4D_CAD_Model_for_release.zip \
    --out_dir data/hoi4d_raw --probe --fetch

export POISE_OT_HOI4D_ROOT=data/hoi4d_raw
python scripts/stage2/run_stage2_pipeline.py --clip_list data/clip_list.txt
python scripts/stage2/run_stage2_checks.py --clip_list data/clip_list.txt \
    --flagged_clips <clip_id_with_visible_occlusion> <another_clip_id>
```

`video_loader.py`'s raw-format parser (`load_clip_raw`) is **verified
against real HOI4D data** (category `Safe`) -- it computes the moving
part's rotation *relative to* the object's static base part (HOI4D is
egocentric video, so a part's raw absolute rotation is contaminated by
camera motion) and projects that onto the joint's axis from
`mobility_v2.json`. See `docs/HOI4D_DOWNLOAD.md`'s "Resolved" section for
what this fixed and what's still unverified for other categories. If a
clip's parser fails for a different category/label convention, preprocess
it into the simple `{t, theta}` npz format instead and use `--npz_clips`:

```bash
python scripts/stage2/run_stage2_checks.py --npz_clips clip1.npz clip2.npz
```

Only HOI4D's `Safe` category (a rotational latch/knob joint) maps directly
onto the reused `KnobDynamics` model; `StorageFurniture` clips may be a
prismatic sliding drawer instead of a revolute door, which `KnobDynamics`
does not model -- `run_stage2_pipeline.py` checks each clip's `joint_type`
and skips (rather than silently mis-processes) any that aren't `"revolute"`.

Every Stage-2 script run prints the *assumed* target object-dynamics
parameters (`configs/poise_ot.yaml`'s `stage2:` section) before doing
anything else -- HOI4D provides no ground-truth mass/inertia/damping/
stiffness/friction, so these values are never measured, only assumed, and
that assumption is logged explicitly every time rather than silently baked
into results.

## POISE-OT: Stage 3 (real IsaacLab simulator, Allegro hand + knob)

**Needs Isaac Sim/Isaac Lab and a GPU -- not runnable on this dev machine.**
Everything below was written by mirroring the already-validated
`knob_allegro` (Physics-OT/Stage-A/Experiment-0) task as closely as
possible and reusing its assets, but **has not been run against real Isaac
Sim** -- the first real test of this code is launching it on your machine,
not something completed here. Stage 3 only covers steps 6-11 of the plan
(bare env through the first small-scale regression check); steps 12-16
(scale-up, domain randomization, the real-HOI4D-reference swap, the
physics sweep, and the full Phase-I ablation table) are not implemented --
each is gated on the previous step actually passing on real hardware,
which can't happen from here.

**Before touching Stage 3**, three prerequisite fixes from the checklist
were verified/applied against the actual code (not just assumed done):
- Two-critic SAC's target-network Polyak averaging: already present in
  `poise_ot/two_critic_sac.py`, no change needed.
- Angle unwrapping before differentiation: real bug, fixed in
  `poise_ot/video/video_loader.py`'s `load_clip_raw` (`Rotation.as_rotvec()`'s
  branch cut at near-zero rotation was producing spurious jumps -- exactly
  what showed up as a blow-up on `N03_S247_s01`).
- `run_stage2_checks.py`'s `differentiation_flagged` threshold: was
  comparing incompatible units (variance in rad²/s⁴ against a raw-angle
  std in rad) regardless of the multiplier's value -- rewritten to compare
  like-for-like (both in rad/s²).

Two items are **not done** and need your own machine's output, not a guess:
recalibrating `confidence_gate.py`'s formula against real clean-vs-corrupted
clips, and re-running Stage 1's check 3 to confirm the critic loss plateaus
past step 2750 with the (already-present) target-network fix. Send me the
real numbers from either and I'll act on them -- I did not fabricate a fix
for either here.

### Layout (additions)

- `source/physics_ot_tasks/physics_ot_tasks/direct/knob_allegro_poise/` --
  a new IsaacLab task, sibling to (not a modification of) `knob_allegro/`.
  Reuses that task's Allegro hand and knob USD assets and physics wiring
  (passive-torque application, inverse-dynamics `tau` recovery) verbatim,
  but returns *only* `r_task` (Eq task_reward) as the step reward and
  exposes raw state via `extras` for two-critic SAC's external alignment
  loop -- unlike `knob_allegro_env.py`, which blends task + shaping reward
  into one PPO-compatible scalar via `compose_reward`. Registered as
  `PoiseOT-Knob-Allegro-Direct-v0`.
- `scripts/stage3/train_stage3.py` -- step 11's training loop: the real env
  above, driving `poise_ot.alignment`/`poise_ot.mech_cost`/
  `poise_ot.two_critic_sac` completely unchanged from Stage 1 (that's the
  entire point of the regression-check gate: if training here behaves
  differently from Stage 1's toy env, the bug is in the simulator/env
  wiring, not in this already-validated code).
- `scripts/stage3/run_stage3_regression_check.py` -- step 11's gate,
  comparing a `train_stage1.py --output ...` history against a
  `train_stage3.py --output ...` history for comparable critic-loss
  trend/magnitude (not exact match -- the environments differ).


  

### Running Stage 3

```bash
# 1. Stage-1 reference run (same synthetic theta_H(t), for the regression-check comparison)
python scripts/stage1/train_stage1.py --episodes 30 --output logs/stage1_history.json

# 2. Bare launch test first (step 6) -- confirm the env launches and steps at all
#    before trusting anything built on top of it:
$ISAACLAB scripts/zero_agent.py --task PoiseOT-Knob-Allegro-Direct-v0 --num_envs 16

# 3. Step 11's small-scale regression check
$ISAACLAB scripts/stage3/train_stage3.py --task PoiseOT-Knob-Allegro-Direct-v0 \
    --num_envs 16 --episodes 20 --cost_mode mechanics_ot --headless \
    --output logs/stage3_history.json

python scripts/stage3/run_stage3_regression_check.py \
    --stage1_history logs/stage1_history.json --stage3_history logs/stage3_history.json
```

If step 2 fails to launch, that's diagnostic on its own (asset paths,
`PHYSICS_OT_ALLEGRO_USD_PATH`/`PHYSICS_OT_DATA_DIR` overrides, IsaacLab
version) -- fix that before running step 3. If step 3's check fails, per
the plan: the bug is in `knob_allegro_poise_env.py`/
`knob_allegro_poise_env_cfg.py`, not in the untouched alignment/cost/SAC
code. Send me whatever actually happens (a traceback, or the check's
printed numbers) -- I can't predict which from here, and I'd rather fix a
real failure than have guessed around one.
