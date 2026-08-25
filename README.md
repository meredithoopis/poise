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

```bash
export ISAACLAB_PATH=/path/to/IsaacLab   # or PowerShell: $env:ISAACLAB_PATH

# once: author the knob USD asset (needs pxr / Isaac Sim)
python scripts/tools/generate_knob_asset.py

# install both packages into the IsaacLab Python environment
python -m pip install -e .
python -m pip install -e source/physics_ot_tasks

# sanity checks
python scripts/list_envs.py
python scripts/zero_agent.py --task PhysicsOT-Knob-Allegro-Direct-v0 --num_envs 16

# reproduce the data pipeline (or scp data/ over from this machine)
python scripts/generate_source_demo.py
python scripts/build_physics_reference.py

# debug-scale training run first (section 18), then scale up
python scripts/train.py --task PhysicsOT-Knob-Allegro-Direct-v0 \
    --reward_mode physics_ot --num_envs 512

# Experiment-0 ablation sweep (sections 25-26)
for mode in sparse state_tracking dtw_physics ot_state pointwise_physics physics_ot; do
    python scripts/train.py --task PhysicsOT-Knob-Allegro-Direct-v0 --reward_mode $mode
done

python scripts/evaluate.py --task PhysicsOT-Knob-Allegro-Direct-v0 --num_episodes 100
```

`scripts/train.py` is a thin wrapper around IsaacLab's unified
`isaaclab.sh train --rl_library rsl_rl --task <TASK>` CLI (the older
per-library `scripts/reinforcement_learning/rsl_rl/train.py` entry point is
deprecated in this IsaacLab checkout) that also forwards `--reward_mode` as
the Hydra override `env.reward_mode=<mode>`.

## What's deliberately not implemented yet

- Domain randomization (section 19) -- knob physics params are per-env
  tensors already (ready for it) but constant for now.
- Fingertip contact sensing in the observation (left as zeros with a
  `TODO(remote)` in `knob_allegro_env.py`).
- Drawer/planar-pushing/6-DoF stages, Franka target embodiment, unbalanced
  OT, and the real egocentric-video pipeline (Experiment 4).

## First things to verify on the remote machine

Nothing in `source/physics_ot_tasks/` or the IsaacLab-dependent scripts has
been executed (no Isaac Sim on the machine this was written on). Before
trusting a training run:

1. `generate_knob_asset.py` actually produces a loadable articulation with a
   free `knob_joint` -- open it in the Isaac Sim USD viewer if anything looks
   off.
2. `zero_agent.py` runs without shape/API errors (observation dim,
   `find_joints`/`find_bodies` name matches, `set_joint_effort_target` on an
   `actuators={}` articulation).
3. `evaluate.py` reaches into a few private env attributes
   (`unwrapped._knob`, `_knob_joint_idx`, `_theta_goal`, `_prev_distance`) --
   fine for a diagnostic script, but re-check those names if
   `knob_allegro_env.py` changes.
