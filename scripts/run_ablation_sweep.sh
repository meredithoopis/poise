#!/usr/bin/env bash
# Experiment-0 ablation sweep (sections 25-26): trains all six reward_mode
# arms sequentially and logs each to logs/sweep/seed<SEED>/train_<mode>.log
# (inside the repo, not /tmp). One failed arm does not stop the rest.
#
# Requires $ISAACLAB to be set (see README -- source ~/poise_env.sh first).
#
# SEED is baked into both the log directory and the RSL-RL experiment_name
# (physics_ot_knob_allegro_<mode>_seed<SEED>) -- the spec calls for 3 seeds
# per arm (section 26), reported as mean +/- std, not a single point
# estimate. Re-run this whole script once per seed; evaluate.py/run_
# evaluation_sweep.sh must be pointed at the SAME seed to find the matching
# checkpoint (experiment_name is how evaluate.py locates "latest run", so
# mismatched seeds between train/eval silently evaluate the wrong policy).
#
# Env overrides:
#   SEED=1 NUM_ENVS=512 MAX_ITERATIONS=1000 DEVICE=cuda:0 bash scripts/run_ablation_sweep.sh

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEED="${SEED:-0}"
LOG_DIR="${REPO_ROOT}/logs/sweep/seed${SEED}"
mkdir -p "$LOG_DIR"

MODES=(sparse state_tracking dtw_physics ot_state pointwise_physics physics_ot)
NUM_ENVS="${NUM_ENVS:-512}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1000}"
DEVICE="${DEVICE:-cuda:0}"

if [ -z "${ISAACLAB:-}" ]; then
    echo "ISAACLAB is not set -- source ~/poise_env.sh (or your equivalent) first." >&2
    exit 1
fi

for mode in "${MODES[@]}"; do
    echo "[sweep] starting $mode seed=$SEED ($(date))"
    PYTHONUNBUFFERED=1 $ISAACLAB "${REPO_ROOT}/scripts/train.py" \
        --task PhysicsOT-Knob-Allegro-Direct-v0 \
        --reward_mode "$mode" --seed "$SEED" \
        --experiment_name "physics_ot_knob_allegro_${mode}_seed${SEED}" \
        --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITERATIONS" \
        --headless --device "$DEVICE" \
        > "${LOG_DIR}/train_${mode}.log" 2>&1
    echo "[sweep] $mode seed=$SEED exit code: $? ($(date))"
done

echo "[sweep] done -- logs in ${LOG_DIR}/"
