#!/usr/bin/env bash
# Evaluates all six Experiment-0 checkpoints (sections 25-26, 32) and writes
# one JSON result per arm to logs/sweep/seed<SEED>/eval_<mode>.json. Run this
# after run_ablation_sweep.sh has produced checkpoints for all six modes AT
# THE SAME SEED.
#
# Requires $ISAACLAB to be set (see README -- source ~/poise_env.sh first).
#
# SEED must match the seed used for run_ablation_sweep.sh -- it selects the
# checkpoint via experiment_name (physics_ot_knob_allegro_<mode>_seed<SEED>),
# same as train.py. Once seeds 0/1/2 have all been trained+evaluated, combine
# them with scripts/aggregate_seeds.py.
#
# Env overrides:
#   SEED=1 NUM_ENVS=256 NUM_EPISODES=100 DEVICE=cuda:0 bash scripts/run_evaluation_sweep.sh

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEED="${SEED:-0}"
LOG_DIR="${REPO_ROOT}/logs/sweep/seed${SEED}"
mkdir -p "$LOG_DIR"

MODES=(sparse state_tracking dtw_physics ot_state pointwise_physics physics_ot)
NUM_ENVS="${NUM_ENVS:-256}"
NUM_EPISODES="${NUM_EPISODES:-100}"
DEVICE="${DEVICE:-cuda:0}"

if [ -z "${ISAACLAB:-}" ]; then
    echo "ISAACLAB is not set -- source ~/poise_env.sh (or your equivalent) first." >&2
    exit 1
fi

for mode in "${MODES[@]}"; do
    echo "[eval-sweep] evaluating $mode seed=$SEED ($(date))"
    PYTHONUNBUFFERED=1 $ISAACLAB "${REPO_ROOT}/scripts/evaluate.py" \
        --task PhysicsOT-Knob-Allegro-Direct-v0 \
        --reward_mode "$mode" --num_envs "$NUM_ENVS" --num_episodes "$NUM_EPISODES" \
        --experiment_name "physics_ot_knob_allegro_${mode}_seed${SEED}" \
        --headless --device "$DEVICE" \
        --output "${LOG_DIR}/eval_${mode}.json" \
        > "${LOG_DIR}/eval_${mode}.log" 2>&1
    echo "[eval-sweep] $mode seed=$SEED exit code: $? ($(date))"
done

echo
echo "[eval-sweep] done -- seed ${SEED} summary:"
python "${REPO_ROOT}/scripts/summarize_evaluation_sweep.py" --log_dir "$LOG_DIR"
