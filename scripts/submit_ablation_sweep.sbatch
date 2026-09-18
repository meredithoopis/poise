#!/bin/bash
#SBATCH --job-name=physics_ot_sweep
#SBATCH --output=%x_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
# --- Cluster-specific: uncomment/adjust as needed (ask your admin if unsure) ---
# #SBATCH --partition=gpu
# #SBATCH --account=your_account
# Some clusters use --gpus=1 or --gres=gpu:h100:1 instead of --gres=gpu:1.

set -u

source ~/poise_env.sh   # sets ISAACLAB_PATH, ISAACLAB, PHYSICS_OT_ALLEGRO_USD_PATH
conda activate issac    # your IsaacLab conda env name
cd ~/mycode/poise

bash scripts/run_ablation_sweep.sh
