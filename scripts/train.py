"""Train a Physics-OT policy with RSL-RL PPO.

Standalone script (same ``AppLauncher``-first pattern as ``zero_agent.py``/
``evaluate.py``) rather than a wrapper around IsaacLab's ``isaaclab.sh
train`` CLI: that subcommand doesn't exist on every IsaacLab checkout (some
only have ``-i/-f/-p/-s/-t/-o/-v/-d/-n/-c/-u``), and the older per-library
``scripts/reinforcement_learning/rsl_rl/train.py`` never imports
``physics_ot_tasks``, so our task would never get registered there either.

Example:
    $ISAACLAB scripts/train.py --task PhysicsOT-Knob-Allegro-Direct-v0 \\
        --reward_mode physics_ot --num_envs 512 --headless
"""

from __future__ import annotations

import argparse
import importlib.metadata
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="PhysicsOT-Knob-Allegro-Direct-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument(
    "--reward_mode",
    type=str,
    default=None,
    choices=["sparse", "state_tracking", "dtw_physics", "ot_state", "pointwise_physics", "physics_ot"],
    help="Experiment-0 ablation arm (sections 25-26); overrides env_cfg.reward_mode directly.",
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default=None,
    help="Overrides the log directory name; defaults to '<task's default>_<reward_mode>' when --reward_mode is set.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import physics_ot_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


def main() -> None:
    from rsl_rl.runners import OnPolicyRunner

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.reward_mode is not None:
        env_cfg.reward_mode = args_cli.reward_mode

    agent_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent)
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    elif args_cli.reward_mode is not None:
        agent_cfg.experiment_name = f"{agent_cfg.experiment_name}_{args_cli.reward_mode}"
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
    env_cfg.seed = agent_cfg.seed

    log_root = Path("logs", "rsl_rl", agent_cfg.experiment_name).resolve()
    log_dir = log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"[train] logging to {log_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
