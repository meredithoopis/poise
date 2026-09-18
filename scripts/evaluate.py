"""Evaluate a trained Physics-OT checkpoint: success rate, D_POT, completion
time, and actuation energy (section 32), for the Experiment-0 ablation table.

Uses ``AppLauncher`` + ``parse_env_cfg``/``load_cfg_from_registry`` (the same
reliable, low-level pattern as ``zero_agent.py``) plus RSL-RL's runner/env
wrapper, rather than the deprecated ``scripts/reinforcement_learning/rsl_rl/
play.py`` entry point or the higher-level preset-CLI helpers (``resolve_task_
config``/``launch_simulation``/``setup_preset_cli``), which import IsaacLab's
Kit-dependent modules before ``AppLauncher`` is constructed and fail with
``ModuleNotFoundError: No module named 'omni'`` in this environment.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a Physics-OT checkpoint.")
parser.add_argument("--task", type=str, default="PhysicsOT-Knob-Allegro-Direct-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--num_episodes", type=int, default=100, help="Minimum number of completed episodes.")
parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path (else latest run).")
parser.add_argument(
    "--reward_mode",
    type=str,
    default=None,
    choices=["sparse", "state_tracking", "dtw_physics", "ot_state", "pointwise_physics", "physics_ot"],
    help=(
        "Which trained arm's checkpoint to load -- must match what --reward_mode "
        "was passed to train.py for that run, since it changes the log directory. "
        "Does NOT change how D_POT is computed here (see below)."
    ),
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default=None,
    help="Overrides the log directory name; defaults to 'physics_ot_knob_allegro[_<reward_mode>]'.",
)
parser.add_argument("--output", type=str, default=None, help="Optional JSON results path.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import physics_ot_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg


def main() -> None:
    import gymnasium as gym
    from rsl_rl.runners import OnPolicyRunner

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # The D_POT metric reported below is meant as a shared yardstick across all
    # six Experiment-0 arms (section 32) -- always compute it with the
    # physics_ot distance, regardless of which reward the checkpoint being
    # evaluated was trained with. reward_mode only affects _get_rewards(), not
    # observations/actions/dynamics, so this has no effect on the policy itself.
    env_cfg.reward_mode = "physics_ot"

    agent_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if args_cli.checkpoint:
        resume_path = Path(args_cli.checkpoint)
    else:
        experiment_name = args_cli.experiment_name
        if experiment_name is None:
            experiment_name = agent_cfg.experiment_name
            if args_cli.reward_mode is not None:
                experiment_name = f"{experiment_name}_{args_cli.reward_mode}"
        log_root = Path("logs", "rsl_rl", experiment_name).resolve()
        resume_path = Path(get_checkpoint_path(str(log_root), agent_cfg.load_run, agent_cfg.load_checkpoint))

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(resume_path))
    policy = runner.get_inference_policy(device=agent_cfg.device)

    unwrapped = env.unwrapped
    num_envs = unwrapped.num_envs
    step_dt = unwrapped.cfg.sim.dt * unwrapped.cfg.decimation
    success_this_episode = torch.zeros(num_envs, dtype=torch.bool, device=unwrapped.device)
    energy = torch.zeros(num_envs, device=unwrapped.device)
    episode_step = torch.zeros(num_envs, device=unwrapped.device)
    # -1 = "not yet succeeded this episode"; set once, on the first step
    # success is achieved, so completion time reflects time-to-success
    # rather than the (fixed, uninformative) episode length -- there is no
    # early termination on success, so every episode otherwise runs the full
    # horizon regardless of policy quality.
    first_success_step = torch.full((num_envs,), -1.0, device=unwrapped.device)
    completed_episodes = 0
    successful_episodes = 0
    completion_times = []
    energies = []
    final_pot_distances = []

    obs, _ = env.reset()
    with torch.inference_mode():
        while completed_episodes < args_cli.num_episodes:
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            episode_step += 1

            # Sustained (hold-duration) success, computed by the env itself --
            # see the note in knob_allegro_env.py on why this must come from
            # extras rather than an independent instantaneous recomputation.
            is_success = extras["task_success"]
            success_this_episode |= is_success
            newly_success = is_success & (first_success_step < 0)
            first_success_step[newly_success] = episode_step[newly_success]
            energy += actions.pow(2).sum(dim=-1) * step_dt

            done_idx = dones.nonzero(as_tuple=False).squeeze(-1)
            if done_idx.numel() > 0:
                completed_episodes += done_idx.numel()
                successful_episodes += int(success_this_episode[done_idx].sum().item())
                succeeded_idx = done_idx[success_this_episode[done_idx]]
                completion_times += (first_success_step[succeeded_idx] * step_dt).tolist()
                energies += energy[done_idx].tolist()
                final_pot_distances += extras["physics_ot_distance"][done_idx].tolist()
                success_this_episode[done_idx] = False
                first_success_step[done_idx] = -1.0
                energy[done_idx] = 0.0
                episode_step[done_idx] = 0.0

    results = {
        "task": args_cli.task,
        "trained_reward_mode": args_cli.reward_mode,
        "num_episodes": completed_episodes,
        "success_rate": successful_episodes / completed_episodes if completed_episodes else None,
        # Mean time-to-first-success, among episodes that succeeded at all
        # (None if none did). Not comparable to "episode length" -- there is
        # no early termination on success, so this is the only meaningful
        # completion-time signal.
        "mean_completion_time_s": sum(completion_times) / len(completion_times) if completion_times else None,
        "mean_actuation_energy": sum(energies) / len(energies) if energies else None,
        "mean_final_pot_distance": (
            sum(final_pot_distances) / len(final_pot_distances) if final_pot_distances else None
        ),
    }
    print(json.dumps(results, indent=2))
    if args_cli.output:
        Path(args_cli.output).write_text(json.dumps(results, indent=2))

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
