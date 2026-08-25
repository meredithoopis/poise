"""Evaluate a trained Physics-OT checkpoint: success rate, D_POT, completion
time, and actuation energy (section 32), for the Experiment-0 ablation table.

Built on the same non-deprecated building blocks as ``zero_agent.py``
(``resolve_task_config`` / ``launch_simulation``) plus RSL-RL's runner/env
wrapper, rather than the deprecated ``scripts/reinforcement_learning/rsl_rl/
play.py`` entry point.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

import physics_ot_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser(description="Evaluate a Physics-OT checkpoint.")
parser.add_argument("--task", type=str, default="PhysicsOT-Knob-Allegro-Direct-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--num_episodes", type=int, default=100, help="Minimum number of completed episodes.")
parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path (else latest run).")
parser.add_argument("--experiment_name", type=str, default="physics_ot_knob_allegro")
parser.add_argument("--output", type=str, default=None, help="Optional JSON results path.")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def main() -> None:
    import gymnasium as gym
    from rsl_rl.runners import OnPolicyRunner

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        if args_cli.checkpoint:
            resume_path = Path(args_cli.checkpoint)
        else:
            log_root = Path("logs", "rsl_rl", args_cli.experiment_name).resolve()
            resume_path = Path(get_checkpoint_path(str(log_root), agent_cfg.load_run, agent_cfg.load_checkpoint))

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, __import__("importlib.metadata", fromlist=["version"]).version("rsl-rl-lib"))
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(resume_path))
        policy = runner.get_inference_policy(device=agent_cfg.device)

        unwrapped = env.unwrapped
        num_envs = unwrapped.num_envs
        step_dt = unwrapped.cfg.sim.dt * unwrapped.cfg.decimation
        success_this_episode = torch.zeros(num_envs, dtype=torch.bool, device=unwrapped.device)
        energy = torch.zeros(num_envs, device=unwrapped.device)
        episode_step = torch.zeros(num_envs, device=unwrapped.device)
        completed_episodes = 0
        successful_episodes = 0
        completion_times = []
        final_pot_distances = []

        obs, _ = env.reset()
        with torch.inference_mode():
            while completed_episodes < args_cli.num_episodes:
                actions = policy(obs)
                obs, _, dones, extras = env.step(actions)

                theta = unwrapped._knob.data.joint_pos[:, unwrapped._knob_joint_idx]
                is_success = (theta - unwrapped._theta_goal).abs() < unwrapped.cfg.success_threshold
                success_this_episode |= is_success
                energy += actions.pow(2).sum(dim=-1) * step_dt
                episode_step += 1

                done_idx = dones.nonzero(as_tuple=False).squeeze(-1)
                if done_idx.numel() > 0:
                    completed_episodes += done_idx.numel()
                    successful_episodes += int(success_this_episode[done_idx].sum().item())
                    completion_times += (episode_step[done_idx] * step_dt).tolist()
                    final_pot_distances += unwrapped._prev_distance[done_idx].tolist()
                    success_this_episode[done_idx] = False
                    energy[done_idx] = 0.0
                    episode_step[done_idx] = 0.0

        results = {
            "task": args_cli.task,
            "reward_mode": unwrapped.cfg.reward_mode,
            "num_episodes": completed_episodes,
            "success_rate": successful_episodes / completed_episodes if completed_episodes else None,
            "mean_completion_time_s": (
                sum(completion_times) / len(completion_times) if completion_times else None
            ),
            "mean_actuation_energy": float(energy.mean().item()),
            "mean_final_pot_distance": (
                sum(final_pot_distances) / len(final_pot_distances) if final_pot_distances else None
            ),
        }
        print(json.dumps(results, indent=2))
        if args_cli.output:
            Path(args_cli.output).write_text(json.dumps(results, indent=2))

        env.close()


if __name__ == "__main__":
    main()
