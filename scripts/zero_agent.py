# Adapted from IsaacLab's scripts/environments/zero_agent.py, importing
# physics_ot_tasks instead of isaaclab_tasks so it exercises our own
# registered environment. Useful as a first smoke test that the env, the
# knob asset, and the reference file all load correctly before training.

"""Run a Physics-OT environment with a zero-action agent."""

import argparse
import sys

import torch

import physics_ot_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser(description="Zero agent for Physics-OT environments.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="PhysicsOT-Knob-Allegro-Direct-v0", help="Name of the task.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

MAX_STEPS = 200


def main() -> None:
    import gymnasium as gym

    torch.manual_seed(42)
    env_cfg, _ = resolve_task_config(args_cli.task, "")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if args_cli.disable_fabric:
            env_cfg.sim.use_fabric = False

        env = gym.make(args_cli.task, cfg=env_cfg)
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        env.reset()

        sim = env.unwrapped.sim
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        step = 0
        while True:
            if sim.visualizers:
                if not any(v.is_running() and not v.is_closed for v in sim.visualizers):
                    break
            elif step >= MAX_STEPS:
                break
            with torch.inference_mode():
                env.step(actions)
            step += 1
        env.close()


if __name__ == "__main__":
    main()
