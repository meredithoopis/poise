# `omni.kit.app` (and everything IsaacLab transitively imports through
# `isaaclab.envs`) is a Kit extension: it only becomes importable *after*
# `AppLauncher` has been constructed, not just imported. So AppLauncher must
# be constructed first, exactly like IsaacLab's own scripts/environments/
# list_envs.py, before importing physics_ot_tasks/gymnasium/etc.

"""Run a Physics-OT environment with a zero-action agent.

If this script appears to hang, find its PID (``ps aux | grep zero_agent.py``)
and run ``kill -USR1 <PID>`` from another terminal -- no sudo/ptrace needed,
since sending a signal to your own process never requires elevated
permissions. That dumps every thread's live Python stack to stderr right
here, showing exactly which call it's blocked in.
"""

import argparse
import faulthandler
import signal
import sys
import traceback

from isaaclab.app import AppLauncher

faulthandler.register(signal.SIGUSR1, all_threads=True)


def _checkpoint(msg: str) -> None:
    """Print with an explicit flush so nothing is lost to buffering or a hard process exit."""
    print(msg, flush=True)

parser = argparse.ArgumentParser(description="Zero agent for Physics-OT environments.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="PhysicsOT-Knob-Allegro-Direct-v0", help="Name of the task.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import physics_ot_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

MAX_STEPS = 200


def main() -> None:
    torch.manual_seed(42)
    _checkpoint("[CHECKPOINT] before parse_env_cfg")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    _checkpoint("[CHECKPOINT] after parse_env_cfg, before gym.make")

    env = gym.make(args_cli.task, cfg=env_cfg)
    _checkpoint("[CHECKPOINT] after gym.make")
    print(f"[INFO]: Gym observation space: {env.observation_space}", flush=True)
    print(f"[INFO]: Gym action space: {env.action_space}", flush=True)
    env.reset()
    _checkpoint("[CHECKPOINT] after env.reset")

    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    for i in range(MAX_STEPS):
        with torch.inference_mode():
            env.step(actions)
        if i == 0:
            _checkpoint("[CHECKPOINT] after first env.step")
    env.close()
    _checkpoint("[CHECKPOINT] main() completed normally")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Force the traceback out even if something downstream (Kit's own
        # exception handling, a hard process exit) would otherwise swallow it.
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()
