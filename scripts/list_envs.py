# Adapted from IsaacLab's scripts/environments/list_envs.py, filtered for
# this extension's own tasks instead of the built-in `Isaac-*` ones.

"""Print all Physics-OT environments registered by physics_ot_tasks."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="List Physics-OT environments.")
parser.add_argument("--keyword", type=str, default=None, help="Keyword to filter environments.")
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
from prettytable import PrettyTable

import physics_ot_tasks  # noqa: F401


def main() -> None:
    task_specs = [
        spec
        for spec in gym.registry.values()
        if "PhysicsOT" in spec.id and (args_cli.keyword is None or args_cli.keyword in spec.id)
    ]

    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Physics-OT Environments"
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"
    for index, spec in enumerate(task_specs):
        table.add_row([index + 1, spec.id, spec.entry_point, spec.kwargs["env_cfg_entry_point"]])

    print(table)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
