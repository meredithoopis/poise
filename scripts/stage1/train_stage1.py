"""Stage-1 training loop: knob_env + reference_generator + alignment
(token_builder + mech_cost) + two_critic_sac, wired together end to end --
no simulator, no video. Torch-only, runs on this dev machine (or anywhere
with torch/numpy/scipy/gymnasium installed).

Exposes :func:`train` so ``run_stage1_checks.py`` can call it directly and
inspect the returned per-step metric histories, rather than duplicating the
training loop.

Usage:
    python scripts/stage1/train_stage1.py --episodes 200 --num_envs 16 --cost_mode mechanics_ot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics_ot.dynamics.knob import KnobParams  # noqa: E402

from poise_ot.alignment import AlignmentConfig, RollingAlignment  # noqa: E402
from poise_ot.knob_env import KnobEnv, KnobEnvConfig  # noqa: E402
from poise_ot.mech_cost import MechCostScales, MechCostWeights  # noqa: E402
from poise_ot.reference_generator import scripted_reference  # noqa: E402
from poise_ot.token_builder import Token  # noqa: E402
from poise_ot.two_critic_sac import ReplayBuffer, SACConfig, TwoCriticSAC  # noqa: E402


def load_configs(knob_cfg_path: Path, poise_cfg_path: Path) -> tuple[dict, dict]:
    with open(knob_cfg_path) as f:
        knob_cfg = yaml.safe_load(f)
    with open(poise_cfg_path) as f:
        poise_cfg = yaml.safe_load(f)
    return knob_cfg, poise_cfg


def build_params(knob_cfg: dict) -> KnobParams:
    k = knob_cfg["knob"]
    return KnobParams(
        inertia=k["inertia"],
        damping=k["damping"],
        stiffness=k["stiffness"],
        theta0=k["theta0"],
        coulomb_torque=k["coulomb_torque"],
        v_eps=k["v_eps"],
    )


def train(
    num_envs: int = 16,
    episodes: int = 200,
    cost_mode: str = "mechanics_ot",
    device: str = "cpu",
    seed: int = 0,
    knob_cfg_path: Path | None = None,
    poise_cfg_path: Path | None = None,
    random_policy: bool = False,
    verbose: bool = True,
) -> dict:
    """Run Stage-1 two-critic SAC training on the synthetic knob task.

    Args:
        random_policy: if True, act with uniform-random actions and never
            call ``agent.update`` -- used by the "Learning signal sanity"
            check as the random-policy baseline to climb above.

    Returns a dict of per-update-step metric histories plus the trained
    agent/env/reference/alignment objects, so callers (run_stage1_checks.py)
    can inspect internals directly instead of re-deriving them.
    """
    torch.manual_seed(seed)

    knob_cfg_path = knob_cfg_path or (REPO_ROOT / "configs" / "knob.yaml")
    poise_cfg_path = poise_cfg_path or (REPO_ROOT / "configs" / "poise_ot.yaml")
    knob_cfg, poise_cfg = load_configs(knob_cfg_path, poise_cfg_path)

    params = build_params(knob_cfg)
    theta_goal = knob_cfg["knob"]["theta_goal"]
    success_threshold = knob_cfg["knob"]["success_threshold"]

    env_cfg = KnobEnvConfig(
        params=params,
        theta_goal=theta_goal,
        success_threshold=success_threshold,
        physics_dt=knob_cfg["simulation"]["physics_dt"],
        control_decimation=knob_cfg["simulation"]["control_decimation"],
        episode_seconds=knob_cfg["simulation"]["episode_seconds"],
        tau_max=1.0,
        success_bonus=poise_cfg["reward"]["success_bonus"],
        action_weight=poise_cfg["reward"]["action_weight"],
        unsafe_weight=poise_cfg["reward"]["unsafe_weight"],
    )
    env = KnobEnv(env_cfg, num_envs=num_envs, device=device)

    ref, ref_token_np = scripted_reference(
        params,
        theta_goal,
        num_tokens=knob_cfg["reference"]["num_tokens"],
        smoothing=knob_cfg["reference"]["smoothing"],
        impulse_window=poise_cfg["token"]["impulse_window"],
    )
    ref_time = torch.from_numpy(ref.time.astype(np.float32)).to(device)
    reference_token = Token(
        z=ref_token_np.z.to(device).float(),
        confidence=ref_token_np.confidence.to(device).float(),
        tau=ref_token_np.tau.to(device).float(),
    )

    mc = poise_cfg["mech_cost"]
    weights = MechCostWeights(
        progress=mc["progress"], velocity=mc["velocity"], impulse=mc["impulse"], power=mc["power"],
        dissipation=mc["dissipation"], time=mc["time"],
    )
    scales = MechCostScales(E0=mc["E0"], P0=mc["P0"], PD0=mc["PD0"])
    cost_terms = mc["cost_terms"]

    align_cfg = AlignmentConfig(
        window_h=poise_cfg["alignment"]["window_h"],
        epsilon=poise_cfg["alignment"]["epsilon"],
        num_iters=poise_cfg["alignment"]["num_iters"],
        impulse_window=poise_cfg["token"]["impulse_window"],
        cost_mode=cost_mode,
    )
    alignment = RollingAlignment(reference_token, ref_time, params, num_envs, device, align_cfg, weights, scales, cost_terms)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    h_dim = alignment.h_dim + obs_dim

    sac_cfg = poise_cfg["sac"]
    agent = TwoCriticSAC(
        SACConfig(
            obs_dim=obs_dim,
            h_dim=h_dim,
            action_dim=action_dim,
            hidden_dims=tuple(sac_cfg["hidden_dims"]),
            actor_lr=sac_cfg["actor_lr"],
            critic_lr=sac_cfg["critic_lr"],
            gamma=sac_cfg["gamma"],
            tau_polyak=sac_cfg["tau_polyak"],
            alpha=sac_cfg["alpha"],
            xi=sac_cfg["xi"],
            grad_clip_norm=sac_cfg.get("grad_clip_norm", 10.0),
            device=device,
        )
    )
    buffer = ReplayBuffer(sac_cfg["replay_capacity"], obs_dim, h_dim, action_dim, device=device)

    max_steps_per_episode = env._max_episode_steps
    total_steps = episodes * max_steps_per_episode

    history: dict[str, list] = {
        "loss_q_task": [], "loss_q_c": [], "loss_actor": [], "step": [],
        "episode_return_task": [], "episode_success": [], "episode_final_c_align": [],
    }

    obs, _ = env.reset()
    elapsed_time = torch.zeros(num_envs, device=device)
    episode_return = torch.zeros(num_envs, device=device)
    episode_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    c_align_prev = torch.zeros(num_envs, device=device)

    global_step = 0
    for step in range(total_steps):
        c_align, h = alignment.step_alignment(obs, elapsed_time)

        if random_policy or global_step < sac_cfg["warmup_steps"]:
            action = torch.rand(num_envs, action_dim, device=device) * 2 - 1
        else:
            action = agent.act(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated
        elapsed_time = elapsed_time + env.dt

        # Push this step's resulting token into the window BEFORE computing
        # next_c_align/next_h -- rho_{t,H}^R (Eq window_rollout) is defined
        # as ending at the current step, so h_{t+1} must see the token from
        # the state the just-taken action produced, not the stale window.
        alignment.push(info["progress"], info["theta_dot"], info["tau"], dt=env.dt)
        _, next_h = alignment.step_alignment(next_obs, elapsed_time)  # only next_h is needed by the replay buffer

        if not random_policy:
            buffer.add_batch(obs, h, action, reward, c_align, next_obs, next_h, done.float())

        episode_return += reward
        episode_success |= info["is_success"]
        c_align_prev = c_align

        if not random_policy and len(buffer) >= sac_cfg["batch_size"] and global_step >= sac_cfg["warmup_steps"]:
            for _ in range(sac_cfg["updates_per_step"]):
                batch = buffer.sample(sac_cfg["batch_size"])
                metrics = agent.update(batch)
            history["loss_q_task"].append(metrics["loss_q_task"])
            history["loss_q_c"].append(metrics["loss_q_c"])
            history["loss_actor"].append(metrics["loss_actor"])
            history["step"].append(global_step)

        done_idx = done.nonzero(as_tuple=False).squeeze(-1)
        if done_idx.numel() > 0:
            history["episode_return_task"] += episode_return[done_idx].tolist()
            history["episode_success"] += episode_success[done_idx].tolist()
            history["episode_final_c_align"] += c_align_prev[done_idx].tolist()
            alignment.reset_idx(done_idx)
            elapsed_time[done_idx] = 0.0
            episode_return[done_idx] = 0.0
            episode_success[done_idx] = False
            # env.reset mutates only the done_idx envs' internal state but
            # returns _get_obs() over ALL envs -- since step() always
            # advances every env's dynamics regardless of its done status,
            # the non-reset entries in this returned obs are still exactly
            # this step's next_obs, so reassigning next_obs here is a
            # correct merge, not a partial/stale overwrite.
            next_obs, _ = env.reset(env_ids=done_idx)

        obs = next_obs
        global_step += 1

        if verbose and (step + 1) % max(1, total_steps // 10) == 0:
            recent_success = np.mean(history["episode_success"][-50:]) if history["episode_success"] else float("nan")
            print(f"[train_stage1] step {step + 1}/{total_steps} recent_success_rate={recent_success:.3f}")

    return {
        "history": history,
        "agent": agent,
        "env": env,
        "reference_token": reference_token,
        "alignment": alignment,
        "params": params,
        "theta_goal": theta_goal,
        "success_threshold": success_threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--cost_mode", type=str, default="mechanics_ot", choices=["mechanics_ot", "euclidean_ot", "pointwise_mechanics"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None, help="Optional json path for the episode history.")
    args = parser.parse_args()

    result = train(num_envs=args.num_envs, episodes=args.episodes, cost_mode=args.cost_mode, device=args.device, seed=args.seed)
    history = result["history"]
    print(f"[train_stage1] done -- {len(history['episode_success'])} episodes, "
          f"final success rate (last 50): {np.mean(history['episode_success'][-50:]):.3f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(history, indent=2))
        print(f"[train_stage1] wrote {args.output}")


if __name__ == "__main__":
    main()
