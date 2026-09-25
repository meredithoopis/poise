"""Stage-3 training loop: the real IsaacLab Allegro-vs-knob env
(``PoiseOT-Knob-Allegro-Direct-v0``) + the *unchanged* Stage-1 alignment/
token/cost/two-critic-SAC machinery (``poise_ot.alignment``,
``poise_ot.mech_cost``, ``poise_ot.two_critic_sac``). This is deliberately
almost line-for-line ``scripts/stage1/train_stage1.py`` with the synthetic
``KnobEnv`` swapped for the real simulator -- step 11's "Regression check
#1" gate exists specifically to prove that swap alone doesn't change the
algorithm's behavior; if it does, the bug is in the simulator/env wiring,
not in the (untouched) alignment/cost/SAC code.

Same "expose pre-reset state via extras, not by re-deriving after reset"
contract as Stage 1's ``KnobEnv``: ``KnobAllegroPoiseEnv._get_rewards()``
populates ``self.extras["theta"/"theta_dot"/"tau"/"progress"/"is_success"]``
each step, so this loop reads them the exact same way
``train_stage1.py`` reads ``info[...]``.

Unlike ``KnobEnv``, IsaacLab's ``DirectRLEnv`` auto-resets internally
per-env when ``terminated | truncated`` fires -- there is no
``env.reset(env_ids=done_idx)`` to call manually; the ``obs`` returned by
``env.step()`` for a just-finished env is already its post-reset
observation, so ``alignment.reset_idx(done_idx)`` is enough to keep the
alignment-window bookkeeping consistent.

**Untested against real Isaac Sim** -- see the env's own module docstring.

Usage (small-scale regression check, step 11):
    $ISAACLAB scripts/stage3/train_stage3.py \\
        --task PoiseOT-Knob-Allegro-Direct-v0 --num_envs 16 --episodes 20 \\
        --cost_mode mechanics_ot --headless
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="PoiseOT-Knob-Allegro-Direct-v0")
parser.add_argument("--num_envs", type=int, default=16, help="Step 11: a handful of envs, not 2048+ yet.")
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--cost_mode", type=str, default="mechanics_ot", choices=["mechanics_ot", "euclidean_ot", "pointwise_mechanics"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--random_policy", action="store_true", help="Random-action baseline, no SAC updates (mirrors train_stage1.py's check-4 baseline).")
parser.add_argument("--poise_cfg_path", type=Path, default=REPO_ROOT / "configs" / "poise_ot.yaml")
parser.add_argument("--knob_cfg_path", type=Path, default=REPO_ROOT / "configs" / "knob.yaml")
parser.add_argument("--output", type=str, default=None, help="Optional json path for the episode history.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
import gymnasium as gym  # noqa: E402

import physics_ot_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from physics_ot.dynamics.knob import KnobParams  # noqa: E402

from poise_ot.alignment import AlignmentConfig, RollingAlignment  # noqa: E402
from poise_ot.mech_cost import MechCostScales, MechCostWeights  # noqa: E402
from poise_ot.reference_generator import scripted_reference  # noqa: E402
from poise_ot.token_builder import Token  # noqa: E402
from poise_ot.two_critic_sac import ReplayBuffer, SACConfig, TwoCriticSAC  # noqa: E402


def build_params(knob_cfg: dict) -> KnobParams:
    k = knob_cfg["knob"]
    return KnobParams(
        inertia=k["inertia"], damping=k["damping"], stiffness=k["stiffness"],
        theta0=k["theta0"], coulomb_torque=k["coulomb_torque"], v_eps=k["v_eps"],
    )


def main() -> None:
    with open(args_cli.knob_cfg_path) as f:
        knob_cfg = yaml.safe_load(f)
    with open(args_cli.poise_cfg_path) as f:
        poise_cfg = yaml.safe_load(f)

    device = args_cli.device
    torch.manual_seed(args_cli.seed)

    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    num_envs = unwrapped.num_envs

    params = build_params(knob_cfg)
    theta_goal = knob_cfg["knob"]["theta_goal"]
    # success_threshold is read by the env cfg itself (KnobAllegroPoiseEnvCfg
    # loads configs/knob.yaml directly) -- not needed again here.

    # The *same synthetic* Stage-1 reference the env itself builds internally
    # (KnobAllegroPoiseEnv.__init__) -- rebuilt here, identically, so this
    # loop's alignment machinery and the env's own z_ref observation channel
    # can never silently diverge (step 11's regression check depends on this).
    ref, ref_token_np = scripted_reference(
        params, theta_goal,
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
        cost_mode=args_cli.cost_mode,
    )
    alignment = RollingAlignment(reference_token, ref_time, params, num_envs, device, align_cfg, weights, scales, cost_terms)

    obs_dim = env.observation_space["policy"].shape[-1]
    action_dim = env.action_space.shape[-1]
    h_dim = alignment.h_dim + obs_dim

    sac_cfg = poise_cfg["sac"]
    agent = TwoCriticSAC(
        SACConfig(
            obs_dim=obs_dim, h_dim=h_dim, action_dim=action_dim,
            hidden_dims=tuple(sac_cfg["hidden_dims"]),
            actor_lr=sac_cfg["actor_lr"], critic_lr=sac_cfg["critic_lr"],
            gamma=sac_cfg["gamma"], tau_polyak=sac_cfg["tau_polyak"],
            alpha=sac_cfg["alpha"], xi=sac_cfg["xi"],
            grad_clip_norm=sac_cfg.get("grad_clip_norm", 10.0),
            device=device,
        )
    )
    buffer = ReplayBuffer(sac_cfg["replay_capacity"], obs_dim, h_dim, action_dim, device=device)

    max_steps_per_episode = int(unwrapped.max_episode_length)
    total_steps = args_cli.episodes * max_steps_per_episode

    history: dict[str, list] = {
        "loss_q_task": [], "loss_q_c": [], "loss_actor": [], "step": [],
        "episode_return_task": [], "episode_success": [],
    }

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    elapsed_time = torch.zeros(num_envs, device=device)
    episode_return = torch.zeros(num_envs, device=device)
    episode_success = torch.zeros(num_envs, dtype=torch.bool, device=device)

    # Computed directly from cfg, not assumed as an env attribute (matches
    # knob_allegro_env.py's own self._dt = cfg.sim.dt * cfg.decimation).
    step_dt = env_cfg.sim.dt * env_cfg.decimation

    # No blanket torch.inference_mode()/no_grad() around this loop: tensors
    # created under inference_mode become permanently "inference tensors"
    # that autograd refuses to use later even inside a nested
    # inference_mode(False) block -- since obs/action/reward here get stored
    # in the replay buffer and later fed through agent.update()'s backward
    # pass, that would break training. agent.act() already wraps its own
    # forward pass in torch.no_grad() internally, same as train_stage1.py's
    # (already-validated) pattern.
    global_step = 0
    for step in range(total_steps):
        c_align, h = alignment.step_alignment(obs, elapsed_time)

        if args_cli.random_policy or global_step < sac_cfg["warmup_steps"]:
            action = torch.rand(num_envs, action_dim, device=device) * 2 - 1
        else:
            action = agent.act(obs)

        next_obs_dict, reward, terminated, truncated, extras = env.step(action)
        next_obs = next_obs_dict["policy"]
        done = terminated | truncated
        elapsed_time = elapsed_time + step_dt

        # Push this step's resulting token into the window BEFORE
        # computing next_h -- same ordering fix as train_stage1.py (Eq
        # window_rollout's window must end at the current step).
        alignment.push(extras["progress"], extras["theta_dot"], extras["tau"], dt=step_dt)
        _, next_h = alignment.step_alignment(next_obs, elapsed_time)

        if not args_cli.random_policy:
            buffer.add_batch(obs, h, action, reward, c_align, next_obs, next_h, done.float())

        episode_return += reward
        episode_success |= extras["is_success"]

        if not args_cli.random_policy and len(buffer) >= sac_cfg["batch_size"] and global_step >= sac_cfg["warmup_steps"]:
            for _ in range(sac_cfg["updates_per_step"]):
                batch = buffer.sample(sac_cfg["batch_size"])
                metrics = agent.update(batch)
            history["loss_q_task"].append(metrics["loss_q_task"])
            history["loss_q_c"].append(metrics["loss_q_c"])
            history["loss_actor"].append(metrics["loss_actor"])
            history["step"].append(global_step)

        done_idx = done.nonzero(as_tuple=False).squeeze(-1)
        if done_idx.numel() > 0:
            # DirectRLEnv already reset these envs internally (unlike
            # KnobEnv, no explicit env.reset(env_ids=...) call here) --
            # next_obs above already reflects their post-reset state.
            history["episode_return_task"] += episode_return[done_idx].tolist()
            history["episode_success"] += episode_success[done_idx].tolist()
            alignment.reset_idx(done_idx)
            elapsed_time[done_idx] = 0.0
            episode_return[done_idx] = 0.0
            episode_success[done_idx] = False

        obs = next_obs
        global_step += 1

        if (step + 1) % max(1, total_steps // 10) == 0:
            recent_success = np.mean(history["episode_success"][-50:]) if history["episode_success"] else float("nan")
            print(f"[train_stage3] step {step + 1}/{total_steps} recent_success_rate={recent_success:.3f}", flush=True)

    print(f"[train_stage3] done -- {len(history['episode_success'])} episodes, "
          f"final success rate (last 50): {np.mean(history['episode_success'][-50:]):.3f}")

    if args_cli.output:
        Path(args_cli.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args_cli.output).write_text(json.dumps(history, indent=2))
        print(f"[train_stage3] wrote {args_cli.output}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
