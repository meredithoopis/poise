"""Stage-3 env: Allegro hand vs. a resistive knob, exposing the observation
(Eq observation) and reward (Eq task_reward) contract POISE-OT's two-critic
SAC needs -- NOT a copy of knob_allegro_env.py with edits. That env bakes a
*single* blended reward (task + Physics-OT shaping + action penalty) into
one scalar via ``compose_reward``, which is exactly what two-critic SAC
must NOT do: ``Q_task`` and ``Q_C`` need ``r_task`` and the raw mechanical-
alignment cost ``c_t^align`` as two separate signals (see
``poise_ot.two_critic_sac``'s module docstring). So this class:

- returns *only* ``r_task`` (Eq task_reward) as the step reward, matching
  ``poise_ot.knob_env.KnobEnv``'s contract from Stage 1 exactly;
- does NOT compute any OT distance/shaping reward itself -- token building
  (``token_builder.build_token``), the rolling alignment window, and
  ``mech_cost``/``two_critic_sac`` all live in ``scripts/stage3/
  train_stage3.py``, completely unchanged from Stage 1 (step 11's explicit
  instruction: "Plug in two_critic_sac.py ... and mech_cost.py unchanged");
- exposes raw per-step state (theta, theta_dot, tau via inverse dynamics,
  progress, success) through ``self.extras``, the same "expose pre-reset
  state via extras/info, not by re-deriving after reset" pattern already
  used by ``knob_allegro_env.py`` and ``poise_ot.knob_env.KnobEnv``, so
  ``train_stage3.py`` can build tokens/alignment externally exactly like
  ``train_stage1.py`` does.

**Untested against real Isaac Sim** -- written by mirroring
``knob_allegro_env.py``'s already-validated physics/action wiring as
closely as possible (same asset, same passive-torque application, same
inverse-dynamics tau_r computation), but this file itself has not been run.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import saturate, unscale_transform

from physics_ot.dynamics.knob import KnobDynamics, KnobParams
from physics_ot.rewards import task_reward

from poise_ot.reference_generator import scripted_reference

from .knob_allegro_poise_env_cfg import KnobAllegroPoiseEnvCfg


class KnobAllegroPoiseEnv(DirectRLEnv):
    cfg: KnobAllegroPoiseEnvCfg

    def __init__(self, cfg: KnobAllegroPoiseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._dt = self.cfg.sim.dt * self.cfg.decimation

        self._actuated_dof_idx, _ = self._robot.find_joints(self.cfg.actuated_joint_names, preserve_order=True)
        self._fingertip_body_idx, _ = self._robot.find_bodies(self.cfg.fingertip_body_names, preserve_order=True)
        self._knob_joint_ids, _ = self._knob.find_joints(self.cfg.knob_joint_name)
        self._knob_joint_idx = self._knob_joint_ids[0]

        joint_pos_limits = self._robot.data.joint_pos_limits.to(self.device)
        self._hand_dof_lower = joint_pos_limits[..., 0]
        self._hand_dof_upper = joint_pos_limits[..., 1]

        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_targets = torch.zeros(self.num_envs, self._robot.num_joints, device=self.device)
        self._cur_targets = torch.zeros(self.num_envs, self._robot.num_joints, device=self.device)

        self._knob_params = KnobParams(
            inertia=self.cfg.knob_inertia,
            damping=self.cfg.knob_damping,
            stiffness=self.cfg.knob_stiffness,
            theta0=self.cfg.knob_theta0,
            coulomb_torque=self.cfg.knob_coulomb_torque,
            v_eps=self.cfg.knob_v_eps,
        )
        self._knob_dynamics = KnobDynamics(self._knob_params)
        self._theta_goal = torch.full((self.num_envs,), self.cfg.knob_theta_goal, device=self.device)

        # The *same synthetic* Stage-1 reference (poise_ot.reference_generator.
        # scripted_reference, same KnobParams) -- built here, not loaded from
        # a file, so z_ref can never drift from what train_stage3.py's own
        # alignment loop uses (step 11's regression check needs these
        # identical). z_ref is the 5-dim POISE token (Eq 30), not the old
        # 3-dim (s, v, tau) used by knob_allegro_env.py.
        _, ref_token = scripted_reference(
            self._knob_params,
            self.cfg.knob_theta_goal,
            num_tokens=self.cfg.reference_num_tokens,
            smoothing=self.cfg.reference_smoothing,
            impulse_window=self.cfg.impulse_window,
        )
        self._reference_z = ref_token.z.to(self.device).float()  # [num_tokens, 5]

        # EMA-smoothed acceleration estimate, exactly like knob_allegro_env.py's
        # _smoothed_theta_ddot -- a raw single-step finite difference of
        # theta_dot is noisy during contact transients, which the inverse-
        # dynamics tau (fed to token_builder's J_tau/P) is far more sensitive
        # to than a scalar reward would be.
        self._prev_theta_dot = torch.zeros(self.num_envs, device=self.device)
        self._smoothed_theta_ddot = torch.zeros(self.num_envs, device=self.device)

        # Success requires *holding* the goal, not just touching it once --
        # same rationale as knob_allegro_env.py (a low-inertia/low-damping
        # knob can be brushed through the goal incidentally by almost any
        # policy otherwise).
        self._success_hold_steps_required = max(1, round(self.cfg.success_hold_time / self._dt))
        self._success_hold_counter = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        self._knob = Articulation(self.cfg.knob_cfg)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["knob"] = self._knob
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        self._cur_targets[:, self._actuated_dof_idx] = unscale_transform(
            self._actions,
            self._hand_dof_lower[:, self._actuated_dof_idx],
            self._hand_dof_upper[:, self._actuated_dof_idx],
        )
        self._cur_targets[:, self._actuated_dof_idx] = (
            self.cfg.act_moving_average * self._cur_targets[:, self._actuated_dof_idx]
            + (1.0 - self.cfg.act_moving_average) * self._prev_targets[:, self._actuated_dof_idx]
        )
        self._cur_targets[:, self._actuated_dof_idx] = saturate(
            self._cur_targets[:, self._actuated_dof_idx],
            self._hand_dof_lower[:, self._actuated_dof_idx],
            self._hand_dof_upper[:, self._actuated_dof_idx],
        )
        self._prev_targets[:, self._actuated_dof_idx] = self._cur_targets[:, self._actuated_dof_idx]
        self._robot.set_joint_position_target(
            self._cur_targets[:, self._actuated_dof_idx], joint_ids=self._actuated_dof_idx
        )

        theta = self._knob.data.joint_pos[:, self._knob_joint_idx]
        theta_dot = self._knob.data.joint_vel[:, self._knob_joint_idx]
        passive_torque = self._knob_dynamics.passive_torque(theta, theta_dot)
        self._knob.set_joint_effort_target(passive_torque.unsqueeze(-1), joint_ids=self._knob_joint_ids)

    def _nearest_reference_token(self) -> torch.Tensor:
        num_tokens = self._reference_z.shape[0]
        time_frac = (self.episode_length_buf.float() / self.max_episode_length).clamp(0.0, 1.0)
        ref_idx = (time_frac * (num_tokens - 1)).round().long()
        return self._reference_z[ref_idx]  # [num_envs, 5]

    def _get_observations(self) -> dict:
        theta = self._knob.data.joint_pos[:, self._knob_joint_idx]
        theta_dot = self._knob.data.joint_vel[:, self._knob_joint_idx]
        z_ref = self._nearest_reference_token()

        # TODO(remote): wire real per-fingertip contact sensing (ContactSensorCfg),
        # same not-yet-wired placeholder as knob_allegro_env.py.
        contact_flags = torch.zeros(self.num_envs, len(self._fingertip_body_idx), device=self.device)

        obs = torch.cat(
            [
                self._robot.data.joint_pos[:, self._actuated_dof_idx],
                self._robot.data.joint_vel[:, self._actuated_dof_idx],
                theta.unsqueeze(-1),
                theta_dot.unsqueeze(-1),
                z_ref,
                contact_flags,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        theta = self._knob.data.joint_pos[:, self._knob_joint_idx]
        theta_dot = self._knob.data.joint_vel[:, self._knob_joint_idx]
        raw_theta_ddot = (theta_dot - self._prev_theta_dot) / self._dt
        alpha = self.cfg.wrench_accel_smoothing
        self._smoothed_theta_ddot = alpha * raw_theta_ddot + (1.0 - alpha) * self._smoothed_theta_ddot
        # Net externally-applied (robot/contact) torque, recovered via
        # inverse dynamics from the observed motion -- exactly
        # knob_allegro_env.py's tau_r computation, reused here as the "tau"
        # fed to token_builder.build_token() by train_stage3.py (there is no
        # single scripted robot-torque scalar in a contact-driven sim, unlike
        # Stage 1's synthetic env, so this is the physically meaningful
        # analogue: how much torque *must* the robot/contact be applying to
        # explain this motion, beyond the knob's own modeled passive
        # dynamics).
        tau = self._knob_dynamics.required_generalized_force(theta, theta_dot, self._smoothed_theta_ddot)
        progress = (theta - self.cfg.knob_theta0) / (self._theta_goal - self.cfg.knob_theta0)

        r_task, at_goal_now = task_reward(theta, self._theta_goal, self.cfg.success_threshold)
        self._success_hold_counter = torch.where(
            at_goal_now, self._success_hold_counter + 1, torch.zeros_like(self._success_hold_counter)
        )
        success = self._success_hold_counter >= self._success_hold_steps_required
        action_penalty = self._actions.pow(2).sum(dim=-1)
        # r_unsafe is 0 -- no unsafe-state model exists for this env yet
        # (same documented assumption as poise_ot.knob_env.KnobEnv), not
        # silently dropped.
        r_task = r_task + self.cfg.success_bonus * success.float() - self.cfg.action_weight * action_penalty

        # Pre-reset state, surviving _reset_idx() below via extras -- same
        # pattern as knob_allegro_env.py's physics_ot_distance/task_success
        # and poise_ot.knob_env.KnobEnv's info dict.
        self.extras["theta"] = theta.clone()
        self.extras["theta_dot"] = theta_dot.clone()
        self.extras["tau"] = tau.clone()
        self.extras["progress"] = progress.clone()
        self.extras["is_success"] = success.clone()
        self.extras["r_task"] = r_task.clone()

        self._prev_theta_dot = theta_dot
        return r_task

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(time_out)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        robot_joint_pos = self._robot.data.default_joint_pos[env_ids]
        robot_joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(robot_joint_pos, robot_joint_vel, None, env_ids)
        self._prev_targets[env_ids] = robot_joint_pos
        self._cur_targets[env_ids] = robot_joint_pos

        knob_joint_pos = self._knob.data.default_joint_pos[env_ids]
        knob_joint_vel = self._knob.data.default_joint_vel[env_ids]
        self._knob.write_joint_state_to_sim(knob_joint_pos, knob_joint_vel, None, env_ids)

        self._prev_theta_dot[env_ids] = 0.0
        self._smoothed_theta_ddot[env_ids] = 0.0
        self._success_hold_counter[env_ids] = 0.0
