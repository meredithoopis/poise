from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import saturate, scale

from physics_ot.dynamics.knob import KnobDynamics, KnobParams
from physics_ot.ot.cost import CostWeights
from physics_ot.references.build_reference import load_reference
from physics_ot.rewards import RewardWeights, compose_reward, task_reward

from .knob_allegro_env_cfg import KnobAllegroEnvCfg
from .rewards import compute_distance


class KnobAllegroEnv(DirectRLEnv):
    """Allegro hand vs. a resistive knob, driven by a Physics-OT reference.

    See ``physics_ot`` for the shared inverse-dynamics/OT math; this class is
    the IsaacLab-specific glue: it applies the knob's own passive dynamics
    each physics step (section 5), tracks the robot's rollout history, and
    composes task + Physics-OT shaping + action-penalty rewards (section 13).
    """

    cfg: KnobAllegroEnvCfg

    def __init__(self, cfg: KnobAllegroEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._dt = self.cfg.sim.dt * self.cfg.decimation

        self._actuated_dof_idx, _ = self._robot.find_joints(self.cfg.actuated_joint_names)
        self._fingertip_body_idx, _ = self._robot.find_bodies(self.cfg.fingertip_body_names)
        self._knob_joint_ids, _ = self._knob.find_joints(self.cfg.knob_joint_name)
        self._knob_joint_idx = self._knob_joint_ids[0]

        joint_pos_limits = self._robot.data.joint_limits.torch.to(self.device)
        self._hand_dof_lower = joint_pos_limits[..., 0]
        self._hand_dof_upper = joint_pos_limits[..., 1]

        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_targets = torch.zeros(self.num_envs, self._robot.num_joints, device=self.device)
        self._cur_targets = torch.zeros(self.num_envs, self._robot.num_joints, device=self.device)

        knob_params = KnobParams(
            inertia=self.cfg.knob_inertia,
            damping=self.cfg.knob_damping,
            stiffness=self.cfg.knob_stiffness,
            theta0=self.cfg.knob_theta0,
            coulomb_torque=self.cfg.knob_coulomb_torque,
            v_eps=self.cfg.knob_v_eps,
        )
        self._knob_dynamics = KnobDynamics(knob_params)
        self._theta_goal = torch.full((self.num_envs,), self.cfg.knob_theta_goal, device=self.device)

        reference = load_reference(self.cfg.reference_path)
        self._reference_progress = torch.tensor(reference.progress, device=self.device, dtype=torch.float32)
        self._reference_velocity = torch.tensor(reference.velocity, device=self.device, dtype=torch.float32)
        self._reference_wrench = torch.tensor(reference.wrench, device=self.device, dtype=torch.float32)

        self._cost_weights = CostWeights(
            progress=self.cfg.cost_lambda_progress,
            velocity=self.cfg.cost_lambda_velocity,
            wrench=self.cfg.cost_lambda_wrench,
            time=self.cfg.cost_lambda_time,
        )
        self._reward_weights = RewardWeights(
            task=self.cfg.reward_task,
            physics_ot=self.cfg.reward_physics_ot,
            action=self.cfg.reward_action,
            success_bonus=self.cfg.reward_success_bonus,
        )

        # Full-episode rollout history of the knob's physical state (s, theta_dot, tau_R),
        # resampled to a fixed-size window each step for the shaping distance (sections 7-10).
        max_steps = self.max_episode_length
        self._history_progress = torch.zeros(self.num_envs, max_steps, device=self.device)
        self._history_velocity = torch.zeros(self.num_envs, max_steps, device=self.device)
        self._history_wrench = torch.zeros(self.num_envs, max_steps, device=self.device)
        self._prev_theta_dot = torch.zeros(self.num_envs, device=self.device)
        self._prev_distance = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        self._knob = Articulation(self.cfg.knob_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
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
        self._cur_targets[:, self._actuated_dof_idx] = scale(
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

        # The knob's own spring/damper/Coulomb-friction torque (section 5) is
        # applied explicitly every physics step; any additional net torque
        # comes from Allegro/knob contact, resolved by PhysX.
        theta = self._knob.data.joint_pos[:, self._knob_joint_idx]
        theta_dot = self._knob.data.joint_vel[:, self._knob_joint_idx]
        passive_torque = self._knob_dynamics.passive_torque(theta, theta_dot)
        self._knob.set_joint_effort_target(passive_torque.unsqueeze(-1), joint_ids=self._knob_joint_ids)

    def _record_history_and_get_window(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        theta = self._knob.data.joint_pos[:, self._knob_joint_idx]
        theta_dot = self._knob.data.joint_vel[:, self._knob_joint_idx]
        theta_ddot = (theta_dot - self._prev_theta_dot) / self._dt
        # tau_R: the same inverse-dynamics operator applied to the robot's own
        # measured trajectory (section 7-8) -- not a separate force estimate.
        tau_r = self._knob_dynamics.required_generalized_force(theta, theta_dot, theta_ddot)
        progress = (theta - self.cfg.knob_theta0) / (self._theta_goal - self.cfg.knob_theta0)

        step = self.episode_length_buf.clamp(max=self._history_progress.shape[1] - 1)
        env_idx = torch.arange(self.num_envs, device=self.device)
        self._history_progress[env_idx, step] = progress
        self._history_velocity[env_idx, step] = theta_dot
        self._history_wrench[env_idx, step] = tau_r

        history_len = (step + 1).clamp(min=1)
        window = self.cfg.robot_window
        # Evenly resample the valid history [0, history_len) onto `window`
        # tokens per env, so the shaping distance always sees a fixed
        # T_R-sized rollout regardless of episode progress.
        frac = torch.linspace(0.0, 1.0, window, device=self.device).unsqueeze(0)
        idx = (frac * (history_len.unsqueeze(-1) - 1).clamp(min=0)).round().long()

        window_progress = torch.gather(self._history_progress, 1, idx)
        window_velocity = torch.gather(self._history_velocity, 1, idx)
        window_wrench = torch.gather(self._history_wrench, 1, idx)

        self._prev_theta_dot = theta_dot
        return window_progress, window_velocity, window_wrench

    def _get_observations(self) -> dict:
        theta = self._knob.data.joint_pos[:, self._knob_joint_idx]
        theta_dot = self._knob.data.joint_vel[:, self._knob_joint_idx]
        progress = (theta - self.cfg.knob_theta0) / (self._theta_goal - self.cfg.knob_theta0)
        s_goal = torch.ones_like(progress)

        # Closest reference token by elapsed-episode-time fraction (section 12).
        num_tokens = self._reference_progress.shape[0]
        time_frac = (self.episode_length_buf.float() / self.max_episode_length).clamp(0.0, 1.0)
        ref_idx = (time_frac * (num_tokens - 1)).round().long()
        z_ref = torch.stack(
            [self._reference_progress[ref_idx], self._reference_velocity[ref_idx], self._reference_wrench[ref_idx]],
            dim=-1,
        )

        # TODO(remote): wire real per-fingertip contact sensing (ContactSensorCfg,
        # section 15) once available; left as zeros for the Stage-A smoke test.
        contact_flags = torch.zeros(self.num_envs, len(self._fingertip_body_idx), device=self.device)

        obs = torch.cat(
            [
                self._robot.data.joint_pos[:, self._actuated_dof_idx],
                self._robot.data.joint_vel[:, self._actuated_dof_idx],
                progress.unsqueeze(-1),
                theta_dot.unsqueeze(-1),
                s_goal.unsqueeze(-1),
                z_ref,
                contact_flags,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        window_progress, window_velocity, window_wrench = self._record_history_and_get_window()

        distance = compute_distance(
            self.cfg.reward_mode,
            self._reference_progress,
            self._reference_velocity,
            self._reference_wrench,
            window_progress,
            window_velocity,
            window_wrench,
            self.cfg.sigma_v,
            self.cfg.sigma_tau,
            self._cost_weights,
            self.cfg.sinkhorn_epsilon,
            self.cfg.sinkhorn_iters,
        )
        shaping_reward = self._prev_distance - distance  # section 13 incremental reward
        self._prev_distance = distance

        theta = self._knob.data.joint_pos[:, self._knob_joint_idx]
        task_r, success = task_reward(theta, self._theta_goal, self.cfg.success_threshold)
        action_penalty = self._actions.pow(2).sum(dim=-1)

        return compose_reward(task_r, success, shaping_reward, action_penalty, self._reward_weights)

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

        self._history_progress[env_ids] = 0.0
        self._history_velocity[env_ids] = 0.0
        self._history_wrench[env_ids] = 0.0
        self._prev_theta_dot[env_ids] = 0.0
        self._prev_distance[env_ids] = 0.0
