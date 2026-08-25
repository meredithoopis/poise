from __future__ import annotations

import os
from pathlib import Path

import yaml

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG


def _default_repo_root() -> Path:
    """Resolve the pot repo root.

    Overridable via ``PHYSICS_OT_REPO_ROOT`` (e.g. if this extension is
    installed from a copy that is not co-located with the repo root) --
    never hardcode an absolute path here, since training happens on a
    separate remote machine from development.
    """
    override = os.environ.get("PHYSICS_OT_REPO_ROOT")
    if override:
        return Path(override)
    # source/physics_ot_tasks/physics_ot_tasks/direct/knob_allegro/<this file> -> repo root
    return Path(__file__).resolve().parents[5]


REPO_ROOT = _default_repo_root()
DATA_DIR = Path(os.environ.get("PHYSICS_OT_DATA_DIR", REPO_ROOT / "data"))
KNOB_USD_PATH = str(DATA_DIR / "objects" / "knob.usd")

# configs/knob.yaml is the single source of truth for these values; loaded
# once here rather than duplicated as hardcoded literals so the env cfg can
# never silently drift from what scripts/build_physics_reference.py used to
# build the reference trajectory.
with open(REPO_ROOT / "configs" / "knob.yaml") as _f:
    _CFG = yaml.safe_load(_f)
_KNOB = _CFG["knob"]
_NORM = _CFG["normalization"]
_COST = _CFG["cost_weights"]
_SINKHORN = _CFG["sinkhorn"]
_REWARD = _CFG["reward"]

# Observation layout (section 12): q(16) + qdot(16) + s(1) + theta_dot(1)
# + s_goal(1) + z_ref[s*,theta_dot*,tau*](3) + fingertip_contact_flags(4) = 42
OBSERVATION_DIM = 16 + 16 + 1 + 1 + 1 + 3 + 4


KNOB_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Knob",
    spawn=sim_utils.UsdFileCfg(
        usd_path=KNOB_USD_PATH,
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, -0.17, 0.5),
        joint_pos={"knob_joint": 0.0},
    ),
    # No implicit PD drive: the env applies the knob's own spring/damper/
    # Coulomb-friction torque explicitly every physics step via
    # KnobDynamics.passive_torque(), matching the target-conditioned inverse
    # dynamics model used to build the Physics-OT reference (section 5).
    actuators={},
)


@configclass
class KnobAllegroEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 6.0
    action_space = 16
    observation_space = OBSERVATION_DIM
    state_space = 0

    # simulation (section 17)
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # scene -- start small (512) while debugging; scale to 2048-4096 for real
    # training runs (section 18).
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=0.75, replicate_physics=True)

    # robot / object
    robot_cfg: ArticulationCfg = ALLEGRO_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    knob_cfg: ArticulationCfg = KNOB_CFG.replace(prim_path="/World/envs/env_.*/Knob")

    actuated_joint_names = [
        "index_joint_0",
        "middle_joint_0",
        "ring_joint_0",
        "thumb_joint_0",
        "index_joint_1",
        "index_joint_2",
        "index_joint_3",
        "middle_joint_1",
        "middle_joint_2",
        "middle_joint_3",
        "ring_joint_1",
        "ring_joint_2",
        "ring_joint_3",
        "thumb_joint_1",
        "thumb_joint_2",
        "thumb_joint_3",
    ]
    fingertip_body_names = [
        "index_link_3",
        "middle_link_3",
        "ring_link_3",
        "thumb_link_3",
    ]
    knob_joint_name = "knob_joint"
    act_moving_average = 1.0

    # knob physics (section 5), loaded from configs/knob.yaml above. Kept as
    # plain floats (broadcast to per-env tensors in the env) so per-env
    # domain randomization can be added later without changing this schema
    # (section 19, deferred beyond Stage A).
    knob_theta0 = _KNOB["theta0"]
    knob_theta_goal = _KNOB["theta_goal"]
    knob_inertia = _KNOB["inertia"]
    knob_damping = _KNOB["damping"]
    knob_stiffness = _KNOB["stiffness"]
    knob_coulomb_torque = _KNOB["coulomb_torque"]
    knob_v_eps = _KNOB["v_eps"]
    success_threshold = _KNOB["success_threshold"]

    # reward_mode selects which Physics-OT / baseline distance function
    # drives the shaping reward (Experiment 0, sections 25-26):
    #   "sparse"            -- task reward only (B0)
    #   "state_tracking"    -- pointwise L2 over (s, theta_dot) (B1)
    #   "dtw_physics"       -- DTW over (s, theta_dot, tau) (B2)
    #   "ot_state"          -- temporal OT over (s, theta_dot) only (B3)
    #   "pointwise_physics" -- pointwise L2 over (s, theta_dot, tau) (B4)
    #   "physics_ot"        -- temporal OT over (s, theta_dot, tau) (Ours)
    reward_mode: str = "physics_ot"

    # reward weights (section 13)
    reward_task = _REWARD["task"]
    reward_physics_ot = _REWARD["physics_ot"]
    reward_action = _REWARD["action"]
    reward_success_bonus = _REWARD["success_bonus"]

    # OT / baseline hyperparameters (sections 9-10)
    sigma_v = _NORM["sigma_v"]
    sigma_tau = _NORM["sigma_tau"]
    cost_lambda_progress = _COST["progress"]
    cost_lambda_velocity = _COST["velocity"]
    cost_lambda_wrench = _COST["wrench"]
    cost_lambda_time = _COST["time"]
    sinkhorn_epsilon = _SINKHORN["epsilon"]
    sinkhorn_iters = _SINKHORN["num_iters"]
    robot_window = _SINKHORN["robot_window"]  # T_R: rollout tokens used for OT

    reference_path: str = str(DATA_DIR / "references" / "knob_target_nominal.npz")
