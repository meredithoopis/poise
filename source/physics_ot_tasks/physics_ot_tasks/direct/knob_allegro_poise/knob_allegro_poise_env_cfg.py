"""Stage-3 IsaacLab env cfg: same Allegro hand + knob USD assets as the
original ``knob_allegro`` task (Stage A/Experiment-0, untouched), but wired
for POISE-OT's two-critic SAC (Eq observation/task_reward) instead of PPO's
single blended reward. See ``knob_allegro_poise_env.py`` for why this is a
new sibling task rather than a modification of ``knob_allegro_env.py``.
"""

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

# Reuses the exact same asset-path override pattern and USD assets as
# knob_allegro_env_cfg.py -- see that file's comments for the rationale
# (PHYSICS_OT_REPO_ROOT/PHYSICS_OT_DATA_DIR/PHYSICS_OT_ALLEGRO_USD_PATH).


def _default_repo_root() -> Path:
    override = os.environ.get("PHYSICS_OT_REPO_ROOT")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[5]


REPO_ROOT = _default_repo_root()
DATA_DIR = Path(os.environ.get("PHYSICS_OT_DATA_DIR", REPO_ROOT / "data"))
KNOB_USD_PATH = str(DATA_DIR / "objects" / "knob.usd")

with open(REPO_ROOT / "configs" / "knob.yaml") as _f:
    _KNOB_CFG = yaml.safe_load(_f)
_KNOB = _KNOB_CFG["knob"]

with open(REPO_ROOT / "configs" / "poise_ot.yaml") as _f:
    _POISE_CFG = yaml.safe_load(_f)
_TOKEN = _POISE_CFG["token"]
_REWARD = _POISE_CFG["reward"]
# mech_cost/alignment/sac hyperparameters are NOT duplicated into this env
# cfg -- token_builder/mech_cost/alignment/two_critic_sac all live in
# scripts/stage3/train_stage3.py, loading configs/poise_ot.yaml directly,
# unchanged from Stage 1 (step 11's explicit instruction).

# Observation layout (Eq observation): q(16) + qdot(16) + x^O(1, knob theta)
# + V^O(1, knob theta_dot) + z_ref(5, the Eq 30 POISE token
# [s,theta_dot,J_tau,P,P_diss]) + "c_t" contact info (4, per-fingertip flags)
# = 43. NOTE: the paper reuses the symbol "c_t" for two different things --
# Eq observation's c_t is *contact* information, Sec 4.4's c_t is the
# *confidence gate* on a reference token; these are unrelated quantities
# despite the shared symbol, and only the latter appears in z_ref/the
# alignment cost.
OBSERVATION_DIM = 16 + 16 + 1 + 1 + 5 + 4

KNOB_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Knob",
    spawn=sim_utils.UsdFileCfg(
        usd_path=KNOB_USD_PATH,
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, -0.17, 0.5),
        joint_pos={"knob_joint": 0.0},
    ),
    actuators={},
)

_allegro_usd_override = os.environ.get("PHYSICS_OT_ALLEGRO_USD_PATH")
if _allegro_usd_override:
    ROBOT_CFG = ALLEGRO_HAND_CFG.replace(spawn=ALLEGRO_HAND_CFG.spawn.replace(usd_path=_allegro_usd_override))
else:
    ROBOT_CFG = ALLEGRO_HAND_CFG


@configclass
class KnobAllegroPoiseEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 6.0
    action_space = 16
    observation_space = OBSERVATION_DIM
    state_space = 0

    # simulation -- same 120Hz physics / 30Hz control split as the paper's
    # "Simulation configuration" section and the original knob_allegro task.
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # Start small (step 6/11: "no asset yet"/"a handful of envs, not
    # 2048+ yet" -- the regression-check script overrides this via
    # parse_env_cfg's --num_envs, this default is just for a bare launch test).
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=0.75, replicate_physics=True)

    robot_cfg: ArticulationCfg = ROBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    knob_cfg: ArticulationCfg = KNOB_CFG.replace(prim_path="/World/envs/env_.*/Knob")

    actuated_joint_names = [
        "index_joint_0", "middle_joint_0", "ring_joint_0", "thumb_joint_0",
        "index_joint_1", "index_joint_2", "index_joint_3",
        "middle_joint_1", "middle_joint_2", "middle_joint_3",
        "ring_joint_1", "ring_joint_2", "ring_joint_3",
        "thumb_joint_1", "thumb_joint_2", "thumb_joint_3",
    ]
    fingertip_body_names = ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"]
    knob_joint_name = "knob_joint"
    act_moving_average = 1.0

    # knob physics (configs/knob.yaml -- same validated values as knob_allegro_env_cfg.py)
    knob_theta0 = _KNOB["theta0"]
    knob_theta_goal = _KNOB["theta_goal"]
    knob_inertia = _KNOB["inertia"]
    knob_damping = _KNOB["damping"]
    knob_stiffness = _KNOB["stiffness"]
    knob_coulomb_torque = _KNOB["coulomb_torque"]
    knob_v_eps = _KNOB["v_eps"]
    success_threshold = _KNOB["success_threshold"]
    success_hold_time = _KNOB["success_hold_time"]
    wrench_accel_smoothing = _KNOB["wrench_accel_smoothing"]

    # task-and-safety reward (Eq task_reward) -- same fields as poise_ot.knob_env.KnobEnvConfig
    success_bonus = _REWARD["success_bonus"]
    action_weight = _REWARD["action_weight"]
    unsafe_weight = _REWARD["unsafe_weight"]

    # token/alignment (only impulse_window is needed inside the env itself,
    # for exposing a comparable "tau" via inverse dynamics -- the rest of
    # token_builder/mech_cost/alignment live in scripts/stage3/train_stage3.py,
    # unchanged from Stage 1, per step 11's explicit instruction)
    impulse_window = _TOKEN["impulse_window"]

    # The env builds its own copy of the *same synthetic* Stage-1 reference
    # (poise_ot.reference_generator.scripted_reference, same KnobParams) for
    # z_ref in the observation -- not loaded from a file, so it can never
    # silently drift from what train_stage3.py's own alignment/reward loop
    # uses (step 11's regression check requires these to be identical).
    reference_num_tokens = _KNOB_CFG["reference"]["num_tokens"]
    reference_smoothing = _KNOB_CFG["reference"]["smoothing"]
