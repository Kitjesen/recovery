# Copyright (c) 2026 Qiongpei Technology
# SPDX-License-Identifier: Apache-2.0

"""Thunder fall-recovery environment — standalone, no robot_lab dependency.

Based on 'Learning to Recover' (arXiv:2506.05516).

Design notes:
- Scene: flat plane (recovery does not use rough terrain).
- Actor obs: 10-frame history on base_ang_vel, projected_gravity,
  joint_pos_rel, joint_vel_rel, last_action; base_lin_vel from IMU (noisy).
- Critic obs: same actor stack (clean) + privileged priv_* (instant).
- No velocity commands, no height scan — the recovery policy is
  command-free by design.
- Events: reset_with_freefall + zero_action_freefall (recovery core), plus
  light DR on base mass (+-10 %), external force/torque on reset, startup
  friction randomisation.
- Rewards: 13 terms (paper Table I + support_state + wheel-leg coord).
- Terminations: timeout only (recovery never ends early).
- No curriculum.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp_core
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from thunder_recovery import mdp as recovery_mdp
from thunder_recovery.config.asset import THUNDER_NOHEAD_CFG


# -- Joint / body naming ------------------------------------------------
LEG_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
WHEEL_JOINT_NAMES = [
    "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint",
]
BASE_LINK_NAME = "base_link"
FOOT_LINK_REGEX = ".*_foot"          # bodies
COLLISION_BODY_REGEXES = [BASE_LINK_NAME, ".*thigh.*", ".*calf.*"]


# -- Scene --------------------------------------------------------------
@configclass
class RecoverySceneCfg(InteractiveSceneCfg):
    """Flat plane + Thunder robot + contact sensor. Minimal by design."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Robot: filled in via replace() to get {ENV_REGEX_NS} per-env prim path.
    robot: ArticulationCfg = THUNDER_NOHEAD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=False,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


# -- Actions ------------------------------------------------------------
@configclass
class RecoveryActionsCfg:
    """Position control for legs, velocity control for wheels.

    Scales mirror the ThunderHist baseline:
      hip joints:   0.125 (narrow range, hips do not need big swings)
      thigh/calf:   0.25
      wheels:       0.8   (paper Section C: reduced wheel scale for recovery)
    """

    joint_pos = mdp_core.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        scale={
            ".*_hip_joint": 0.125,
            "^(?!.*_hip_joint).*": 0.25,
        },
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )
    joint_vel = mdp_core.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINT_NAMES,
        scale=0.8,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )


# -- Observations -------------------------------------------------------
@configclass
class RecoveryObservationsCfg:
    """Asymmetric actor-critic observations (paper Fig. 3).

    Actor  : base_lin_vel (IMU noise) + base_ang_vel + proj_grav +
             joint_pos + joint_vel + last_action, 10-frame history.
    Critic : clean actor-side + priv_base_lin_vel_clean + priv_base_ang_vel_clean +
             priv_base_height + priv_foot_contact + priv_body_contact_force
             (history_length=0 for priv — instantaneous signals).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=mdp_core.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
            history_length=10,
        )
        base_ang_vel = ObsTerm(
            func=mdp_core.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            history_length=10,
        )
        projected_gravity = ObsTerm(
            func=mdp_core.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            history_length=10,
        )
        joint_pos = ObsTerm(
            func=mdp_core.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            history_length=10,
        )
        joint_vel = ObsTerm(
            func=mdp_core.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            history_length=10,
        )
        actions = ObsTerm(
            func=mdp_core.last_action,
            clip=(-100.0, 100.0),
            history_length=10,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        # Clean actor-style obs (no noise)
        base_ang_vel = ObsTerm(func=mdp_core.base_ang_vel, clip=(-100.0, 100.0), history_length=10)
        projected_gravity = ObsTerm(func=mdp_core.projected_gravity, clip=(-100.0, 100.0), history_length=10)
        joint_pos = ObsTerm(
            func=mdp_core.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            history_length=10,
        )
        joint_vel = ObsTerm(
            func=mdp_core.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            history_length=10,
        )
        actions = ObsTerm(func=mdp_core.last_action, clip=(-100.0, 100.0), history_length=10)

        # Privileged (instant, sim-ground-truth — not observable by onboard sensors)
        priv_base_lin_vel = ObsTerm(
            func=recovery_mdp.priv_base_lin_vel_clean,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-100.0, 100.0),
            history_length=0,
        )
        priv_base_ang_vel = ObsTerm(
            func=recovery_mdp.priv_base_ang_vel_clean,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-100.0, 100.0),
            history_length=0,
        )
        priv_base_height = ObsTerm(
            func=recovery_mdp.priv_base_height,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-5.0, 5.0),
            history_length=0,
        )
        priv_foot_contact = ObsTerm(
            func=recovery_mdp.priv_foot_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_REGEX),
                "threshold": 1.0,
            },
            clip=(0.0, 1.0),
            history_length=0,
        )
        priv_body_contact_force = ObsTerm(
            func=recovery_mdp.priv_body_contact_force,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=COLLISION_BODY_REGEXES)},
            clip=(0.0, 500.0),
            scale=0.01,
            history_length=0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# -- Events -------------------------------------------------------------
@configclass
class RecoveryEventsCfg:
    """Reset + DR + free-fall torque override.

    Order of reset-mode events is important: `reset_recovery` must run
    before `freefall_zero_action_on_reset` so the step counter is
    initialised before the gain-zeroing logic reads it. Python dataclass
    field order is preserved by isaaclab's reward/event manager.
    """

    # 1. Startup: friction randomisation
    randomize_friction = EventTerm(
        func=mdp_core.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    # 2. Reset: recovery random-fallen init (paper Section III-A)
    reset_recovery = EventTerm(
        func=recovery_mdp.reset_with_freefall,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "drop_height": 1.1},
    )

    # 3. Reset: zero actuator gains at t=0 (prevent first-step PD impulse)
    freefall_zero_action_on_reset = EventTerm(
        func=recovery_mdp.zero_action_freefall,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 4. Reset: base mass DR +-10 % (paper IV-A)
    randomize_mass = EventTerm(
        func=mdp_core.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    # 5. Reset: external force/torque perturbation
    apply_external_force = EventTerm(
        func=mdp_core.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "force_range": (-10.0, 10.0),
            "torque_range": (-10.0, 10.0),
        },
    )

    # 6. Interval (every control step): keep gains zero during free-fall
    freefall_zero_action = EventTerm(
        func=recovery_mdp.zero_action_freefall,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 7. Interval: periodic push (paper IV-A DR)
    push_robot = EventTerm(
        func=mdp_core.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


# -- Rewards (paper Table I + support_state + wheel-leg coord) ----------
@configclass
class RecoveryRewardsCfg:
    # Step counter: advances ED clock (side-effect term).
    recovery_step_counter = RewTerm(
        func=recovery_mdp.recovery_step_counter,
        weight=1e-6,
    )

    # Logging only: success indicator over last 1 s of episode.
    recovery_success_rate = RewTerm(
        func=recovery_mdp.recovery_success_rate,
        weight=1e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Task rewards (xED, paper Table I)
    recovery_stand_joint_pos = RewTerm(
        func=recovery_mdp.recovery_stand_joint_pos,
        weight=42.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "sigma": 0.5},
    )
    recovery_base_height = RewTerm(
        func=recovery_mdp.recovery_base_height,
        weight=120.0,
        params={"target_height": 0.426, "sigma": 0.1, "asset_cfg": SceneEntityCfg("robot")},
    )
    recovery_base_orientation = RewTerm(
        func=recovery_mdp.recovery_base_orientation,
        weight=50.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Behavior rewards (xCW, paper Table I)
    recovery_body_collision = RewTerm(
        func=recovery_mdp.recovery_body_collision,
        weight=-5e-2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=COLLISION_BODY_REGEXES),
            "force_clip": 50.0,
        },
    )
    recovery_action_rate_legs = RewTerm(
        func=recovery_mdp.recovery_action_rate_legs,
        weight=-1e-2,
    )

    # Support state (paper Section E, per-step binary)
    recovery_support_state = RewTerm(
        func=recovery_mdp.recovery_support_state,
        weight=5.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_REGEX), "threshold": 1.0},
    )

    # Wheel-leg coordination (paper core contribution, early-phase only)
    recovery_wheel_leg_coord = RewTerm(
        func=recovery_mdp.recovery_wheel_leg_coord,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "max_wheel_speed": 40.0},
    )

    # Constant penalties (paper Table I)
    recovery_joint_velocity = RewTerm(
        func=recovery_mdp.recovery_joint_velocity,
        weight=-2e-2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    recovery_torques = RewTerm(
        func=recovery_mdp.recovery_torques,
        weight=-2.5e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    recovery_joint_acceleration = RewTerm(
        func=recovery_mdp.recovery_joint_acceleration,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    recovery_wheel_velocity = RewTerm(
        func=recovery_mdp.recovery_wheel_velocity,
        weight=-2e-2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


# -- Terminations -------------------------------------------------------
@configclass
class RecoveryTerminationsCfg:
    """Timeout only — recovery never ends early (paper Section III)."""
    time_out = DoneTerm(func=mdp_core.time_out, time_out=True)


# -- Main env cfg -------------------------------------------------------
@configclass
class ThunderRecoveryEnvCfg(ManagerBasedRLEnvCfg):
    """Standalone recovery env — 5 s episodes, 50 Hz control, flat plane."""

    # Scene
    scene: RecoverySceneCfg = RecoverySceneCfg(num_envs=4096, env_spacing=2.5)

    # MDP
    observations: RecoveryObservationsCfg = RecoveryObservationsCfg()
    actions: RecoveryActionsCfg = RecoveryActionsCfg()
    events: RecoveryEventsCfg = RecoveryEventsCfg()
    rewards: RecoveryRewardsCfg = RecoveryRewardsCfg()
    terminations: RecoveryTerminationsCfg = RecoveryTerminationsCfg()

    # No commands (recovery is command-free by design).
    # No curriculum (paper does not use one).

    def __post_init__(self):
        # -- Sim --
        # 50 Hz control (decimation 4 x 0.005 s physics = 0.02 s)
        self.decimation = 4
        self.episode_length_s = 5.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        # -- Viewer --
        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.3)
