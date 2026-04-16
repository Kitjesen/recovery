# Copyright (c) 2026 Qiongpei Technology
# SPDX-License-Identifier: Apache-2.0

"""Thunder articulation config, vendored from robot_lab/assets/thunder.py.

Only the `THUNDER_NOHEAD_CFG` variant (Thunder without sensor head) is
needed for recovery — it carries the canonical joint layout:

  leg joints (12):  FR/FL/RR/RL × hip/thigh/calf
  wheel joints (4): FR/FL/RR/RL _foot_joint

Actuators: DCMotor for leg joints (effort 120 Nm, stiff 100, damp 5),
ImplicitActuator for wheels (velocity drive, damp 1).

URDF ships with this package under `assets/thunder/urdf/thunder.urdf`
(resolved via `_ASSETS_DIR` below).
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Resolve bundled URDF path relative to this file.
# Package layout:  src/thunder_recovery/config/asset.py
#                  assets/thunder/urdf/thunder.urdf
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
_ASSETS_DIR = os.path.join(_PKG_ROOT, "assets")
THUNDER_URDF_PATH = os.path.join(_ASSETS_DIR, "thunder", "urdf", "thunder.urdf")


THUNDER_NOHEAD_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=THUNDER_URDF_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0, damping=0
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        joint_pos={
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": -0.8,
            "FR_calf_joint": 1.7,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.7,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 0.8,
            "RR_calf_joint": -1.7,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": -0.8,
            "RL_calf_joint": 1.7,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip": DCMotorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=17.48,
            stiffness=100.0,
            damping=5.0,
            friction=0.0,
        ),
        "thigh": DCMotorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=17.48,
            stiffness=100.0,
            damping=5.0,
            friction=0.0,
        ),
        "calf": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=17.48,
            stiffness=100.0,
            damping=5.0,
            friction=0.0,
        ),
        "wheel": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit_sim=60.0,
            velocity_limit_sim=16.956,
            stiffness=0.0,
            damping=1.0,
            friction=0.0,
        ),
    },
)


def _sanity_check_urdf_present() -> None:
    """Raise a clear error at import if the bundled URDF is missing."""
    if not os.path.isfile(THUNDER_URDF_PATH):
        raise FileNotFoundError(
            f"Thunder URDF not found at {THUNDER_URDF_PATH}. "
            f"If you installed via `pip install -e .`, make sure "
            f"`assets/thunder/` is present at the repo root."
        )


_sanity_check_urdf_present()
