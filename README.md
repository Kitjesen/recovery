# Thunder Fall Recovery Policy

Standalone Isaac Lab task that teaches a wheeled-legged quadruped
(Thunder, Go2-W, etc.) to stand up from an arbitrary fallen pose.
Trained independently of the locomotion policy and switched in at
deployment when the robot detects a fall.

Reference: Deng et al., *Learning to Recover: Dynamic Reward Shaping with
Wheel-Leg Coordination for Fallen Robots*
(arXiv:[2506.05516](https://arxiv.org/abs/2506.05516)).

## What ships in this package

- **URDF + meshes** for Thunder (12 leg joints + 4 wheel joints), bundled
  under `assets/thunder/` — no external asset downloads required.
- **Recovery MDP**: 13 reward terms (paper Table I + support state +
  wheel-leg coord), free-fall reset with actuator-gain override,
  asymmetric actor-critic observations (priv_* signals for critic only).
- **Standalone env cfg**: `ThunderRecoveryEnvCfg` inherits
  `isaaclab.envs.ManagerBasedRLEnvCfg` directly — no `robot_lab`
  dependency.
- **Training script** wired to RSL-RL's `OnPolicyRunner`.

## Dependencies

Only **Isaac Lab** and **RSL-RL**:

- `isaaclab` + `isaacsim` >= 5.0
- `isaaclab_rl`
- `rsl-rl-lib >= 3.0.1`
- PyTorch (installed by Isaac Sim)

## Install

```bash
# 1. Clone this repo
git clone https://github.com/Kitjesen/recovery.git
cd recovery

# 2. Install the package (editable)
pip install -e .
# This installs `thunder_recovery` and registers the gym task on import:
#   RobotLab-Isaac-Velocity-Recovery-Thunder-v0
```

## Quick smoke test

```bash
python scripts/train.py \
    --task RobotLab-Isaac-Velocity-Recovery-Thunder-v0 \
    --num_envs 64 --headless --max_iterations 3 --seed 42
```

Expected output within ~30 s:
- `Actor MLP in=570, Critic MLP in=560` (asymmetric actor-critic)
- 13 `Episode_Reward/recovery_*` terms in the log
- `recovery_support_state > 0` by iteration 2 (some envs already 4-foot
  grounded thanks to diverse reset init)

## Full training (1 × RTX 3090)

```bash
CUDA_VISIBLE_DEVICES=0 nohup python scripts/train.py \
    --task RobotLab-Isaac-Velocity-Recovery-Thunder-v0 \
    --num_envs 4096 --headless --max_iterations 10000 \
    --seed 42 \
    > /tmp/recovery_full.log 2>&1 &

tensorboard --logdir logs/rsl_rl/thunder_recovery
```

Target, per paper: `recovery_success_rate` climbs toward **0.97+** in the
last 1 s of the episode by iter 6k-8k.

## Episode timeline

```
+------------------+---------------------+----------------------+
| Free-fall        | Exploration         | Convergence          |
| t in [0, 2 s]    | t in [2, ~3.5 s]    | t in [~3.5, 5 s]     |
| steps 0-99       | steps 100-~174      | steps ~175-249       |
+------------------+---------------------+----------------------+
| ED 0 -> 0.064    | ED 0.064 -> 0.34    | ED 0.34 -> 1.0       |
| actuator gains 0 | policy output used  | policy output used   |
| joints floppy    | (torques active)    |                      |
| (true torques=0) |                     |                      |
+------------------+---------------------+----------------------+
| Diverse fallen   | Task rewards weak;  | Task rewards dominate|
| states emerge:   | wheel-leg coord     | -> policy converges  |
| reset noise +    | reward drives wheel-| to precise standing  |
| floppy free-fall | assisted flipping   | posture              |
+------------------+---------------------+----------------------+
```

`ED(t) = (t / T)^3  in [0, 1]` — paper Eq. 1 (normalised).
`CW(i) = beta * decay^i` with beta=0.3, decay=0.968 — paper Eq. 3.

## Package layout

```
recovery/
├── pyproject.toml
├── README.md
├── assets/
│   └── thunder/
│       ├── urdf/thunder.urdf
│       └── meshes/*.STL
├── src/
│   └── thunder_recovery/
│       ├── __init__.py                 # gym.register on import
│       ├── config/
│       │   ├── asset.py                # THUNDER_NOHEAD_CFG
│       │   ├── recovery_env_cfg.py     # ThunderRecoveryEnvCfg
│       │   └── recovery_ppo_cfg.py
│       └── mdp/
│           ├── _utils.py               # ED / CW / step counter / dt
│           ├── events.py               # reset_with_freefall, zero_action_freefall
│           ├── observations.py         # priv_*
│           └── rewards.py              # 13 reward terms
└── scripts/
    ├── cli_args.py
    ├── train.py
    └── play.py
```

## Reward terms

All weights are the paper's Table I except `support_state` and
`wheel_leg_coord`, which are this implementation's interpretations of
section E and the wheel-leg coordination contribution.

| Term | Weight | Multiplier | Active window | Math |
| --- | --- | --- | --- | --- |
| `recovery_stand_joint_pos` | 42 | x ED | convergence | `exp(-|q - q_default|^2 / sigma^2)`, sigma=0.5 |
| `recovery_base_height` | 120 | x ED | convergence | `exp(-clip(h_t - h, 0)^2 / sigma^2)`, h_t=0.426, sigma=0.1 |
| `recovery_base_orientation` | 50 | x ED | convergence | `exp(-|g_b - [0,0,-1]|^2)` |
| `recovery_wheel_leg_coord` | 5 | x (1-ED) * tilt | exploration | `|omega_wheel|/40 * |g_xy|` |
| `recovery_support_state` | 5 | — (per-step binary) | post-free-fall | `1{all 4 feet contact}` |
| `recovery_body_collision` | -5e-2 | x CW | early training | `sum clip(|lambda_b|, 0, 50)^2` on base/thigh/calf |
| `recovery_action_rate_legs` | -1e-2 | x CW | early training | `sum (a_t - a_{t-1})^2` on leg action dims |
| `recovery_joint_velocity` | -2e-2 | — | all | `sum q_dot^2` over leg joints |
| `recovery_torques` | -2.5e-5 | — | all | `sum tau^2` over leg joints |
| `recovery_joint_acceleration` | -2.5e-7 | — | all | `sum q_dd^2` over leg joints |
| `recovery_wheel_velocity` | -2e-2 | x ED | convergence | `sum omega_wheel^2` |
| `recovery_success_rate` | 1e-6 | — | logging only | end-of-episode success flag |
| `recovery_step_counter` | 1e-6 | — | side-effect | advances per-env ED step counter |

## Observations (asymmetric actor-critic)

**Actor** (570-dim, noisy; 10-frame history):
- `base_lin_vel` (3 x 10) with Unoise +-0.1
- `base_ang_vel` (3 x 10) with Unoise +-0.2
- `projected_gravity` (3 x 10) with Unoise +-0.05
- `joint_pos_rel` (16 x 10) with Unoise +-0.01
- `joint_vel_rel` (16 x 10) with Unoise +-1.5
- `last_action` (16 x 10)

**Critic** (560-dim, clean + privileged):
- Clean actor-side obs minus the IMU-noisy `base_lin_vel` (540-dim, 10-frame)
- `priv_base_lin_vel_clean` (3), `priv_base_ang_vel_clean` (3)
- `priv_base_height` (1), `priv_foot_contact` (4), `priv_body_contact_force` (9)

## Success criteria (paper-equivalent)

Applied by `recovery_success_rate` (logging only):

- `base_height > 0.30 m`
- `|q - q_default| < 0.5 rad`
- `max |q_dot| < 0.1 rad/s`
- `|g_b - [0,0,-1]| < 0.1`

Paper reports 99.1 % (KYON) and 97.8 % (Go2-W) with the same criteria.

## Deployment policy switching

```python
if projected_gravity_z > -0.5 or base_height < 0.25:
    action = recovery_policy(obs)
else:
    action = locomotion_policy(obs)
```

## Free-fall implementation

Paper §III-A says "joint torques set to zero". Isaac Lab has no clean
per-env torque override, so we use two events with a fallback:

1. **Primary path** — zero each actuator's per-env `stiffness` and
   `damping` tensors for envs still in free-fall, and pin the PD target to
   the current `joint_pos`. Cached originals are restored when an env
   exits free-fall. Works on `ImplicitActuator` / `IdealPDActuator` with
   `(num_envs, num_joints)` gain tensors.
2. **Fallback** — rigid-at-default teleport (write `(default_pos, 0 vel)`
   every step). Used if the actuator class exposes only scalar gains.

Two event bindings:

- `freefall_zero_action_on_reset` (mode="reset") zeros gains at t=0
  immediately, so the very first physics step has no PD impulse.
- `freefall_zero_action` (mode="interval", 0.02 s) keeps gains zero each
  subsequent step while `step_count < 100`; restores gains for envs that
  have exited free-fall.

On reset, `joint_pos` gets `Uniform(+-0.3)` rad noise on legs and is then
clamped to `soft_joint_pos_limits` to prevent PhysX clamping from
producing impulse torques on the first physics step.

## Unverified assumptions

These need confirmation against the actual URDF and Isaac Lab build on
your training server. They will surface as clean init-time errors if
wrong.

- Base body is named `base_link` (used by DR mass event and body-collision
  sensor cfg).
- Lower-leg bodies match regex `.*calf.*` (KYON-style robots use `shank`).
- Thigh bodies match `.*thigh.*`.
- Foot/wheel bodies match `.*_foot` (Thunder) — pass a different regex if
  your URDF uses a different convention (e.g. `.*wheel.*` for some Go2-W
  variants).
- Actuator class exposes `stiffness`, `damping` as mutable
  `(num_envs, num_joints)` tensors. The code validates this at first call
  and falls back to rigid-at-default teleport on mismatch.
- Isaac Lab `randomize_rigid_body_mass(operation="scale", ...,
  mass_distribution_params=(0.9, 1.1))` interprets the tuple as a scale
  multiplier range — this is the current API (>= 2024) but older pins took
  it as an absolute kg range.

## License

Apache-2.0. Thunder URDF / meshes: bundled with permission of Qiongpei
Technology.

## References

- Paper: [Learning to Recover (arXiv:2506.05516)](https://arxiv.org/abs/2506.05516)
- Project page: [L2R-WheelLegCoordination](https://boyuandeng.github.io/L2R-WheelLegCoordination/)
- Author's public deployment stack: [Recovery_go2w](https://github.com/boyuandeng/Recovery_go2w)
