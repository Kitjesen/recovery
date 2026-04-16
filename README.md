# Thunder Fall Recovery Policy

Isaac Lab / robot_lab task that teaches a wheeled-legged quadruped (Thunder,
Go2-W, etc.) to stand up from an arbitrary fallen pose. Trained independently
of the locomotion policy and switched in at deployment when the robot
detects a fall.

Reference: Deng et al., *Learning to Recover: Dynamic Reward Shaping with
Wheel-Leg Coordination for Fallen Robots* (arXiv:[2506.05516](https://arxiv.org/abs/2506.05516)).

## Status

Paper-faithful implementation with 14 commits of audit-driven fixes on
branch `claude/fall-recovery-status-eMt7z`. Not yet trained — code is
ready for a smoke test on the GPU server.

Verbatim paper facts that the code aligns to:

| Paper requirement | Where enforced |
| --- | --- |
| Free-fall 2 s, 1.1 m drop, random orientation *and* joint angles, torques = 0 | `reset_with_freefall` + `zero_action_freefall` (reset-mode + interval) |
| Episode 5 s @ 50 Hz, terminate only on timeout | `episode_length_s = 5.0`, `terminations.illegal_contact = None` |
| Support state = all four wheels in contact | `recovery_support_state` (per-step binary) |
| Actor obs = base lin/ang vel, orientation, joint pos/vel, wheel speeds, history at 0.01 s interval | Inherited from `ThunderHistRoughEnvCfg`; `velocity_commands` stripped |
| Asymmetric actor-critic with privileged info | 5 × `priv_*` obs terms injected into `observations.critic` |
| DR: base mass ±10 %, friction, COM, external force, push | `randomize_rigid_body_mass_base` + base-class events |
| 15.85 % / 26.2 % torque reduction via wheel-leg coordination | `recovery_wheel_leg_coord` reward + ED-gated wheel penalty |

What the paper does not tell us verbatim (equations 1/3, Table I weights,
exact critic privileged-obs list, PPO hyperparameters) is implemented from
the project README, prior-commit context, and best-practice inference; see
[§ Unverified assumptions](#unverified-assumptions).

## Episode timeline

```
┌──────────────────┬─────────────────────┬──────────────────────┐
│ Free-fall        │ Exploration         │ Convergence          │
│ t ∈ [0, 2 s]     │ t ∈ [2, ~3.5 s]     │ t ∈ [~3.5, 5 s]      │
│ steps 0 – 99     │ steps 100 – ~174    │ steps ~175 – 249     │
├──────────────────┼─────────────────────┼──────────────────────┤
│ ED 0 → 0.064     │ ED 0.064 → 0.34     │ ED 0.34 → 1.0        │
│ actuator gains 0 │ policy output used  │ policy output used   │
│ joints floppy    │ (torques active)    │                      │
│ (true torques=0) │                     │                      │
├──────────────────┼─────────────────────┼──────────────────────┤
│ Diverse fallen   │ Task rewards weak;  │ Task rewards dominate│
│ states emerge:   │ wheel-leg coord     │ → policy converges   │
│ reset noise +    │ reward drives wheel-│ to precise standing  │
│ floppy free-fall │ assisted flipping   │ posture              │
└──────────────────┴─────────────────────┴──────────────────────┘
```

`ED(t) = (t / T)^3  ∈ [0, 1]`   — paper Eq. 1, normalised.
`CW(i) = β · decay^i`  with β = 0.3, decay = 0.968 — paper Eq. 3.

## File layout

```
recovery/
├── README.md                ← this file
├── config/
│   ├── recovery_env_cfg.py  ← ThunderRecoveryEnvCfg (env, events, DR, obs)
│   └── recovery_ppo_cfg.py  ← RecoveryPPORunnerCfg (PPO hyperparams)
├── mdp/
│   ├── __init__.py          ← re-exports from .rewards
│   └── rewards.py           ← ED / CW, 12 reward terms, free-fall hooks,
│                              asymmetric privileged observations
└── scripts/
    ├── train.py             ← skeleton (train via robot_lab's runner)
    └── play.py               ← skeleton
```

`recovery_env_cfg.py` inherits `ThunderHistRoughEnvCfg` from robot_lab —
reuses the articulation, sensors, action cfg, history stack, and base-class
DR; replaces rewards, termination, curriculum, commands, reset, and
observation specialisation.

## Reward terms

All weights are the paper's Table I except `support_state` and
`wheel_leg_coord`, which are this implementation's interpretations of
sections E and the wheel-leg coordination contribution.

| Term | Weight | Multiplier | Active window | Math |
| --- | --- | --- | --- | --- |
| `recovery_stand_joint_pos` | 42 | × ED | convergence | `exp(-‖q − q_default‖² / σ²)`, σ = 0.5 |
| `recovery_base_height` | 120 | × ED | convergence | `exp(-clip(h_t − h, 0)² / σ²)`, h_t = 0.426, σ = 0.1 |
| `recovery_base_orientation` | 50 | × ED | convergence | `exp(-‖g_b − [0,0,-1]‖²)` |
| `recovery_wheel_leg_coord` | 5 | × (1 − ED) · tilt | exploration | `‖ω_wheel‖/40 · ‖g_xy‖` |
| `recovery_support_state` | 5 | — (per-step binary) | post-free-fall | `1{all 4 feet contact}` |
| `recovery_body_collision` | −5e-2 | × CW | early training | `Σ clip(‖λ_b‖, 0, 50)²` over base/thigh/calf |
| `recovery_action_rate_legs` | −1e-2 | × CW | early training | `Σ (aₜ − aₜ₋₁)²` on leg action dims |
| `recovery_joint_velocity` | −2e-2 | — | all | `Σ q̇²` over leg joints |
| `recovery_torques` | −2.5e-5 | — | all | `Σ τ²` over leg joints |
| `recovery_joint_acceleration` | −2.5e-7 | — | all | `Σ q̈²` over leg joints |
| `recovery_wheel_velocity` | −2e-2 | × ED | convergence | `Σ ω_wheel²` |
| `recovery_success_rate` | 1e-6 | — | logging only | end-of-episode success flag |
| `recovery_step_counter` | 1e-6 | — | side-effect | advances per-env ED step counter |

Paper's contribution is the ED/CW shaping and wheel-leg coordination, not
specific weights — we use the Table-I-reported scales for the ED/CW
terms and keep the auxiliary terms tuned so no single term dominates PPO's
value head.

## Observations (asymmetric actor-critic)

Actor (deployment-realistic, noisy):

- `base_lin_vel` (Unoise ±0.1)
- `base_ang_vel`, `projected_gravity`, `joint_pos`, `joint_vel`, `last_action` — inherited from `ThunderHistRoughEnvCfg`
- history stack at 0.01 s interval — inherited
- `velocity_commands` removed (recovery has no cmd)
- `height_scan_group` removed (flat terrain)

Critic (actor obs without noise, plus privileged sim signals; all
`history_length = 0`):

- `priv_base_lin_vel_clean` (3) — ground-truth body-frame lin vel
- `priv_base_ang_vel_clean` (3) — ground-truth body-frame ang vel
- `priv_base_height` (1) — `root_pos_w[:, 2]`, invisible to encoders
- `priv_foot_contact` (4) — binary per-foot contact
- `priv_body_contact_force` (≈ 9) — contact magnitudes on base + thighs + calves

Inherited noisy `base_lin_vel` is explicitly dropped from the critic so the
field is not duplicated at two noise levels.

## Free-fall implementation

Paper §III-A says "joint torques set to zero". Isaac Lab has no clean
per-env torque override, so we use **two events** with a fallback:

1. **Primary path** — zero each actuator's per-env `stiffness` and
   `damping` tensors for envs still in free-fall, and pin the PD target to
   the current `joint_pos`. Cached originals are restored when an env
   exits free-fall. Works on `ImplicitActuator` / `IdealPDActuator` with
   `(num_envs, num_joints)` gain tensors.
2. **Fallback** — rigid-at-default teleport (write `(default_pos, 0 vel)`
   every step). Used if the actuator class exposes only scalar gains.

Two event bindings:

- `freefall_zero_action_on_reset` (mode = "reset") — zeros gains at
  `t = 0` immediately, so the very first physics step has no PD impulse.
- `freefall_zero_action` (mode = "interval", 0.02 s) — keeps gains zero
  each subsequent step while `step_count < 100`; restores gains for envs
  that have exited free-fall.

On reset, `joint_pos` gets `Uniform(±0.3)` rad noise on legs and is then
clamped to `soft_joint_pos_limits` to prevent PhysX clamping from
producing impulse torques on the first physics step.

## Usage

### Integration into robot_lab

`recovery/` is not yet installed as a gym task. On the training server:

```bash
cd /home/bsrl/hongsenpang/RLbased/robot_lab/
git fetch origin claude/fall-recovery-status-eMt7z
git checkout claude/fall-recovery-status-eMt7z
```

You then need to register `ThunderRecoveryEnvCfg` as a gym task and make
the `mdp.recovery_*` functions importable from
`robot_lab.tasks.manager_based.locomotion.velocity.mdp`. Integration
commands are in `TODO: integration notes` (pending until server recon
output is pasted).

### Smoke test

```bash
conda activate thunder2
python scripts/rsl_rl/train.py \
    --task <registered-recovery-task-id> \
    --num_envs 1024 --headless --max_iterations 500
```

What to look for in the first 500 iterations:

- `Episode_Reward/recovery_step_counter` > 0 → ED step counter is
  advancing; if zero, reward manager is pruning the side-effect term.
- `Episode_Reward/recovery_base_orientation` flat near 0 for iter 0–20,
  then rising — ED is suppressing task rewards during free-fall as
  intended.
- `Episode_Reward/recovery_support_state` non-zero some time after
  iteration 50 — at least some envs are reaching 4-foot contact.
- No NaN / inf in loss or reward.

### Full training

```bash
nohup python scripts/rsl_rl/train.py \
    --task <registered-recovery-task-id> \
    --num_envs 4096 --headless --max_iterations 10000 \
    > /tmp/recovery_full.log 2>&1 &
```

Target: `recovery_success_rate` climbs toward 0.9+ in the last 1 s of the
episode by iteration 6k–8k (author's public deployment checkpoint is
`model_7999.pt`).

## Success criteria (paper-equivalent)

Applied by `recovery_success_rate` (logging only):

- `base_height > 0.30 m`
- `‖q − q_default‖ < 0.5 rad`
- `max |q̇| < 0.1 rad/s`
- `‖g_b − [0,0,-1]‖ < 0.1`

Paper reports 99.1 % (KYON) and 97.8 % (Go2-W) with the same criteria.

## Deployment policy switching

```python
if projected_gravity_z > -0.5 or base_height < 0.25:
    action = recovery_policy(obs)
else:
    action = locomotion_policy(obs)
```

## Unverified assumptions

These need confirmation against the actual Thunder URDF / Isaac Lab build
on the training server; they will surface as clean init-time errors if
wrong:

- Base body is named `base_link` (used by DR mass event and body-collision
  sensor cfg).
- Lower-leg bodies match regex `.*calf.*` (KYON-style robots use `shank`).
- Thigh bodies match `.*thigh.*`.
- Wheel-driving joints match `.*foot.*` (user-confirmed for Thunder;
  configurable via `_get_joint_split(wheel_joint_regex=...)`).
- Actuator class exposes `stiffness`, `damping` as mutable
  `(num_envs, num_joints)` tensors. The code validates this at first call
  and falls back to rigid-at-default teleport on mismatch.
- Isaac Lab `randomize_rigid_body_mass(operation="scale", ...,
  mass_distribution_params=(0.9, 1.1))` interprets the tuple as a scale
  multiplier range — this is the current API (≥ 2024) but older pins took
  it as an absolute kg range.

## Commit history on this branch

```
eeb9bfd  fix: second audit pass — episode-boundary bugs
82de95a  fix: wheel joint regex — Thunder URDF uses '.*foot.*'
7ef7835  fix: 6 more shortcuts — step_counter weight, action_rate indices, CW coupling
0f385dd  refactor: cross-check pass — paper-faithful support, history, mass DR
fbdb861  refactor: full paper alignment — normalized ED, true torques=0
763dba4  fix: legs-only penalties, success metric, dt not hardcoded
3ecab63  feat: asymmetric actor-critic + wheel-leg coordination reward
2bf68ff  docs: clarify episode timeline
88d06d9  fix: restore ED shaping, realign weights with paper Table I
c074c11  feat: fall recovery policy for wheeled-legged quadruped (initial)
```

## References

- Paper: [Learning to Recover (arXiv:2506.05516)](https://arxiv.org/abs/2506.05516)
- Project page: [L2R-WheelLegCoordination](https://boyuandeng.github.io/L2R-WheelLegCoordination/)
- Author's public deployment stack: [Recovery_go2w](https://github.com/boyuandeng/Recovery_go2w)
