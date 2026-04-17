# Copyright (c) 2026 Qiongpei Technology
# SPDX-License-Identifier: Apache-2.0

"""Method 2 — mujoco_playground Go1 getup style.

Reference: github.com/google-deepmind/mujoco_playground
           → mujoco_playground/_src/locomotion/go1/getup.py

9 plain-weight rewards (2 gated on upright ∧ at-height) + 60/40 reset.
Public names are re-exported flat from `thunder_recovery.mdp`.
"""
