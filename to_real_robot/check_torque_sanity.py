"""
MuJoCo position-controller torque sanity check.

Verifies that τ_expected = kp × (q_cmd - q_actual) [rad] matches
the joint_torque_nm reported by the sim (data.qfrc_actuator).

Run from to_real_robot/ directory:
    python check_torque_sanity.py
"""

import sys
import time
import numpy as np

sys.path.insert(0, "/home/virgile/devel/lerobot-humanoid-design/to_real_robot")

from sim_robot import SimBipedalRobotController

# ---------------------------------------------------------------------------
# Known MJCF kp values (Nm/rad), joint order:
#   L-hipz, L-hipx, L-hipy, L-knee, L-ankP, L-ankR,
#   R-hipz, R-hipx, R-hipy, R-knee, R-ankP, R-ankR
# ---------------------------------------------------------------------------
KP = np.array([30, 40, 60, 60, 20, 20, 30, 40, 60, 60, 20, 20], dtype=float)

JOINT_NAMES = [
    "L-hipz", "L-hipx", "L-hipy", "L-knee", "L-ankP", "L-ankR",
    "R-hipz", "R-hipx", "R-hipy", "R-knee", "R-ankP", "R-ankR",
]

# Real robot reference torques for comparison
REAL_TAU_NM = np.array(
    [-0.065, -0.435, 2.505, 0.220, 0.204, -0.142,
      0.509,  0.454, -2.330, 0.923, -0.320, -0.048],
    dtype=float,
)

# ---------------------------------------------------------------------------
# 1. Instantiate with fixed_base=True and tilted quaternion
# ---------------------------------------------------------------------------
print("[1] Creating SimBipedalRobotController (fixed_base=True, 7.72° tilt) ...")
robot = SimBipedalRobotController(
    fixed_base=True,
    control_hz=200.0,
    fixed_base_quat_wxyz=(0.997730, 0.010104, 0.066584, 0.0),
)

robot.start(mode="control")

# ---------------------------------------------------------------------------
# 2. Standing pose (all zeros)
# ---------------------------------------------------------------------------
print("[2] Applying standing pose (all zeros) ...")
robot.set_action(
    left={"hipz": 0.0, "hipx": 0.0, "hipy": 0.0,
          "knee": 0.0, "ankle_pitch": 0.0, "ankle_roll": 0.0},
    right={"hipz": 0.0, "hipx": 0.0, "hipy": 0.0,
           "knee": 0.0, "ankle_pitch": 0.0, "ankle_roll": 0.0},
)

# ---------------------------------------------------------------------------
# 3. Wait 2 s for equilibrium
# ---------------------------------------------------------------------------
print("[3] Waiting 2 s for equilibrium ...")
time.sleep(2.0)

# ---------------------------------------------------------------------------
# 4. Apply test pose
#    L: hipy=+20°, knee=+40°, ankle_pitch=-20°
#    R: hipy=-20°, knee=+40°, ankle_pitch=+20°
#    All other joints stay at 0.
# ---------------------------------------------------------------------------
print("[4] Applying test pose ...")
robot.set_action(
    left={"hipz": 0.0, "hipx": 0.0, "hipy": 20.0,
          "knee": 40.0, "ankle_pitch": -20.0, "ankle_roll": 0.0},
    right={"hipz": 0.0, "hipx": 0.0, "hipy": -20.0,
           "knee": 40.0, "ankle_pitch": 20.0, "ankle_roll": 0.0},
)

# Track the commanded angles for the comparison table.
# These are in the joint (calibrated) space that get_combined_state_snapshot returns.
cmd_deg = np.array(
    [0.0,   0.0,  20.0, 40.0, -20.0,  0.0,
     0.0,   0.0, -20.0, 40.0,  20.0,  0.0],
    dtype=float,
)

# ---------------------------------------------------------------------------
# 5. Wait 3 s for equilibrium
# ---------------------------------------------------------------------------
print("[5] Waiting 3 s for equilibrium ...")
time.sleep(3.0)

# ---------------------------------------------------------------------------
# 6. Get snapshot
# ---------------------------------------------------------------------------
snap = robot.get_combined_state_snapshot()

# ---------------------------------------------------------------------------
# 7. Compute τ_expected
# ---------------------------------------------------------------------------
q_actual_deg = np.asarray(snap["joint_state_deg"], dtype=float)
tau_sim = np.asarray(snap["joint_torque_nm"], dtype=float)

delta_deg = cmd_deg - q_actual_deg
delta_rad = np.deg2rad(delta_deg)
tau_expected = KP * delta_rad

# ---------------------------------------------------------------------------
# 8. Print comparison table
# ---------------------------------------------------------------------------
print()
print(
    f"{'Joint':<12} {'q_cmd':>7} {'q_act':>7} {'Δq_deg':>8} "
    f"{'τ_exp':>11} {'τ_sim':>11} {'ratio':>7} {'sign_ok':>8}"
)
print("-" * 80)

for i, name in enumerate(JOINT_NAMES):
    q_c = cmd_deg[i]
    q_a = q_actual_deg[i]
    dq = delta_deg[i]
    te = tau_expected[i]
    ts = tau_sim[i]

    if abs(te) > 1e-6:
        ratio = ts / te
        ratio_str = f"{ratio:7.3f}"
    else:
        ratio_str = "    N/A"

    if abs(te) < 1e-6 and abs(ts) < 1e-6:
        sign_ok = " N/A"
    elif abs(te) < 1e-6:
        sign_ok = "  NO"
    else:
        sign_ok = " YES" if np.sign(te) == np.sign(ts) else "  NO"

    print(
        f"{name:<12} {q_c:>7.2f} {q_a:>7.2f} {dq:>8.3f} "
        f"{te:>11.4f} {ts:>11.4f} {ratio_str} {sign_ok:>8}"
    )

print()
print("[9] Real robot reference torques (for comparison):")
print(
    f"{'Joint':<12} {'τ_real':>11} {'τ_sim':>11} {'τ_exp':>11}"
)
print("-" * 48)
for i, name in enumerate(JOINT_NAMES):
    print(
        f"{name:<12} {REAL_TAU_NM[i]:>11.4f} {tau_sim[i]:>11.4f} {tau_expected[i]:>11.4f}"
    )

robot.stop()
print("\nDone.")
