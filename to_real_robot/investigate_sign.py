"""
Investigate torque sign convention: sim vs real robot.

Both robots hang above ground (fixed base). Joint positions verified to match.
This script sets sim qpos to the EXACT real-robot joint angles (no PD dynamics),
then compares:
  1. mj_inverse gravity torques (ground truth for the model)
  2. PD-expected torques τ = kp*(cmd - actual) using MJCF kp
  3. Real robot torques from hardware

If signs match between (1) or (2) and (3), the convention is correct.
If ALL joints flip, there's a systematic sign issue.

Run:
    cd /home/virgile/devel/lerobot-humanoid-design/to_real_robot && \
        /home/virgile/micromamba/envs/mujoco313/bin/python investigate_sign.py
"""
import sys
import numpy as np

sys.path.insert(0, "/home/virgile/devel/lerobot-humanoid-design/to_real_robot")

import mujoco

# ── Real robot reference data (from snapshot) ────────────────────────────────
REAL_STATE_DEG = np.array([
    10.001898945809245,   # L-hipz
    10.393033592507663,   # L-hipx
    17.552887998780648,   # L-hipy
    39.9113955254921,     # L-knee
    9.462104885035544,    # L-ankP
    5.4288964148751955,   # L-ankR
    8.943862547563981,    # R-hipz
    9.409922071635485,    # R-hipx
    -17.773108990310902,  # R-hipy
    39.03179440260044,    # R-knee
    10.780865147758217,   # R-ankP
    5.14316502461862,     # R-ankR
])

REAL_CMD_DEG = np.array([
    10.0,  10.0,  20.0,  40.0,  10.0,  5.0,   # left
    10.0,  10.0, -20.0,  40.0,  10.0,  5.0,   # right
])

REAL_TAU_NM = np.array([
    -0.065, -0.435, 2.505, 0.220, 0.204, -0.142,
     0.509,  0.454, -2.330, 0.923, -0.320, -0.048,
])

# Real robot IMU quaternion (xyzw): (0.0633, -0.0230, -0.9782, -0.1964)
# Pure-tilt quat (MuJoCo wxyz, yaw removed): (0.997730, 0.010104, 0.066584, 0.0)
FIXED_BASE_QUAT_WXYZ = (0.997730, 0.010104, 0.066584, 0.0)

JOINT_NAMES = [
    "L-hipz", "L-hipx", "L-hipy", "L-knee", "L-ankP", "L-ankR",
    "R-hipz", "R-hipx", "R-hipy", "R-knee", "R-ankP", "R-ankR",
]

# MJCF joint names (order must match JOINT_NAMES)
MJCF_JOINT_NAMES = [
    "hipz_left", "hipx_left", "hipy_left", "knee_left", "ankley_left", "anklex_left",
    "hipz_right", "hipx_right", "hipy_right", "knee_right", "ankley_right", "anklex_right",
]

# MJCF kp values (from sim_scene.xml actuator section)
KP_MJCF = np.array([30, 40, 60, 60, 20, 20, 30, 40, 60, 60, 20, 20], dtype=float)

# ── Load model ───────────────────────────────────────────────────────────────
SCENE_PATH = "/home/virgile/devel/lerobot_legged_robots/models/bipedal_plateform_no_arms_high_gain/mjcf/sim_scene.xml"
model = mujoco.MjModel.from_xml_path(SCENE_PATH)
data = mujoco.MjData(model)

# ── Identify joint DOF addresses ─────────────────────────────────────────────
joint_ids = []
joint_qpos_adr = []
joint_dof_adr = []
for jname in MJCF_JOINT_NAMES:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    assert jid >= 0, f"Joint {jname} not found"
    joint_ids.append(jid)
    joint_qpos_adr.append(int(model.jnt_qposadr[jid]))
    joint_dof_adr.append(int(model.jnt_dofadr[jid]))

# ── Set qpos to real robot configuration ─────────────────────────────────────
mujoco.mj_resetData(model, data)

# Base: fixed at tilt matching real robot
data.qpos[0] = 0.0
data.qpos[1] = 0.0
data.qpos[2] = 0.77  # height
qw, qx, qy, qz = FIXED_BASE_QUAT_WXYZ
data.qpos[3] = qw
data.qpos[4] = qx
data.qpos[5] = qy
data.qpos[6] = qz

# Set leg joints to real robot angles
for i, adr in enumerate(joint_qpos_adr):
    data.qpos[adr] = np.deg2rad(REAL_STATE_DEG[i])

# Zero velocity and acceleration
data.qvel[:] = 0.0
data.qacc[:] = 0.0

# ── Run mj_forward to update kinematics ──────────────────────────────────────
mujoco.mj_forward(model, data)

# ── Run mj_inverse to get gravity torques ────────────────────────────────────
# With qacc=0 and qvel=0, qfrc_inverse = gravity + Coriolis(=0) torque needed
# at each DOF to hold the configuration static.
mujoco.mj_inverse(model, data)
tau_inverse = np.array([float(data.qfrc_inverse[adr]) for adr in joint_dof_adr])

# ── Also read qfrc_actuator (from the position actuators at current ctrl) ────
# Set ctrl to the commanded angles (same as real robot command)
actuator_ids = {}
for aid in range(int(model.nu)):
    jid_act = int(model.actuator_trnid[aid, 0])
    aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
    for i, jid in enumerate(joint_ids):
        if jid_act == jid:
            actuator_ids[i] = aid
            break

for i, aid in actuator_ids.items():
    data.ctrl[aid] = np.deg2rad(REAL_CMD_DEG[i])

mujoco.mj_forward(model, data)
tau_actuator = np.array([float(data.qfrc_actuator[adr]) for adr in joint_dof_adr])

# ── Compute expected PD torque ───────────────────────────────────────────────
delta_deg = REAL_CMD_DEG - REAL_STATE_DEG
delta_rad = np.deg2rad(delta_deg)
tau_pd_expected = KP_MJCF * delta_rad

# ── Read kv values from model and compute kd contribution at zero vel ────────
# (should be zero, but verify)
kv_vals = []
for i, aid in sorted(actuator_ids.items()):
    kv_vals.append(float(model.actuator_biasprm[aid, 2]))  # kv is biasprm[2] for position actuators
kv_vals = np.abs(np.array(kv_vals))

# ── Print results ────────────────────────────────────────────────────────────
print()
print("=" * 100)
print("TORQUE SIGN INVESTIGATION: sim vs real at EXACT same joint configuration")
print("=" * 100)
print()
print("Configuration: real robot joint angles, tilted base (7.72° from IMU)")
print()

print(f"{'Joint':<8} {'q_act':>7} {'q_cmd':>7} {'Δq':>7}"
      f" │ {'τ_real':>8} {'τ_inv':>8} {'τ_act':>8} {'τ_pd':>8}"
      f" │ {'inv=real?':>9} {'act=real?':>9} {'pd=real?':>9}")
print("─" * 100)

n_sign_match_inv = 0
n_sign_match_act = 0
n_sign_match_pd = 0
n_meaningful = 0

for i, name in enumerate(JOINT_NAMES):
    q_a = REAL_STATE_DEG[i]
    q_c = REAL_CMD_DEG[i]
    dq = delta_deg[i]
    t_real = REAL_TAU_NM[i]
    t_inv = tau_inverse[i]
    t_act = tau_actuator[i]
    t_pd = tau_pd_expected[i]

    meaningful = abs(t_real) > 0.05

    def sign_cmp(a, b):
        if abs(a) < 0.01 or abs(b) < 0.01:
            return "~0"
        return "YES" if np.sign(a) == np.sign(b) else "NO"

    s_inv = sign_cmp(t_inv, t_real)
    s_act = sign_cmp(t_act, t_real)
    s_pd = sign_cmp(t_pd, t_real)

    if meaningful:
        n_meaningful += 1
        if s_inv == "YES": n_sign_match_inv += 1
        if s_act == "YES": n_sign_match_act += 1
        if s_pd == "YES": n_sign_match_pd += 1

    print(f"{name:<8} {q_a:>7.2f} {q_c:>7.2f} {dq:>7.2f}"
          f" │ {t_real:>8.3f} {t_inv:>8.3f} {t_act:>8.3f} {t_pd:>8.3f}"
          f" │ {s_inv:>9} {s_act:>9} {s_pd:>9}")

print("─" * 100)
print()
print(f"Sign matches with real (|τ_real| > 0.05):")
print(f"  mj_inverse:    {n_sign_match_inv}/{n_meaningful}")
print(f"  qfrc_actuator: {n_sign_match_act}/{n_meaningful}")
print(f"  kp*Δq:         {n_sign_match_pd}/{n_meaningful}")

# ── Check if negation fixes it ───────────────────────────────────────────────
n_neg_inv = sum(1 for i in range(12) if abs(REAL_TAU_NM[i]) > 0.05 and np.sign(-tau_inverse[i]) == np.sign(REAL_TAU_NM[i]))
n_neg_act = sum(1 for i in range(12) if abs(REAL_TAU_NM[i]) > 0.05 and np.sign(-tau_actuator[i]) == np.sign(REAL_TAU_NM[i]))
print(f"\nWith NEGATION applied:")
print(f"  -mj_inverse:    {n_neg_inv}/{n_meaningful}")
print(f"  -qfrc_actuator: {n_neg_act}/{n_meaningful}")

# ── Ratio analysis (for matching signs) ──────────────────────────────────────
print()
print("Torque ratio analysis (τ_real / τ_pd_expected):")
print(f"{'Joint':<8} {'τ_real':>8} {'τ_pd':>8} {'ratio':>8}")
print("─" * 30)
for i, name in enumerate(JOINT_NAMES):
    t_r = REAL_TAU_NM[i]
    t_p = tau_pd_expected[i]
    if abs(t_p) > 0.01:
        ratio = t_r / t_p
        print(f"{name:<8} {t_r:>8.3f} {t_p:>8.3f} {ratio:>8.3f}")
    else:
        print(f"{name:<8} {t_r:>8.3f} {t_p:>8.3f}      N/A")
