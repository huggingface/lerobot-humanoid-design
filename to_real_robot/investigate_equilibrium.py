"""
Investigate why sim and real robot don't settle at the same equilibrium,
given the same gains, same tilt, same command.

Also checks L/R asymmetry from the tilted base.

Run:
    cd /home/virgile/devel/lerobot-humanoid-design/to_real_robot && \
        /home/virgile/micromamba/envs/mujoco313/bin/python investigate_equilibrium.py
"""
import sys
import time
import numpy as np

sys.path.insert(0, "/home/virgile/devel/lerobot-humanoid-design/to_real_robot")

import mujoco
from sim_robot import SimBipedalRobotController

# ── Real robot reference ─────────────────────────────────────────────────────
REAL_STATE_DEG = np.array([
    10.00, 10.39, 17.55, 39.91, 9.46, 5.43,
     8.94,  9.41,-17.77, 39.03, 10.78, 5.14,
])
REAL_TAU_NM = np.array([
    -0.065, -0.435, 2.505, 0.220, 0.204, -0.142,
     0.509,  0.454,-2.330, 0.923,-0.320, -0.048,
])
CMD_DEG = np.array([
    10.0, 10.0, 20.0, 40.0, 10.0, 5.0,
    10.0, 10.0,-20.0, 40.0, 10.0, 5.0,
])
JOINT_NAMES = [
    "L-hipz","L-hipx","L-hipy","L-knee","L-ankP","L-ankR",
    "R-hipz","R-hipx","R-hipy","R-knee","R-ankP","R-ankR",
]
KP = np.array([30,40,60,60,20,20, 30,40,60,60,20,20], dtype=float)

# ── Start sim (tilted base) ──────────────────────────────────────────────────
robot = SimBipedalRobotController(
    control_hz=200.0,
    fixed_base=True,
    fixed_base_quat_wxyz=(0.997730, 0.010104, 0.066584, 0.0),
)
robot.start(mode="control", auto_enable=True)

# Apply command
robot.set_action(
    left= {"hipz":10.0,"hipx":10.0,"hipy": 20.0,"knee":40.0,"ankle_pitch":10.0,"ankle_roll":5.0},
    right={"hipz":10.0,"hipx":10.0,"hipy":-20.0,"knee":40.0,"ankle_pitch":10.0,"ankle_roll":5.0},
)
print("Waiting 5s for equilibrium...")
time.sleep(5.0)

snap = robot.get_combined_state_snapshot()
sim_deg = np.asarray(snap["joint_state_deg"], dtype=float)
sim_tau = np.asarray(snap["joint_torque_nm"], dtype=float)

# ── Also read internal MuJoCo state for diagnostics ──────────────────────────
with robot._sim_lock:
    qpos = np.asarray(robot.data.qpos, dtype=float).copy()
    qvel = np.asarray(robot.data.qvel, dtype=float).copy()
    qfrc_bias = np.asarray(robot.data.qfrc_bias, dtype=float).copy()
    qfrc_applied = np.asarray(robot.data.qfrc_applied, dtype=float).copy()
    qfrc_actuator = np.asarray(robot.data.qfrc_actuator, dtype=float).copy()
    qfrc_passive = np.asarray(robot.data.qfrc_passive, dtype=float).copy()

    # Get per-joint DOF addresses
    joint_dof_adr = robot._joint_dof_adr

    # Read joint damping and frictionloss from model
    dof_damping = np.array([float(robot.model.dof_damping[adr]) for adr in joint_dof_adr])
    dof_fric = np.array([float(robot.model.dof_frictionloss[adr]) for adr in joint_dof_adr])

robot.stop()

# ── Joint-level forces ───────────────────────────────────────────────────────
j_bias = np.array([float(qfrc_bias[adr]) for adr in joint_dof_adr])
j_passive = np.array([float(qfrc_passive[adr]) for adr in joint_dof_adr])
j_actuator = np.array([float(qfrc_actuator[adr]) for adr in joint_dof_adr])
j_vel = np.array([float(qvel[adr]) for adr in joint_dof_adr])

# ── Print base state ─────────────────────────────────────────────────────────
print("\n" + "=" * 110)
print("BASE STATE")
print("=" * 110)
print(f"  qpos[0:7] = [{', '.join(f'{v:.6f}' for v in qpos[0:7])}]")
print(f"  qvel[0:6] = [{', '.join(f'{v:.6f}' for v in qvel[0:6])}]")
print(f"  qfrc_bias[0:6]    = [{', '.join(f'{v:.3f}' for v in qfrc_bias[0:6])}]")
print(f"  qfrc_applied[0:6] = [{', '.join(f'{v:.3f}' for v in qfrc_applied[0:6])}]")

# ── Comparison table ─────────────────────────────────────────────────────────
print("\n" + "=" * 110)
print("EQUILIBRIUM COMPARISON: sim (dynamic) vs real robot")
print("=" * 110)
print(f"{'Joint':<8} {'cmd':>6} │ {'sim_q':>7} {'real_q':>7} {'Δq_pos':>7}"
      f" │ {'sim_τ':>8} {'real_τ':>8} {'Δτ':>7} {'sign':>5}"
      f" │ {'damping':>7} {'fric':>5} {'|vel|':>7} {'τ_passive':>9}")
print("─" * 110)

for i, name in enumerate(JOINT_NAMES):
    dq = sim_deg[i] - REAL_STATE_DEG[i]
    dt = sim_tau[i] - REAL_TAU_NM[i]
    s = "OK" if (abs(sim_tau[i]) < 0.01 and abs(REAL_TAU_NM[i]) < 0.05) or np.sign(sim_tau[i]) == np.sign(REAL_TAU_NM[i]) else "BAD"
    print(f"{name:<8} {CMD_DEG[i]:>6.1f} │ {sim_deg[i]:>7.2f} {REAL_STATE_DEG[i]:>7.2f} {dq:>+7.2f}"
          f" │ {sim_tau[i]:>8.3f} {REAL_TAU_NM[i]:>8.3f} {dt:>+7.3f} {s:>5}"
          f" │ {dof_damping[i]:>7.4f} {dof_fric[i]:>5.3f} {abs(j_vel[i]):>7.5f} {j_passive[i]:>9.4f}")

# ── L/R asymmetry analysis ───────────────────────────────────────────────────
print("\n" + "=" * 110)
print("L/R ASYMMETRY (effect of base tilt)")
print("=" * 110)
pairs = [("hipz",0,6), ("hipx",1,7), ("hipy",2,8), ("knee",3,9), ("ankP",4,10), ("ankR",5,11)]
print(f"{'Joint':<8} │ {'sim_L':>7} {'sim_R':>7} {'|L|-|R|':>8} │ {'real_L':>7} {'real_R':>7} {'|L|-|R|':>8}"
      f" │ {'sim_τ_L':>8} {'sim_τ_R':>8} {'|τL|-|τR|':>10} │ {'real_τ_L':>8} {'real_τ_R':>8} {'|τL|-|τR|':>10}")
print("─" * 110)
for jname, il, ir in pairs:
    sdl, sdr = sim_deg[il], sim_deg[ir]
    rdl, rdr = REAL_STATE_DEG[il], REAL_STATE_DEG[ir]
    stl, str_ = sim_tau[il], sim_tau[ir]
    rtl, rtr = REAL_TAU_NM[il], REAL_TAU_NM[ir]
    print(f"{jname:<8} │ {sdl:>7.2f} {sdr:>7.2f} {abs(sdl)-abs(sdr):>+8.2f}"
          f" │ {rdl:>7.2f} {rdr:>7.2f} {abs(rdl)-abs(rdr):>+8.2f}"
          f" │ {stl:>8.3f} {str_:>8.3f} {abs(stl)-abs(str_):>+10.3f}"
          f" │ {rtl:>8.3f} {rtr:>8.3f} {abs(rtl)-abs(rtr):>+10.3f}")

# ── Force balance analysis ───────────────────────────────────────────────────
print("\n" + "=" * 110)
print("FORCE BALANCE at each joint DOF: qfrc_actuator + qfrc_passive + qfrc_applied = qfrc_bias + M*qacc")
print("(at equilibrium qacc≈0, so actuator + passive + applied ≈ bias)")
print("=" * 110)
print(f"{'Joint':<8} {'τ_actuator':>10} {'τ_passive':>10} {'τ_bias':>10} {'residual':>10}")
print("─" * 50)
for i, name in enumerate(JOINT_NAMES):
    adr = joint_dof_adr[i]
    residual = j_actuator[i] + j_passive[i] - j_bias[i]
    print(f"{name:<8} {j_actuator[i]:>10.4f} {j_passive[i]:>10.4f} {j_bias[i]:>10.4f} {residual:>10.4f}")

# ── Key insight ──────────────────────────────────────────────────────────────
print("\n" + "=" * 110)
print("PASSIVE FORCE IMPACT ON EQUILIBRIUM")
print("=" * 110)
print("At equilibrium: kp*(cmd - q) = τ_gravity - τ_passive")
print("If τ_passive is large, the settled position shifts away from the real robot.")
print()
for i, name in enumerate(JOINT_NAMES):
    tau_grav_est = j_bias[i]  # gravity torque at this joint
    tau_passive_i = j_passive[i]
    if abs(KP[i]) > 0 and abs(tau_grav_est) > 0.01:
        # Without passive: q_eq = cmd - τ_gravity/kp
        # With passive: q_eq = cmd - (τ_gravity - τ_passive)/kp
        # Shift = τ_passive / kp (in radians)
        shift_deg = np.rad2deg(tau_passive_i / KP[i])
        print(f"  {name:<8}: τ_passive={tau_passive_i:>+7.3f} Nm → shifts equilibrium by {shift_deg:>+6.2f}°"
              f"   (damping={dof_damping[i]:.3f}, fric={dof_fric[i]:.3f})")
