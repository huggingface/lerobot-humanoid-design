"""
Diagnostic script: investigate asymmetric knee torques (L=0.37 Nm vs R=0.32 Nm).

Three control experiments:
  Exp A: hipy=0° both sides, knee=40° — isolates model asymmetry from hip geometry
  Exp B: hipy=+20° both sides (same sign) — checks if same local-frame command gives symmetric result
  Exp C: hipy=+20° left / hipy=-20° right (original "symmetric" command)

Run with:
  cd /home/virgile/devel/lerobot-humanoid-design/to_real_robot && \\
      /home/virgile/micromamba/envs/mujoco313/bin/python debug_torque.py 2>&1
"""
import sys
import time
import numpy as np

sys.path.insert(0, "/home/virgile/devel/lerobot-humanoid-design/to_real_robot")

import mujoco
from sim_robot import SimBipedalRobotController

JOINT_NAMES = [
    "L-hipz", "L-hipx", "L-hipy", "L-knee", "L-ankle_p", "L-ankle_r",
    "R-hipz", "R-hipx", "R-hipy", "R-knee", "R-ankle_p", "R-ankle_r",
]


def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def run_experiment(robot, label, left_cmd, right_cmd, settle_s=3.0):
    """Apply a command, wait to settle, then print full diagnostic output."""
    print_section(f"EXPERIMENT: {label}")
    print(f"  left:  {left_cmd}")
    print(f"  right: {right_cmd}")

    robot.set_action(left=left_cmd, right=right_cmd)
    time.sleep(settle_s)

    snap = robot.get_combined_state_snapshot()
    sim_deg = snap["joint_state_deg"]
    sim_tau = snap["joint_torque_nm"]

    model = robot.model
    with robot._sim_lock:
        ctrl_vals = np.asarray(robot.data.ctrl, dtype=float).copy()
        qfrc_act  = np.asarray(robot.data.qfrc_actuator, dtype=float).copy()
        qpos      = np.asarray(robot.data.qpos, dtype=float).copy()
        qvel      = np.asarray(robot.data.qvel, dtype=float).copy()

    print()
    print(f"  {'Joint':<14} {'cmd_deg':>9} {'qpos_deg':>9} {'delta':>7} {'qfrc_act':>10} {'tau_nm':>9}")
    print("  " + "-" * 60)
    for aid in range(int(model.nu)):
        jid   = int(model.actuator_trnid[aid, 0])
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) if jid >= 0 else "?"
        q_adr = int(model.jnt_qposadr[jid]) if jid >= 0 else -1
        qp    = float(np.rad2deg(qpos[q_adr])) if q_adr >= 0 else float("nan")
        cd    = float(np.rad2deg(ctrl_vals[aid]))
        dof_adr = int(model.jnt_dofadr[jid]) if jid >= 0 else -1
        qf    = float(qfrc_act[dof_adr]) if dof_adr >= 0 else float("nan")
        tau   = sim_tau[aid] if aid < len(sim_tau) else float("nan")
        print(f"  {jname:<14} {cd:>9.2f} {qp:>9.2f} {cd-qp:>7.2f} {qf:>10.4f} {tau:>9.4f}")

    # Knee-specific summary
    left_knee_aid  = next(aid for aid in range(int(model.nu))
                          if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
                                               int(model.actuator_trnid[aid, 0])) == "knee_left")
    right_knee_aid = next(aid for aid in range(int(model.nu))
                          if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
                                               int(model.actuator_trnid[aid, 0])) == "knee_right")
    lk_jid = int(model.actuator_trnid[left_knee_aid, 0])
    rk_jid = int(model.actuator_trnid[right_knee_aid, 0])
    lk_dof = int(model.jnt_dofadr[lk_jid])
    rk_dof = int(model.jnt_dofadr[rk_jid])

    lk_tau = float(qfrc_act[lk_dof])
    rk_tau = float(qfrc_act[rk_dof])
    lk_vel = float(qvel[lk_dof])
    rk_vel = float(qvel[rk_dof])

    print()
    print(f"  KNEE SUMMARY:")
    print(f"    knee_left  qfrc_actuator={lk_tau:+.4f} Nm   qvel={lk_vel:.4f} rad/s")
    print(f"    knee_right qfrc_actuator={rk_tau:+.4f} Nm   qvel={rk_vel:.4f} rad/s")
    print(f"    asymmetry = {abs(lk_tau - rk_tau):.4f} Nm  ({100*abs(lk_tau - rk_tau)/max(abs(lk_tau), abs(rk_tau), 1e-9):.1f}%)")
    print(f"    knee_left  qpos={np.rad2deg(qpos[int(model.jnt_qposadr[lk_jid])]):.2f} deg")
    print(f"    knee_right qpos={np.rad2deg(qpos[int(model.jnt_qposadr[rk_jid])]):.2f} deg")


# ─── STATIC MODEL ANALYSIS ──────────────────────────────────────────────────

print_section("STATIC MJCF ASYMMETRY ANALYSIS")
print()
print("Segment masses (should be identical L/R):")
print("  hipz body:  R=0.8 kg   L=0.8 kg   -> EQUAL")
print("  hipx body:  R=0.5 kg   L=0.5 kg   -> EQUAL")
print("  thigh body: R=2.4 kg   L=2.4 kg   -> EQUAL")
print("  shin body:  R=0.7 kg   L=0.7 kg   -> EQUAL")
print("  ankle body: R=0.012 kg L=0.012 kg -> EQUAL")
print("  foot body:  R=0.22 kg  L=0.22 kg  -> EQUAL")
print()
print("Segment inertia tensors (fullinertia = ixx iyy izz ixy ixz iyz):")
print()
print("  THIGH:")
print("    R: ixx=0.01744  iyy=0.00866  izz=0.02361  (COM: +0.050, +0.079, -0.035)")
print("    L: ixx=0.01806  iyy=0.00826  izz=0.02384  (COM: -0.043, -0.083, -0.035)")
print("    ixx diff: 3.5%   iyy diff: 4.6%   (different local frames -> expected)")
print()
print("  SHIN (most asymmetric):")
print("    R: ixx=0.000600  iyy=0.001388  izz=0.001765  (COM: -0.077, -0.049, +0.000)")
print("    L: ixx=0.001739  iyy=0.000242  izz=0.001760  (COM: +0.001, -0.091, -0.000)")
print("    ixx diff: 65.5%!  iyy diff: 82.6%!  (same mass, VERY different tensor)")
print("    -> The shin body has a different local-frame orientation on left vs right.")
print("    -> This is physically inconsistent unless the body quat accounts for it.")
print()
print("Knee joint parameters (CRITICAL for torque asymmetry):")
print("  knee_right: damping=2.2641  frictionloss=0.9195  armature=0.12330025")
print("  knee_left:  damping=1.9641  frictionloss=1.0765  armature=0.12330331")
print("  damping diff:      0.300 (13.3%)")
print("  frictionloss diff: 0.157 (14.6%)")
print()
print("  At static equilibrium, qvel~=0 so damping contribution ~= 0.")
print("  But frictionloss acts as a STATIC friction floor (Coulomb model).")
print("  The PD controller output = kp*(q_target - q) + kd*(qvel_target - qvel)")
print("  At equilibrium, residual error drives the actuator output to oppose frictionloss.")
print("  -> frictionloss 14.6% diff is a strong candidate for the 15% torque asymmetry.")

# ─── LOAD SIM ────────────────────────────────────────────────────────────────

print_section("Loading sim robot (fixed_base=True)")
robot = SimBipedalRobotController(control_hz=200.0, fixed_base=True)
robot.start(mode="control", auto_enable=True)
print("Robot started.")

model = robot.model

# Print knee joint params from loaded model
print()
print("Knee joint params from loaded model:")
for jname in ("knee_left", "knee_right"):
    jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    damp = float(model.dof_damping[int(model.jnt_dofadr[jid])])
    fric = float(model.dof_frictionloss[int(model.jnt_dofadr[jid])])
    arm  = float(model.dof_armature[int(model.jnt_dofadr[jid])])
    lim  = (np.rad2deg(float(model.jnt_range[jid, 0])),
            np.rad2deg(float(model.jnt_range[jid, 1])))
    print(f"  {jname}: damping={damp:.5f}  frictionloss={fric:.5f}  armature={arm:.8f}  range=[{lim[0]:.1f}, {lim[1]:.1f}] deg")

# ─── EXPERIMENTS ─────────────────────────────────────────────────────────────

# Settle from zero first
robot.set_action(
    left= {"hipz": 0.0, "hipx": 0.0, "hipy": 0.0, "knee": 0.0, "ankle_pitch": 0.0, "ankle_roll": 0.0},
    right={"hipz": 0.0, "hipx": 0.0, "hipy": 0.0, "knee": 0.0, "ankle_pitch": 0.0, "ankle_roll": 0.0},
)
time.sleep(1.0)

# Experiment A: hipy=0, knee=40 (pure symmetric, no hip flex)
run_experiment(
    robot,
    label="A: hipy=0° both, knee=40° both — isolates model asymmetry",
    left_cmd= {"hipz": 0.0, "hipx": 0.0, "hipy": 0.0, "knee": 40.0, "ankle_pitch": 0.0, "ankle_roll": 0.0},
    right_cmd={"hipz": 0.0, "hipx": 0.0, "hipy": 0.0, "knee": 40.0, "ankle_pitch": 0.0, "ankle_roll": 0.0},
    settle_s=3.0,
)

# Experiment B: hipy=+20 both sides (same local-frame sign)
run_experiment(
    robot,
    label="B: hipy=+20° BOTH sides (same local-frame sign), knee=40°",
    left_cmd= {"hipz": 0.0, "hipx": 0.0, "hipy": 20.0, "knee": 40.0, "ankle_pitch": 0.0, "ankle_roll": 0.0},
    right_cmd={"hipz": 0.0, "hipx": 0.0, "hipy": 20.0, "knee": 40.0, "ankle_pitch": 0.0, "ankle_roll": 0.0},
    settle_s=3.0,
)

# Experiment C: original "symmetric" command (hipy=+20 left, hipy=-20 right)
run_experiment(
    robot,
    label="C: hipy=+20° left / hipy=-20° right (original command), knee=40°",
    left_cmd= {"hipz": 10.0, "hipx": 10.0, "hipy":  20.0, "knee": 40.0, "ankle_pitch": 10.0, "ankle_roll": 5.0},
    right_cmd={"hipz": 10.0, "hipx": 10.0, "hipy": -20.0, "knee": 40.0, "ankle_pitch": 10.0, "ankle_roll": 5.0},
    settle_s=3.0,
)

# Experiment D: hipy=-20 both sides (flipped sign for both)
run_experiment(
    robot,
    label="D: hipy=-20° BOTH sides, knee=40°",
    left_cmd= {"hipz": 0.0, "hipx": 0.0, "hipy": -20.0, "knee": 40.0, "ankle_pitch": 0.0, "ankle_roll": 0.0},
    right_cmd={"hipz": 0.0, "hipx": 0.0, "hipy": -20.0, "knee": 40.0, "ankle_pitch": 0.0, "ankle_roll": 0.0},
    settle_s=3.0,
)

robot.stop()
print()
print_section("DIAGNOSIS SUMMARY")
print("""
The 0.37 vs 0.32 Nm asymmetry is almost certainly case (B): MODEL ASYMMETRY.

Root causes found in MJCF robot.xml:

1. KNEE FRICTIONLOSS (strongest candidate):
   knee_right frictionloss = 0.9195
   knee_left  frictionloss = 1.0765
   diff = 0.157 (14.6%) — matches the ~15% torque asymmetry almost exactly.

   At static equilibrium, the PD controller needs to overcome joint frictionloss
   to hold the target angle. Higher frictionloss on the left knee means the
   controller must push harder, producing higher measured torque.

2. KNEE DAMPING (secondary, only matters at nonzero velocity):
   knee_right damping = 2.2641
   knee_left  damping = 1.9641
   diff = 0.300 (13.3%) — will compound asymmetry during dynamic motion.

3. SHIN INERTIA TENSOR (physically suspicious but may be frame-artifact):
   shin_right: ixx=0.000600  iyy=0.001388  (COM at -0.077, -0.049)
   shin_left:  ixx=0.001739  iyy=0.000242  (COM at +0.001, -0.091)
   ixx diff = 65%!  iyy diff = 83%!
   Same mass (0.7 kg) but wildly different moment of inertia axes.
   The COM positions are also inconsistent: L.x ≈ 0 but R.x = -0.077 m.
   This is likely a different-orientation local frame, but the NET gravity
   loading on the knee is set by the COM position in world frame, which
   MuJoCo correctly computes from the body chain. So this alone may not
   change quasi-static torques but will affect dynamics.

4. THIGH INERTIA TENSOR (minor, ~4% difference):
   Diagonal elements differ by 3-5% between L and R thighs.

WHAT TO FIX:
  - Equalize knee_left and knee_right frictionloss (e.g., set both to the mean ~0.998)
  - Equalize knee_left and knee_right damping (e.g., set both to ~2.114)
  - Inspect whether shin body quat/pos chain produces the same world-frame COM
    for both legs when joints are at the same commanded angle.

If Exp A shows asymmetry with hipy=0, the issue is definitely model asymmetry (not hip geometry).
If Exp A is symmetric but Exp C is not, the asymmetry comes from geometric loading differences.
""")
