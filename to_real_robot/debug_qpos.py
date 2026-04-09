#!/usr/bin/env python3
"""Debug qpos vs joint_state mismatch."""
import sys
import os
sys.path.insert(0, '/home/virgile/devel/lerobot-humanoid-design/to_real_robot')
os.chdir('/home/virgile/devel/lerobot-humanoid-design/to_real_robot')

import time
import numpy as np
import mujoco

from sim_robot import SimBipedalRobotController, SIM_KNEES_BENT_POSE_RAD

print("Creating SimBipedalRobotController with fixed_base=True...")
robot = SimBipedalRobotController(control_hz=200.0, fixed_base=True)

# Disable floor contact
floor_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
if floor_id >= 0:
    robot.model.geom_contype[floor_id] = 0
    robot.model.geom_conaffinity[floor_id] = 0
    print("Floor contact disabled")

# Don't start, just check initial state
print("\n=== Before start ===")
print(f"data.qpos shape: {robot.data.qpos.shape}")
print(f"data.qpos[:7] = {robot.data.qpos[:7]}")  # free joint
print(f"data.qpos[7:] = {robot.data.qpos[7:]}")  # actual joints

# Print per joint
print("\n=== Joint qpos (in sim) ===")
for i, jname in enumerate(robot._joint_name_order):
    jid = robot._joint_id_order[i]
    qadr = robot._joint_qpos_adr[i]
    val_rad = robot.data.qpos[qadr]
    val_deg = np.rad2deg(val_rad)
    ref_val_deg = np.rad2deg(SIM_KNEES_BENT_POSE_RAD.get(jname, 0.0))
    print(f"  {jname}: qpos={val_deg:.2f}° (ref={ref_val_deg:.2f}°)")

# Check what the reference pose is
print("\n=== Reference pose (SIM_KNEES_BENT_POSE_RAD) ===")
for jname, val in SIM_KNEES_BENT_POSE_RAD.items():
    print(f"  {jname}: {np.rad2deg(val):.4f}°")

# Check what _read_joint_q_deg returns
q_deg = robot._read_joint_q_deg()
print("\n=== _read_joint_q_deg() ===")
for i, jname in enumerate(robot._joint_name_order):
    print(f"  [{i}] {jname}: {q_deg[i]:.4f}°")

# Check what ctrl would be sent for target knee=40 deg
target_knee_rad = np.deg2rad(40.107)
print(f"\n=== Ctrl for knee target=40.107° = {target_knee_rad:.4f} rad ===")
print(f"  qpos of knee_left = {robot.data.qpos[robot._joint_qpos_adr[3]]:.4f} rad")
print(f"  error = {target_knee_rad - robot.data.qpos[robot._joint_qpos_adr[3]]:.4f} rad")
print(f"  kp=60, expected torque = {60 * (target_knee_rad - robot.data.qpos[robot._joint_qpos_adr[3]]):.4f} Nm")

# Now start the robot and check after settling
robot.start(mode="control", auto_enable=True)
time.sleep(2.0)

print("\n=== After 2s of control ===")
with robot._sim_lock:
    q_mujoco = robot.data.qpos.copy()
    ctrl_mujoco = robot.data.ctrl.copy()
    qfrc_actuator = robot.data.qfrc_actuator.copy()

print(f"data.qpos[:7] = {q_mujoco[:7]}")
print(f"data.qpos[7:] = {q_mujoco[7:]}")
print(f"data.ctrl = {ctrl_mujoco}")
print(f"qfrc_actuator = {qfrc_actuator}")

print("\n=== Joint qpos after settling ===")
for i, jname in enumerate(robot._joint_name_order):
    qadr = robot._joint_qpos_adr[i]
    val_rad = q_mujoco[qadr]
    val_deg = np.rad2deg(val_rad)
    print(f"  {jname}: qpos={val_deg:.4f}°")

# Find knee actuator
print("\n=== Actuator mapping ===")
for mid, aid in robot._actuator_index_by_motor_id.items():
    jid = int(robot.model.actuator_trnid[aid, 0])
    jname = mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
    kp = float(robot.model.actuator_gainprm[aid, 0])
    kv = -float(robot.model.actuator_biasprm[aid, 2])
    print(f"  motor_id={mid} -> actuator[{aid}] joint='{jname}' kp={kp} kv={kv} ctrl={ctrl_mujoco[aid]:.4f}")

robot.stop()
