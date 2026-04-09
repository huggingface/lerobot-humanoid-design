#!/usr/bin/env python3
"""Test with floor contact disabled to check if that's the root cause."""
import sys
import os
sys.path.insert(0, '/home/virgile/devel/lerobot-humanoid-design/to_real_robot')
os.chdir('/home/virgile/devel/lerobot-humanoid-design/to_real_robot')

import time
import numpy as np
import mujoco

from sim_robot import SimBipedalRobotController

print("Creating SimBipedalRobotController with fixed_base=True...")
robot = SimBipedalRobotController(control_hz=200.0, fixed_base=True)

# Disable floor contact before starting the control loop
# Find floor geom and disable it
floor_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
print(f"Floor geom id: {floor_id}")
if floor_id >= 0:
    robot.model.geom_contype[floor_id] = 0
    robot.model.geom_conaffinity[floor_id] = 0
    print("Floor contact disabled!")

robot.start(mode="control", auto_enable=True)

print("Started. Waiting 1s for initial settle...")
time.sleep(1.0)

s0 = robot.get_combined_state_snapshot()
print("Initial snapshot (no floor contact, before any set_action):")
print("  joint_torque_nm:", [round(x,3) for x in s0['joint_torque_nm']])
print("  joint_state_deg:", [round(x,2) for x in s0['joint_state_deg']])

print("\nSending standing pose...")
robot.set_action(
    left={"hipz": 0.0, "hipx": 0.0, "hipy": -20.0535, "knee": 40.107, "ankle_pitch": -20.0535, "ankle_roll": 0.0},
    right={"hipz": 0.0, "hipx": 0.0, "hipy": 20.0535, "knee": 40.107, "ankle_pitch": 20.0535, "ankle_roll": 0.0},
)
time.sleep(2.0)

s1 = robot.get_combined_state_snapshot()
print("After standing pose settle:")
print("  joint_torque_nm:", [round(x,3) for x in s1['joint_torque_nm']])
print("  joint_state_deg:", [round(x,2) for x in s1['joint_state_deg']])

print("\nSending test action...")
robot.set_action(
    left={"hipz": 10.0, "hipx": 10.0, "hipy": 20.0, "knee": 40.0, "ankle_pitch": 10.0, "ankle_roll": 5.0},
    right={"hipz": 10.0, "hipx": 10.0, "hipy": -20.0, "knee": 40.0, "ankle_pitch": 10.0, "ankle_roll": 5.0},
)

print("Waiting for equilibrium...")
for i in range(200):
    s = robot.get_combined_state_snapshot()
    max_vel = max(abs(v) for v in s['joint_velocity_deg_s'])
    if max_vel < 0.5:
        print(f"  Converged at iteration {i} (max_vel={max_vel:.4f})")
        break
    time.sleep(0.05)
else:
    print(f"  Did not converge (max_vel={max_vel:.4f})")

print("\nFINAL SNAPSHOT (no floor contact):")
print("joint_torque_nm:", [round(x,3) for x in s['joint_torque_nm']])
print("joint_velocity_deg_s:", [round(x,3) for x in s['joint_velocity_deg_s']])
print("joint_state_deg:", [round(x,2) for x in s['joint_state_deg']])

robot.stop()
print("Done.")
