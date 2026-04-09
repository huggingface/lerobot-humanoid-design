#!/usr/bin/env python3
"""Investigate the correct fixed_base_height_m by checking foot positions."""
import sys
import os
sys.path.insert(0, '/home/virgile/devel/lerobot-humanoid-design/to_real_robot')
os.chdir('/home/virgile/devel/lerobot-humanoid-design/to_real_robot')

import numpy as np
import mujoco
from pathlib import Path

MJCF_PATH = Path("/home/virgile/devel/lerobot_legged_robots/models/bipedal_plateform_no_arms_high_gain/mjcf/sim_scene.xml")

# Load model
model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
data = mujoco.MjData(model)

from sim_robot import SIM_KNEES_BENT_POSE_RAD

# Reset to zero, then set the reference pose
data.qpos[:] = 0.0
data.qvel[:] = 0.0

# Set torso height
data.qpos[2] = 0.77  # fixed_base_height_m
data.qpos[3] = 1.0   # quaternion w

# Set joint positions to knees-bent pose
for jn, val in SIM_KNEES_BENT_POSE_RAD.items():
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
    if jid >= 0:
        qadr = int(model.jnt_qposadr[jid])
        data.qpos[qadr] = float(val)

mujoco.mj_forward(model, data)

# Find foot bodies / ankle bodies
print("=== Body positions ===")
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    if name and ('foot' in name.lower() or 'ankle' in name.lower() or 'torso' in name.lower()):
        pos = data.xpos[i]
        print(f"  {name}: z={pos[2]:.4f}m  (x={pos[0]:.4f}, y={pos[1]:.4f})")

print("\n=== Site positions ===")
for i in range(model.nsite):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
    if name:
        pos = data.site_xpos[i]
        print(f"  {name}: z={pos[2]:.4f}m  (x={pos[0]:.4f}, y={pos[1]:.4f})")

print("\n=== All geom positions (collision geoms near floor) ===")
for i in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    pos = data.geom_xpos[i]
    if pos[2] < 0.15:  # Near the floor
        contype = int(model.geom_contype[i])
        conaffinity = int(model.geom_conaffinity[i])
        print(f"  geom[{i}] '{name}': z={pos[2]:.4f}m contype={contype} conaffinity={conaffinity}")

print("\n=== Torso base position ===")
print(f"  torso z = {data.qpos[2]:.4f}m")

# Now check with the test action (knee=40 degrees)
print("\n\n=== With test action (knee=40 deg) ===")
data.qpos[:] = 0.0
data.qvel[:] = 0.0
data.qpos[2] = 0.77
data.qpos[3] = 1.0

# Set knee to 40 degrees explicitly
test_joints = {
    "hipz_left": 0.0,
    "hipx_left": 0.0,
    "hipy_left": np.deg2rad(-20.0535),
    "knee_left": np.deg2rad(40.107),
    "ankley_left": np.deg2rad(-20.0535),
    "anklex_left": 0.0,
    "hipz_right": 0.0,
    "hipx_right": 0.0,
    "hipy_right": np.deg2rad(20.0535),
    "knee_right": np.deg2rad(40.107),
    "ankley_right": np.deg2rad(20.0535),
    "anklex_right": 0.0,
}
for jn, val in test_joints.items():
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
    if jid >= 0:
        qadr = int(model.jnt_qposadr[jid])
        data.qpos[qadr] = float(val)

mujoco.mj_forward(model, data)

# Check foot geom positions
min_z = 1e9
print("All geom near floor (z<0.2) with test pose:")
for i in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    pos = data.geom_xpos[i]
    contype = int(model.geom_contype[i])
    if pos[2] < 0.2 and contype > 0:
        size = model.geom_size[i]
        geom_type = int(model.geom_type[i])
        print(f"  geom[{i}] '{name}': z={pos[2]:.4f}m type={geom_type} size={size} contype={contype}")
        if pos[2] < min_z:
            min_z = pos[2]

print(f"\nMinimum collision geom z-pos = {min_z:.4f}m")
print(f"For feet to touch floor at z=0, need to lower torso by {min_z:.4f}m")
print(f"Correct fixed_base_height_m = 0.77 - {min_z:.4f} = {0.77 - min_z:.4f}m")

# Check contact forces
print("\n=== Contacts with current pose ===")
mujoco.mj_step(model, data)
print(f"ncon = {data.ncon}")
for i in range(data.ncon):
    c = data.contact[i]
    print(f"  contact[{i}]: geom1={c.geom1} geom2={c.geom2} dist={c.dist:.4f}")
