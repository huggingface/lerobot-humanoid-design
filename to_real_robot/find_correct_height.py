#!/usr/bin/env python3
"""Find the correct fixed_base_height_m using binary search."""
import sys
import os
sys.path.insert(0, '/home/virgile/devel/lerobot-humanoid-design/to_real_robot')
os.chdir('/home/virgile/devel/lerobot-humanoid-design/to_real_robot')

import numpy as np
import mujoco
from pathlib import Path

MJCF_PATH = Path("/home/virgile/devel/lerobot_legged_robots/models/bipedal_plateform_no_arms_high_gain/mjcf/sim_scene.xml")

model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
data = mujoco.MjData(model)

from sim_robot import SIM_KNEES_BENT_POSE_RAD

def setup_pose(height_m):
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[2] = height_m
    data.qpos[3] = 1.0
    for jn, val in SIM_KNEES_BENT_POSE_RAD.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid >= 0:
            qadr = int(model.jnt_qposadr[jid])
            data.qpos[qadr] = float(val)
    mujoco.mj_forward(model, data)

def get_min_contact_dist(height_m):
    """Returns minimum contact penetration depth (negative means penetrating)."""
    setup_pose(height_m)
    # Check penetration via contact
    mujoco.mj_collision(model, data)
    if data.ncon == 0:
        return 0.0  # No contact, feet are above floor
    min_dist = min(data.contact[i].dist for i in range(data.ncon))
    return min_dist

def get_foot_geom_bottom_z(height_m):
    """Get the z-position of foot collision geom."""
    setup_pose(height_m)
    # Find foot collision geoms
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision")
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision")
    if right_id < 0:
        print("right_foot_collision not found!")
        return None
    # Get geom center position and rotation
    right_pos = data.geom_xpos[right_id]
    right_mat = data.geom_xmat[right_id].reshape(3,3)
    right_size = model.geom_size[right_id]
    geom_type = int(model.geom_type[right_id])
    print(f"  height={height_m:.4f}: right_foot_collision center z={right_pos[2]:.4f}m")
    print(f"    geom_type={geom_type} (7=capsule), size={right_size}")
    print(f"    rotation matrix (z-col)={right_mat[:, 2]}")
    # Capsule: bottom = center + R * [0, 0, -(half_length+radius)]  or similar
    # For z-rotation, the bottom of the capsule in world frame:
    # Actually for capsule in MuJoCo: the capsule axis is the local z-axis
    # The bounding box bottom = center_z - max_radius (if horizontal)
    # Capsule size: [radius, half-cylinder-length] for capsule
    radius = float(right_size[0])
    half_len = float(right_size[1])
    # Local Z-axis direction in world frame
    local_z = right_mat[:, 2]
    print(f"    local_z world = {local_z}")
    # The two end caps are at +/-half_len along local z
    p1 = right_pos + local_z * half_len
    p2 = right_pos - local_z * half_len
    bottom_z = min(p1[2], p2[2]) - radius
    print(f"    p1_z={p1[2]:.4f}, p2_z={p2[2]:.4f}, bottom_z={bottom_z:.4f}m")
    return bottom_z

# First check what the foot's actual bottom z is at height=0.77
print("=== Analysis at height=0.77 ===")
bot = get_foot_geom_bottom_z(0.77)

# The floor is at z=0. We need bottom_z = 0 (touching floor, not penetrating)
# So correct height = 0.77 + |bottom_z| if bottom_z < 0
if bot is not None and bot < 0:
    correct_height = 0.77 - bot  # add the penetration depth
    print(f"\nBottom of foot at z={bot:.4f}m, penetrating by {-bot:.4f}m")
    print(f"Need to raise torso by {-bot:.4f}m")
    print(f"Correct fixed_base_height_m = {correct_height:.4f}m")

    # Verify
    print(f"\n=== Verification at height={correct_height:.4f} ===")
    get_foot_geom_bottom_z(correct_height)

# Binary search for height where contact dist = 0
print("\n=== Binary search for zero-contact height ===")
lo, hi = 0.77, 1.0
for _ in range(50):
    mid = (lo + hi) / 2
    d = get_min_contact_dist(mid)
    if d < -1e-6:  # Penetrating
        lo = mid
    else:
        hi = mid
print(f"Zero-contact height: {hi:.5f}m (contact_dist={get_min_contact_dist(hi):.6f})")

# Check contact at 0.77 and proposed fix
print(f"\nContact dist at 0.77m: {get_min_contact_dist(0.77):.6f}")
proposed = 0.77 - get_min_contact_dist(0.77)  # lift by penetration depth
print(f"Proposed height (0.77 - dist): {proposed:.4f}m")
print(f"Contact dist at {proposed:.4f}m: {get_min_contact_dist(proposed):.6f}")
