#!/usr/bin/env python3
"""Hanging test: compare sim_robot against real robot reference data.

Run with: micromamba run -n mujoco313 python test_hanging.py
"""
import sys
import time
sys.path.insert(0, ".")

from sim_robot import SimBipedalRobotController

# Real robot reference: set_action return values (raw motor deg)
REAL_RAW_TARGETS = {
    1: 237.32, 2: 350.606, 3: 291.904, 4: 17.352, 5: 5.0, 6: -15.0,
    7: 142.68, 8: 29.394, 9: 68.096, 10: 342.648, 11: 5.0, 12: -15.0,
}

# Real robot reference: converged joint states (degrees)
REAL_JOINT_DEG = [
    10.0, 10.4, 17.6, 39.9, 9.5, 5.4,      # left
    8.9,  9.4,  -17.8, 39.0, 10.8, 5.1,     # right
]

# Real robot reference: converged joint torques (Nm) — small since hanging
REAL_JOINT_TORQUE = [
    -0.065, -0.435, 2.505, 0.220, 0.204, -0.142,
    0.509, 0.454, -2.330, 0.923, -0.320, -0.048,
]

def main():
    print("=== Sim Robot Hanging Test ===\n")

    robot = SimBipedalRobotController(
        fixed_base=True,
        fixed_base_height_m=1.5,   # well above ground
        control_hz=200.0,
        sim_dt=0.005,
        hardcode_mjlab_spawn=False,
    )
    robot.start(mode="control")
    time.sleep(0.5)  # let it settle

    # Send same action as real robot
    raw_ret = robot.set_action(
        left={"hipz": 10.0, "hipx": 10.0, "hipy": 20.0, "knee": 40.0,
              "ankle_pitch": 10.0, "ankle_roll": 5.0},
        right={"hipz": 10.0, "hipx": 10.0, "hipy": -20.0, "knee": 40.0,
               "ankle_pitch": 10.0, "ankle_roll": 5.0},
    )

    # Wait for PD controller to converge
    time.sleep(2.0)

    snap = robot.get_combined_state_snapshot(include_joint_state=True)
    robot.stop()

    # 1. Compare set_action return values
    print("1. set_action() raw motor targets (sim vs real):")
    print(f"   {'Motor':>6} {'Sim':>10} {'Real':>10} {'Diff':>10} {'360-wrap':>10}")
    all_raw_ok = True
    for mid in sorted(raw_ret.keys()):
        sim_val = raw_ret[mid]
        real_val = REAL_RAW_TARGETS[mid]
        diff = sim_val - real_val
        # Account for 360-degree wraps
        wrapped_diff = diff % 360
        if wrapped_diff > 180:
            wrapped_diff -= 360
        ok = abs(wrapped_diff) < 0.01
        if not ok:
            all_raw_ok = False
        print(f"   m{mid:>4}: {sim_val:>10.3f} {real_val:>10.3f} {diff:>10.3f} {wrapped_diff:>10.3f} {'OK' if ok else 'FAIL'}")
    print(f"   => {'PASS' if all_raw_ok else 'FAIL'}\n")

    # 2. Compare converged joint states
    sim_joint_deg = snap.get("joint_state_deg", [])
    print("2. Converged joint states (degrees, sim vs real):")
    joint_names = [
        "L.hipz", "L.hipx", "L.hipy", "L.knee", "L.ankpitch", "L.ankroll",
        "R.hipz", "R.hipx", "R.hipy", "R.knee", "R.ankpitch", "R.ankroll",
    ]
    commanded = [10.0, 10.0, 20.0, 40.0, 10.0, 5.0,
                 10.0, 10.0, -20.0, 40.0, 10.0, 5.0]
    all_joint_ok = True
    for i, name in enumerate(joint_names):
        sim_v = sim_joint_deg[i] if i < len(sim_joint_deg) else float('nan')
        real_v = REAL_JOINT_DEG[i]
        cmd_v = commanded[i]
        err_sim = sim_v - cmd_v
        err_real = real_v - cmd_v
        ok = abs(err_sim) < 2.0  # sim should converge within 2 deg of command
        if not ok:
            all_joint_ok = False
        print(f"   {name:>12}: cmd={cmd_v:>7.1f}  sim={sim_v:>7.2f} (err={err_sim:>+6.2f})  "
              f"real={real_v:>7.1f} (err={err_real:>+6.1f})  {'OK' if ok else 'FAIL'}")
    print(f"   => {'PASS' if all_joint_ok else 'FAIL'}\n")

    # 3. Compare torques (should be small, same sign pattern)
    sim_torque = snap.get("joint_torque_nm", [])
    print("3. Joint torques (Nm, sim vs real):")
    for i, name in enumerate(joint_names):
        sim_t = sim_torque[i] if i < len(sim_torque) else float('nan')
        real_t = REAL_JOINT_TORQUE[i]
        print(f"   {name:>12}: sim={sim_t:>+8.3f}  real={real_t:>+8.3f}")
    print(f"   (Torque magnitudes differ due to different dynamics params; signs should roughly match)\n")

    # 4. Check IMU
    imu = snap.get("imu", {})
    print("4. IMU state:")
    print(f"   available: {imu.get('available')}")
    print(f"   quat_xyzw: {snap.get('orientation_quaternion_xyzw')}")
    print(f"   gyro: {imu.get('gyro_rads')}")
    print(f"   lin_vel: {imu.get('linear_velocity_mps')}")
    print()

    if all_raw_ok and all_joint_ok:
        print("=== ALL CHECKS PASSED ===")
    else:
        print("=== SOME CHECKS FAILED ===")

if __name__ == "__main__":
    main()
