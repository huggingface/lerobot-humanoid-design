#!/usr/bin/env python3
"""Sim2sim walking test: run RL policy on sim_robot, verify the robot walks.

Run with: micromamba run -n mujoco313 python test_sim2sim_walk.py
"""
import sys
import time
import numpy as np

sys.path.insert(0, ".")

from sim_robot import SimBipedalRobotController
from rl_agent import RLAgent

POLICY_DIR = "policies/less_noise_high_gain_torque_obs"
CONFIG_PATH = f"{POLICY_DIR}/config.yaml"
POLICY_PATH = f"{POLICY_DIR}/policy.onnx"

def main():
    print("=== Sim2Sim Walking Test ===\n")

    # 1. Verify obs term ordering fix
    from pathlib import Path
    from rl_agent import _load_config, infer_agent_spec
    cfg = _load_config(Path(CONFIG_PATH))
    spec = infer_agent_spec(cfg)
    print(f"Policy terms (should match training order):")
    print(f"  {spec.policy_terms}")
    expected = ["actions", "base_ang_vel", "command", "joint_pos",
                "joint_torques", "joint_vel", "projected_gravity"]
    assert spec.policy_terms == expected, f"WRONG ORDER!\n  got: {spec.policy_terms}\n  expected: {expected}"
    print(f"  => CORRECT\n")

    print(f"Action scales: {dict(zip(spec.action_keys, spec.action_scales_rad))}")
    print(f"Obs term scales: {spec.obs_term_scales}")
    print(f"Velocity source: {spec.joint_vel_source}")
    print(f"Inference Hz: {spec.inference_hz}")
    print(f"History len: {spec.history_len}")
    print()

    # 2. Create sim robot — match training init exactly:
    #    z=0.72m, all joints=0, identity quaternion, no knees-bent reference
    robot = SimBipedalRobotController(
        control_hz=200.0,
        sim_dt=0.005,
        initial_height_m=0.72,
        hardcode_mjlab_spawn=False,
        use_lerobot_reference=False,  # straight legs, not knees-bent
        auto_reset_on_flip=True,
    )

    # 3. Create RL agent BEFORE starting sim to minimize uncontrolled steps.
    #    MJLab runs the policy before the first physics step; we want the same.
    agent = RLAgent.from_files(
        robot=robot,
        config_path=CONFIG_PATH,
        policy_path=POLICY_PATH,
        log_observation=True,
        log_action=True,
        log_path="sim2sim_debug_log.csv",
    )

    # Override velocity source to "snapshot" for sim2sim fidelity
    agent.spec.joint_vel_source = "snapshot"

    # Set forward walking command
    agent.set_command_twist(lin_x=0.3, lin_y=0.0, yaw_rate=0.0)

    # Start sim and agent back-to-back (no settle time)
    robot.start(mode="control")

    print("Starting RL agent (vx=0.3 m/s)...")
    agent.start()

    # 4. Monitor for 10 seconds
    t0 = time.time()
    duration = 10.0
    resets = robot.get_reset_counter()
    positions = []
    step_count = 0

    while time.time() - t0 < duration:
        snap = robot.get_combined_state_snapshot(include_joint_state=True)
        if snap.get("sim_step_count", 0) > step_count:
            step_count = snap["sim_step_count"]

        # Track base position (approximate from IMU or sim data)
        # The sim has a free-floating base; we can read qpos directly
        with robot._sim_lock:
            base_x = float(robot.data.qpos[0])
            base_y = float(robot.data.qpos[1])
            base_z = float(robot.data.qpos[2])

        positions.append((time.time() - t0, base_x, base_y, base_z))
        new_resets = robot.get_reset_counter()
        if new_resets > resets:
            print(f"  [t={time.time()-t0:.1f}s] RESET #{new_resets} (robot fell)")
            resets = new_resets

        time.sleep(0.2)

    agent.stop()
    time.sleep(0.1)
    robot.stop()

    # 5. Analyze results
    positions = np.array(positions)
    final_resets = robot.get_reset_counter()

    print(f"\n=== Results ===")
    print(f"Duration: {duration:.0f}s")
    print(f"Resets (falls): {final_resets}")
    print(f"Sim steps: {step_count}")

    if len(positions) > 1:
        dx = positions[-1, 1] - positions[0, 1]
        dy = positions[-1, 2] - positions[0, 2]
        final_z = positions[-1, 3]
        print(f"Base displacement: dx={dx:.3f}m, dy={dy:.3f}m")
        print(f"Final base height: z={final_z:.3f}m")
        print(f"Average forward speed: {dx/duration:.3f} m/s (commanded: 0.3)")

        # Check if robot walked forward
        if dx > 0.5 and final_resets == 0:
            print("\n=== WALKING SUCCESS ===")
        elif final_resets == 0 and abs(dx) < 0.1:
            print("\n=== STANDING (not walking) === — policy may need different command")
        elif final_resets > 0:
            print(f"\n=== FELL {final_resets} TIMES === — inference pipeline likely has issues")
        else:
            print(f"\n=== PARTIAL (dx={dx:.2f}m, resets={final_resets}) ===")

    # 6. Check debug log
    print(f"\nDebug log written to: sim2sim_debug_log.csv")
    print(f"Agent debug state: {agent.get_debug_state()}")

if __name__ == "__main__":
    main()
