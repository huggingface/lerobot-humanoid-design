#!/usr/bin/env python3
"""Synchronous sim2sim walking test: no threads, lockstep policy+physics.

Matches MJLab training execution exactly:
  1. Reset sim (z=0.72, joints=0, quat=identity)
  2. For each RL step (50 Hz = every 4 sim steps at dt=0.005):
     a. Read obs from sim state
     b. Run policy
     c. Apply action to ctrl
     d. Step physics 4x (decimation)
  3. Check robot walks forward without falling.

Run with: micromamba run -n mujoco313 python test_sim2sim_sync.py
"""
import sys
import numpy as np

sys.path.insert(0, ".")

import mujoco
from pathlib import Path
from rl_agent import (
    RLAgent, _load_config, infer_agent_spec, PolicyWrapper,
    _extract_default_joint_pos_rad_from_cfg,
    SNAPSHOT_TO_POLICY_JOINT_IDX, POLICY_ACTION_KEYS,
)
from sim_robot import SimBipedalRobotController, DEFAULT_MJCF_PATH

POLICY_DIR = "policies/less_noise_high_gain_torque_obs"
CONFIG_PATH = f"{POLICY_DIR}/config.yaml"
POLICY_PATH = f"{POLICY_DIR}/policy.onnx"

DECIMATION = 4
SIM_DT = 0.005
RL_DT = DECIMATION * SIM_DT  # 0.02s = 50 Hz
DURATION_S = 10.0
N_RL_STEPS = int(DURATION_S / RL_DT)

def main():
    print("=== Synchronous Sim2Sim Walking Test ===\n")

    # Load config + policy
    cfg = _load_config(Path(CONFIG_PATH))
    spec = infer_agent_spec(cfg)
    print(f"Policy terms: {spec.policy_terms}")
    print(f"Action scales: {spec.action_scales_rad}")
    print(f"Obs term scales: {spec.obs_term_scales}")
    print(f"Vel source: {spec.joint_vel_source}")

    q_ref = _extract_default_joint_pos_rad_from_cfg(cfg)
    print(f"Default joint pos (rad): {q_ref}")
    print()

    policy = PolicyWrapper.load(Path(POLICY_PATH), cfg)
    print(f"Policy input dim: {policy.expected_input_dim}")

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_MJCF_PATH))
    data = mujoco.MjData(model)
    model.opt.timestep = SIM_DT

    # Build joint ordering maps
    joint_names_snapshot_order = [
        "hipz_left", "hipx_left", "hipy_left", "knee_left", "ankley_left", "anklex_left",
        "hipz_right", "hipx_right", "hipy_right", "knee_right", "ankley_right", "anklex_right",
    ]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names_snapshot_order]
    assert all(jid >= 0 for jid in joint_ids), f"Missing joints: {[n for n, j in zip(joint_names_snapshot_order, joint_ids) if j < 0]}"
    qpos_adr = [int(model.jnt_qposadr[jid]) for jid in joint_ids]
    dof_adr = [int(model.jnt_dofadr[jid]) for jid in joint_ids]

    # Build actuator map: actuator name "m_<joint>" -> actuator index
    actuator_name_to_idx = {}
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name:
            actuator_name_to_idx[name] = i

    # Map: joint_names_snapshot_order[j] -> actuator index
    act_idx_by_joint = []
    for jn in joint_names_snapshot_order:
        aname = f"m_{jn}"
        assert aname in actuator_name_to_idx, f"Actuator {aname} not found"
        act_idx_by_joint.append(actuator_name_to_idx[aname])

    # IMU sensor
    imu_gyro_adr = None
    for i in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        if name == "imu_ang_vel":
            imu_gyro_adr = model.sensor_adr[i]
            break

    # Reset sim: z=0.72, all joints=0, identity quaternion
    def reset_sim():
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        data.qfrc_applied[:] = 0.0
        data.xfrc_applied[:] = 0.0
        if data.act.size:
            data.act[:] = 0.0
        data.qacc_warmstart[:] = 0.0
        # Free joint: [x, y, z, qw, qx, qy, qz]
        data.qpos[2] = 0.72
        data.qpos[3] = 1.0  # qw=1 (identity)
        mujoco.mj_forward(model, data)

    # Read joint positions in snapshot order (deg)
    def read_joint_pos_deg():
        return np.array([np.rad2deg(data.qpos[a]) for a in qpos_adr])

    # Read joint velocities in snapshot order (rad/s)
    def read_joint_vel_rads():
        return np.array([data.qvel[a] for a in dof_adr])

    # Read actuator torques in snapshot order (Nm)
    def read_joint_torques():
        return np.array([data.actuator_force[act_idx_by_joint[j]] for j in range(12)])

    # Read IMU gyro (body-frame angular velocity)
    def read_gyro_rads():
        if imu_gyro_adr is not None:
            return data.sensordata[imu_gyro_adr:imu_gyro_adr+3].copy()
        # Fallback: qvel[3:6] for free joint = body-frame angular velocity
        return data.qvel[3:6].copy()

    # Read base orientation quaternion (xyzw from MuJoCo wxyz)
    def read_quat_xyzw():
        qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
        return [float(qx), float(qy), float(qz), float(qw)]

    # Projected gravity from quaternion
    def projected_gravity(quat_xyzw):
        x, y, z, w = quat_xyzw
        # Rotation matrix from quaternion (scipy convention: xyzw)
        r00 = 1 - 2*(y*y + z*z)
        r01 = 2*(x*y - z*w)
        r02 = 2*(x*z + y*w)
        r10 = 2*(x*y + z*w)
        r11 = 1 - 2*(x*x + z*z)
        r12 = 2*(y*z - x*w)
        r20 = 2*(x*z - y*w)
        r21 = 2*(y*z + x*w)
        r22 = 1 - 2*(x*x + y*y)
        # R^T @ [0, 0, -1]
        return np.array([-r02, -r12, -r22], dtype=np.float32)

    # Create the RLAgent using the robot API (for building obs via standard pipeline)
    # But we'll also do it manually for cross-checking.
    # Let's use the agent's obs building to keep everything consistent.
    robot = SimBipedalRobotController(
        control_hz=200.0,
        sim_dt=SIM_DT,
        initial_height_m=0.72,
        hardcode_mjlab_spawn=False,
        use_lerobot_reference=False,
        auto_reset_on_flip=False,  # we handle this ourselves
    )

    agent = RLAgent.from_files(
        robot=robot,
        config_path=CONFIG_PATH,
        policy_path=POLICY_PATH,
    )
    agent.spec.joint_vel_source = "snapshot"
    agent.set_command_twist(lin_x=0.3, lin_y=0.0, yaw_rate=0.0)

    # Initialize agent state (normally done in start())
    agent.obs_history.clear()
    agent._prev_q_rad = None
    agent._prev_q_t_s = None
    agent._prev_obs_joint_pos = np.zeros(12, dtype=np.float32)
    agent._curr_obs_joint_pos = None
    agent._prev_obs_joint_vel = np.zeros(12, dtype=np.float32)
    agent._curr_obs_joint_vel = None
    agent._prev_obs_joint_torque = np.zeros(12, dtype=np.float32)
    agent._curr_obs_joint_torque = None
    agent._last_policy_action = np.zeros(len(spec.action_keys), dtype=np.float32)

    # Reset the robot's sim to match our init
    reset_sim()
    # Also reset the robot's internal sim to the same state
    with robot._sim_lock:
        robot.data.qpos[:] = data.qpos.copy()
        robot.data.qvel[:] = data.qvel.copy()
        robot.data.ctrl[:] = 0.0
        robot.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(robot.model, robot.data)
        robot._sync_state_from_sim(0.0)

    print(f"\nRunning {N_RL_STEPS} RL steps ({DURATION_S}s at {1/RL_DT:.0f} Hz)...")
    print(f"Initial base pos: x={data.qpos[0]:.4f}, y={data.qpos[1]:.4f}, z={data.qpos[2]:.4f}")

    fell = False
    fell_step = -1
    positions = []

    for step in range(N_RL_STEPS):
        sim_time = step * RL_DT

        # 1. Sync robot's internal state from our sim data
        with robot._sim_lock:
            robot.data.qpos[:] = data.qpos.copy()
            robot.data.qvel[:] = data.qvel.copy()
            # Copy actuator forces too for torque observation
            robot.data.actuator_force[:] = data.actuator_force.copy()
            mujoco.mj_forward(robot.model, robot.data)
            robot._sync_state_from_sim(sim_time)

        # 2. Build obs via agent's standard pipeline
        snapshot = robot.get_combined_state_snapshot(include_joint_state=True)
        # Override time_s with sim time for consistent finite-diff
        snapshot["time_s"] = sim_time

        obs_now = agent._build_obs_now(snapshot)
        obs_hist = agent._build_history_obs(obs_now)
        obs_in = agent._adapt_obs_dim_for_policy(obs_hist)

        # 3. Run policy
        action = agent.policy.infer(obs_in)

        # 4. Apply action via agent pipeline (updates agent._last_policy_action)
        #    But we need to intercept the ctrl values for our sim.
        #    Use _apply_action which calls robot.set_action, then read robot's action state.
        agent._last_policy_action[:] = 0.0
        agent._last_policy_action[:min(len(spec.action_keys), action.size)] = action[:len(spec.action_keys)]
        agent._apply_action(action)

        # 5. Read the ctrl values that set_action computed and apply to our sim
        #    The robot's _apply_position_actuator_ctrl will have set data.ctrl,
        #    but since robot isn't running its loop, we need to do it manually.
        with robot._sim_lock:
            robot._apply_position_actuator_ctrl()
            ctrl = robot.data.ctrl.copy()

        data.ctrl[:] = ctrl

        # 6. Step physics (decimation=4)
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)

        # 7. Record and check
        bx, by, bz = float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2])
        positions.append((sim_time, bx, by, bz))

        # Check for fall (base too low or flipped)
        quat_xyzw = read_quat_xyzw()
        pg = projected_gravity(quat_xyzw)
        tilt = np.arccos(np.clip(-pg[2], -1, 1))  # angle from upright

        if step % 50 == 0:
            print(f"  step={step:4d} t={sim_time:5.2f}s  x={bx:+.3f} y={by:+.3f} z={bz:.3f}  tilt={np.rad2deg(tilt):.1f}°  action_norm={np.linalg.norm(action):.3f}")

        if bz < 0.3 or tilt > np.deg2rad(70):
            fell = True
            fell_step = step
            print(f"  FELL at step={step} t={sim_time:.2f}s  z={bz:.3f}  tilt={np.rad2deg(tilt):.1f}°")
            break

        # Check for non-finite
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            fell = True
            fell_step = step
            print(f"  NON-FINITE at step={step}")
            break

    # Results
    positions = np.array(positions)
    print(f"\n=== Results ===")
    if fell:
        print(f"Robot FELL at step {fell_step} ({fell_step * RL_DT:.2f}s)")
    else:
        dx = positions[-1, 1] - positions[0, 1]
        dy = positions[-1, 2] - positions[0, 2]
        fz = positions[-1, 3]
        print(f"Survived {DURATION_S:.0f}s!")
        print(f"Displacement: dx={dx:.3f}m, dy={dy:.3f}m")
        print(f"Final height: z={fz:.3f}m")
        print(f"Avg forward speed: {dx/DURATION_S:.3f} m/s (commanded: 0.3)")
        if dx > 0.5:
            print("\n=== WALKING SUCCESS ===")
        elif abs(dx) < 0.1:
            print("\n=== STANDING (not walking) ===")
        else:
            print(f"\n=== PARTIAL (dx={dx:.2f}m) ===")

if __name__ == "__main__":
    main()
