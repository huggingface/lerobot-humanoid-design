#!/usr/bin/env python3
"""Minimal sim2sim: build obs directly from MuJoCo data, no sim_robot.

Matches MJLab execution exactly:
  - Obs built from raw MuJoCo arrays, same as training obs functions
  - Actions applied directly to data.ctrl
  - One-step delay on joint_pos, joint_vel, joint_torques
  - All ordering matches training (joint order = XML definition order)

Run with: micromamba run -n mujoco313 python test_sim2sim_minimal.py
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, ".")

import mujoco
from rl_agent import _load_config, infer_agent_spec, PolicyWrapper, _extract_default_joint_pos_rad_from_cfg

POLICY_DIR = "policies/less_noise_high_gain_torque_obs"
CONFIG_PATH = f"{POLICY_DIR}/config.yaml"
POLICY_PATH = f"{POLICY_DIR}/policy.onnx"
MJCF_PATH = str(Path(__file__).resolve().parent / "bipedal_plateform_no_arms" / "mjcf" / "sim_scene.xml")

DECIMATION = 4
SIM_DT = 0.005
RL_DT = DECIMATION * SIM_DT
DURATION_S = 10.0
N_RL_STEPS = int(DURATION_S / RL_DT)

def main():
    print("=== Minimal Sim2Sim (no sim_robot) ===\n")

    # Load config + policy
    cfg = _load_config(Path(CONFIG_PATH))
    spec = infer_agent_spec(cfg)
    policy = PolicyWrapper.load(Path(POLICY_PATH), cfg)
    q_default = _extract_default_joint_pos_rad_from_cfg(cfg)

    print(f"Policy terms: {spec.policy_terms}")
    print(f"Action scales: {[f'{s:.4f}' for s in spec.action_scales_rad]}")
    print(f"Obs term scales: {spec.obs_term_scales}")
    print(f"Default joint pos (rad): {q_default}")
    print(f"Policy input dim: {policy.expected_input_dim}")
    print()

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    model.opt.timestep = SIM_DT

    # Joint ordering: follows XML body tree definition
    # robot_training.xml defines: right(6) then left(6)
    joint_names_xml_order = [
        "hipz_right", "hipx_right", "hipy_right", "knee_right", "ankley_right", "anklex_right",
        "hipz_left", "hipx_left", "hipy_left", "knee_left", "ankley_left", "anklex_left",
    ]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names_xml_order]
    assert all(jid >= 0 for jid in joint_ids), f"Missing: {[n for n, j in zip(joint_names_xml_order, joint_ids) if j < 0]}"
    qpos_adr = np.array([model.jnt_qposadr[jid] for jid in joint_ids])
    dof_adr = np.array([model.jnt_dofadr[jid] for jid in joint_ids])

    # Actuator ordering: follows XML actuator definition
    # sim_scene.xml defines: left(6) then right(6)
    act_names_xml_order = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        act_names_xml_order.append(name)
    print(f"Actuator order (XML): {act_names_xml_order}")

    # Verify: which order does actuator_force come in?
    # We need to know if training uses actuator order or joint order for torques.
    # Training code: `return env.scene["robot"].data.actuator_force`
    # In IsaacLab, this maps to MuJoCo actuator_force in ACTUATOR ORDER.
    #
    # But for our sim_scene.xml, actuators are [left, right] while joints are [right, left].
    # So actuator_force[0:6] = left torques, actuator_force[6:12] = right torques.
    #
    # HOWEVER: In MJLab/IsaacLab, when the articulation is created from the MJCF,
    # the actuator indices may be remapped to joint order. Let me check both
    # and see which one makes the policy work.

    # Build actuator->joint mapping for reordering
    # actuator i acts on which joint?
    act_joint_map = []
    for i in range(model.nu):
        # actuator's transmission target joint
        trnid = model.actuator_trnid[i, 0]
        act_joint_map.append(trnid)
    print(f"Actuator -> joint ID map: {act_joint_map}")
    print(f"Joint IDs (xml order): {joint_ids}")

    # Map actuator_force to joint order [right(6), left(6)]
    # This tells us: for joint j in xml order, which actuator index has its force?
    joint_to_act_idx = {}
    for act_i, jid in enumerate(act_joint_map):
        joint_to_act_idx[jid] = act_i
    act_idx_for_joint_order = [joint_to_act_idx[jid] for jid in joint_ids]
    print(f"Actuator index for each joint (xml order): {act_idx_for_joint_order}")
    print()

    # IMU gyro sensor
    gyro_adr = None
    for i in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        if name == "imu_ang_vel":
            gyro_adr = model.sensor_adr[i]
            break

    # Reset sim
    def reset():
        mujoco.mj_resetData(model, data)
        data.qpos[2] = 0.72  # z height
        data.qpos[3] = 1.0   # qw (identity quaternion)
        mujoco.mj_forward(model, data)

    def read_joint_pos():
        """Joint positions in XML joint order [right(6), left(6)], radians."""
        return data.qpos[qpos_adr].copy()

    def read_joint_vel():
        """Joint velocities in XML joint order [right(6), left(6)], rad/s."""
        return data.qvel[dof_adr].copy()

    def read_actuator_force_joint_order():
        """Actuator forces remapped to XML joint order [right(6), left(6)]."""
        return data.actuator_force[act_idx_for_joint_order].copy()

    def read_actuator_force_actuator_order():
        """Actuator forces in XML actuator order [left(6), right(6)]."""
        return data.actuator_force[:12].copy()

    def read_qfrc_actuator():
        """Generalized actuator force at each DOF, in XML joint order."""
        return data.qfrc_actuator[dof_adr].copy()

    def read_gyro():
        if gyro_adr is not None:
            return data.sensordata[gyro_adr:gyro_adr+3].copy()
        return data.qvel[3:6].copy()

    def read_quat_xyzw():
        qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
        return np.array([qx, qy, qz, qw])

    def projected_gravity(q_xyzw):
        x, y, z, w = q_xyzw
        r02 = 2*(x*z + y*w)
        r12 = 2*(y*z - x*w)
        r22 = 1 - 2*(x*x + y*y)
        return np.array([-r02, -r12, -r22], dtype=np.float32)

    # Obs building: exactly match MJLab training order and semantics
    # Delayed buffers
    prev_joint_pos = np.zeros(12, dtype=np.float32)
    prev_joint_vel = np.zeros(12, dtype=np.float32)
    prev_joint_torque = np.zeros(12, dtype=np.float32)
    last_action = np.zeros(12, dtype=np.float32)
    command = np.array([0.3, 0.0, 0.0], dtype=np.float32)

    torque_scale = float(spec.obs_term_scales.get("joint_torques", 1.0))

    def build_obs():
        """Build 57-dim obs matching MJLab training exactly."""
        nonlocal prev_joint_pos, prev_joint_vel, prev_joint_torque

        # Current values (will become "prev" for next step)
        cur_joint_pos = (read_joint_pos() - q_default).astype(np.float32)
        cur_joint_vel = read_joint_vel().astype(np.float32)
        # Use actuator_force in actuator order (matching training's data.actuator_force)
        cur_joint_torque = (read_actuator_force_actuator_order() * torque_scale).astype(np.float32)

        # Obs uses DELAYED values (one-step delay)
        obs_parts = {
            "actions": last_action.copy(),
            "base_ang_vel": read_gyro().astype(np.float32),
            "command": command.copy(),
            "joint_pos": prev_joint_pos.copy(),
            "joint_torques": prev_joint_torque.copy(),
            "joint_vel": prev_joint_vel.copy(),
            "projected_gravity": projected_gravity(read_quat_xyzw()),
        }

        # Update delay buffers
        prev_joint_pos = cur_joint_pos.copy()
        prev_joint_vel = cur_joint_vel.copy()
        prev_joint_torque = (read_actuator_force_actuator_order() * torque_scale).astype(np.float32)

        # Concatenate in alphabetical (YAML) order = training order
        obs = np.concatenate([obs_parts[k] for k in sorted(obs_parts.keys())])
        return obs, obs_parts

    def apply_action(action):
        """Apply action to ctrl. Action is in joint order [right(6), left(6)]."""
        nonlocal last_action
        last_action = action.copy().astype(np.float32)

        # ctrl = default_pos + action * scale
        # But we need to map action (in JOINT order) to ctrl (in ACTUATOR order)
        target_rad = q_default.copy()
        for i in range(12):
            target_rad[i] += float(action[i]) * float(spec.action_scales_rad[i])

        # Map from joint order to actuator order
        for j_idx in range(12):
            act_idx = act_idx_for_joint_order[j_idx]
            data.ctrl[act_idx] = target_rad[j_idx]

    # Also try: torques in JOINT order instead of actuator order
    def build_obs_joint_order_torque():
        """Same as build_obs but torques in joint order."""
        nonlocal prev_joint_pos, prev_joint_vel, prev_joint_torque

        cur_joint_pos = (read_joint_pos() - q_default).astype(np.float32)
        cur_joint_vel = read_joint_vel().astype(np.float32)
        cur_joint_torque = (read_actuator_force_joint_order() * torque_scale).astype(np.float32)

        obs_parts = {
            "actions": last_action.copy(),
            "base_ang_vel": read_gyro().astype(np.float32),
            "command": command.copy(),
            "joint_pos": prev_joint_pos.copy(),
            "joint_torques": prev_joint_torque.copy(),
            "joint_vel": prev_joint_vel.copy(),
            "projected_gravity": projected_gravity(read_quat_xyzw()),
        }

        prev_joint_pos = cur_joint_pos.copy()
        prev_joint_vel = cur_joint_vel.copy()
        prev_joint_torque = cur_joint_torque.copy()

        obs = np.concatenate([obs_parts[k] for k in sorted(obs_parts.keys())])
        return obs, obs_parts

    # Run sim2sim with actuator-order torques
    print("=" * 60)
    print("Test A: torques in ACTUATOR order (training's data.actuator_force)")
    print("=" * 60)
    reset()
    prev_joint_pos = np.zeros(12, dtype=np.float32)
    prev_joint_vel = np.zeros(12, dtype=np.float32)
    prev_joint_torque = np.zeros(12, dtype=np.float32)
    last_action = np.zeros(12, dtype=np.float32)

    fell_a = False
    positions_a = []
    for step in range(N_RL_STEPS):
        t = step * RL_DT
        obs, parts = build_obs()
        action = policy.infer(obs)
        apply_action(action)

        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)

        bx, by, bz = float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2])
        positions_a.append((t, bx, by, bz))

        pg = projected_gravity(read_quat_xyzw())
        tilt = np.arccos(np.clip(-pg[2], -1, 1))

        if step < 5 or step % 50 == 0:
            print(f"  step={step:4d} t={t:5.2f}s  x={bx:+.3f} z={bz:.3f}  tilt={np.rad2deg(tilt):.1f}°  |a|={np.linalg.norm(action):.3f}")
            if step < 3:
                print(f"    obs={np.array2string(obs, precision=4, separator=', ', max_line_width=200)}")
                print(f"    action={np.array2string(action, precision=4, separator=', ', max_line_width=200)}")

        if bz < 0.3 or tilt > np.deg2rad(70):
            fell_a = True
            print(f"  FELL at step={step} t={t:.2f}s  z={bz:.3f}  tilt={np.rad2deg(tilt):.1f}°")
            break
        if not np.all(np.isfinite(data.qpos)):
            fell_a = True
            print(f"  NON-FINITE at step={step}")
            break

    positions_a = np.array(positions_a)
    if not fell_a:
        dx = positions_a[-1, 1] - positions_a[0, 1]
        print(f"  Survived! dx={dx:.3f}m, avg_vx={dx/DURATION_S:.3f} m/s")
    print()

    # Run sim2sim with joint-order torques
    print("=" * 60)
    print("Test B: torques in JOINT order (remapped to match joint_pos/vel)")
    print("=" * 60)
    reset()
    prev_joint_pos = np.zeros(12, dtype=np.float32)
    prev_joint_vel = np.zeros(12, dtype=np.float32)
    prev_joint_torque = np.zeros(12, dtype=np.float32)
    last_action = np.zeros(12, dtype=np.float32)

    fell_b = False
    positions_b = []
    for step in range(N_RL_STEPS):
        t = step * RL_DT
        obs, parts = build_obs_joint_order_torque()
        action = policy.infer(obs)
        apply_action(action)

        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)

        bx, by, bz = float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2])
        positions_b.append((t, bx, by, bz))

        pg = projected_gravity(read_quat_xyzw())
        tilt = np.arccos(np.clip(-pg[2], -1, 1))

        if step < 5 or step % 50 == 0:
            print(f"  step={step:4d} t={t:5.2f}s  x={bx:+.3f} z={bz:.3f}  tilt={np.rad2deg(tilt):.1f}°  |a|={np.linalg.norm(action):.3f}")
            if step < 3:
                print(f"    obs={np.array2string(obs, precision=4, separator=', ', max_line_width=200)}")
                print(f"    action={np.array2string(action, precision=4, separator=', ', max_line_width=200)}")

        if bz < 0.3 or tilt > np.deg2rad(70):
            fell_b = True
            print(f"  FELL at step={step} t={t:.2f}s  z={bz:.3f}  tilt={np.rad2deg(tilt):.1f}°")
            break
        if not np.all(np.isfinite(data.qpos)):
            fell_b = True
            print(f"  NON-FINITE at step={step}")
            break

    positions_b = np.array(positions_b)
    if not fell_b:
        dx = positions_b[-1, 1] - positions_b[0, 1]
        print(f"  Survived! dx={dx:.3f}m, avg_vx={dx/DURATION_S:.3f} m/s")

    # Summary
    print(f"\n{'='*60}")
    print(f"Test A (actuator-order torques): {'FELL' if fell_a else 'SURVIVED'}")
    print(f"Test B (joint-order torques):    {'FELL' if fell_b else 'SURVIVED'}")

if __name__ == "__main__":
    main()
