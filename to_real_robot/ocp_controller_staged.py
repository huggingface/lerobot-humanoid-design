#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict

import numpy as np

from bipedal_robot import BipedalRobotController, MotorCommand
from mock_bus import MockBus
from ocp_follower import OCPTrajectory, OCPFollower
from root_constant import MOTOR_IDS


def _swap_hx_hy_order(x: np.ndarray) -> np.ndarray:
    # OCP order: [hipz, hipy, hipx, knee, anklex, ankley] x 2
    # Controller order: [hipz, hipx, hipy, knee, ankle_pitch, ankle_roll] x 2
    idx = [0, 2, 1, 3, 4, 5, 6, 8, 7, 9, 10, 11]
    return x[:, idx]


def load_ocp_npy(path: Path, *, dt_s: float) -> OCPTrajectory:
    arr = np.load(path, allow_pickle=True)
    if not (isinstance(arr, np.ndarray) and arr.shape == () and arr.dtype == object):
        raise RuntimeError(f"Unsupported npy format in {path}")

    blob = arr.item()
    if not isinstance(blob, dict) or "xs" not in blob or "us" not in blob:
        raise RuntimeError(f"Missing xs/us in {path}")

    xs = np.asarray(blob["xs"], dtype=float)
    us = np.asarray(blob["us"], dtype=float)
    if xs.ndim != 2 or us.ndim != 2 or xs.shape[1] < 37 or us.shape[1] < 12:
        raise RuntimeError(f"Unexpected trajectory dimensions: xs={xs.shape}, us={us.shape}")

    nq = xs.shape[1] - 18
    q_act = _swap_hx_hy_order(np.asarray(xs[:, (nq - 12):nq], dtype=float))
    v_act = _swap_hx_hy_order(np.asarray(xs[:, -12:], dtype=float))
    tau = _swap_hx_hy_order(np.asarray(us[:, :12], dtype=float))

    q_deg = np.rad2deg(q_act)
    qd_deg_s = np.rad2deg(v_act)

    n = xs.shape[0]
    if tau.shape[0] == n - 1:
        tau = np.vstack([tau, tau[-1]])
    elif tau.shape[0] != n:
        raise RuntimeError(f"Control horizon mismatch: xs rows={n}, us rows={tau.shape[0]}")

    t_s = np.arange(n, dtype=float) * float(dt_s)
    return OCPTrajectory(t_s=t_s, q_deg=q_deg, qd_deg_s=qd_deg_s, tau_ff_nm=tau)


def _has_valid_state(robot: BipedalRobotController) -> bool:
    snap = robot.get_state_snapshot()
    return any(float(snap[mid].stamp) > 0.0 for mid in MOTOR_IDS)


def _current_joint_state_deg(robot: BipedalRobotController, fallback_q_deg: np.ndarray) -> np.ndarray:
    if not _has_valid_state(robot):
        return np.asarray(fallback_q_deg, dtype=float).copy()
    snap = robot.get_state_snapshot()
    raw = {mid: float(snap[mid].position_deg) for mid in MOTOR_IDS}
    return np.asarray(robot.motor_state_to_joint_state(raw, output_radians=False, nq=12), dtype=float)


def go_to_pose(
    robot: BipedalRobotController,
    q_target_deg: np.ndarray,
    *,
    duration_s: float,
    hold_torque_nm: float = 0.0,
) -> None:
    q_target = np.asarray(q_target_deg, dtype=float).reshape(12)
    q_start = _current_joint_state_deg(robot, q_target)

    steps = max(1, int(float(duration_s) * max(1.0, float(robot.control_hz))))
    period = 1.0 / max(1.0, float(robot.control_hz))

    for i in range(steps):
        alpha = float(i + 1) / float(steps)
        q_cmd = (1.0 - alpha) * q_start + alpha * q_target
        pos_raw = robot.joint_state_to_motor_state(q_cmd, input_radians=False, output_space="raw")

        with robot._action_lock:
            for mid in MOTOR_IDS:
                prev = robot.action[mid]
                robot.action[mid] = MotorCommand(
                    position_deg=float(pos_raw[mid]),
                    velocity_deg_s=0.0,
                    torque_nm=float(hold_torque_nm),
                    kp=prev.kp,
                    kd=prev.kd,
                )
        time.sleep(period)


def run_staged_controller(
    robot: BipedalRobotController,
    traj: OCPTrajectory,
    *,
    go_to_init_s: float,
    squat_s: float,
    loop_count: int,
    wait_user: bool,
) -> None:
    q0 = np.asarray(traj.q_deg[0], dtype=float)

    # Use largest joint-space deviation from q0 as squat pose candidate.
    dev = np.linalg.norm(np.asarray(traj.q_deg, dtype=float) - q0[None, :], axis=1)
    i_squat = int(np.argmax(dev))
    q_squat = np.asarray(traj.q_deg[i_squat], dtype=float)

    print(f"[stage] go_to_initial_pose: {go_to_init_s:.2f}s")
    go_to_pose(robot, q0, duration_s=float(go_to_init_s))

    if wait_user:
        input("[stage] ready at initial pose. Press Enter to continue...")

    print(f"[stage] squat motion: down/up total {squat_s:.2f}s (idx={i_squat})")
    go_to_pose(robot, q_squat, duration_s=float(squat_s) * 0.5)
    go_to_pose(robot, q0, duration_s=float(squat_s) * 0.5)

    follower = OCPFollower(robot)
    follower.set_trajectory(traj)
    for i in range(max(1, int(loop_count))):
        print(f"[stage] replay loop {i + 1}/{loop_count}")
        follower.run()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Staged OCP controller: init pose -> user gate -> squat -> loop replay.")
    p.add_argument("--traj", type=Path, default=Path("/tmp/real_robot_jump_mirrored.npy"))
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--control-hz", type=float, default=200.0)
    p.add_argument("--go-to-init-s", type=float, default=2.0)
    p.add_argument("--squat-s", type=float, default=2.0)
    p.add_argument("--loop-count", type=int, default=3)
    p.add_argument("--no-wait-user", action="store_true", help="Do not wait for Enter before squat/replay.")
    p.add_argument("--mock", action="store_true", help="Use mock CAN buses instead of real socketcan buses.")
    p.add_argument("--log", type=Path, default=Path("bipedal_state_log.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    traj = load_ocp_npy(args.traj, dt_s=float(args.dt))
    print(f"[info] loaded {args.traj} | N={traj.t_s.size} | duration={traj.duration_s:.3f}s")

    if args.mock:
        bus0 = MockBus()
        bus1 = MockBus()
        robot = BipedalRobotController(
            bus_can0=bus0,
            bus_can1=bus1,
            control_hz=float(args.control_hz),
            log_path=args.log,
        )
        robot.set_startup_wrap_policy(enabled=False)
        robot.set_max_command_delta(1000.0)
        for mid in MOTOR_IDS:
            robot.set_joint_limit(mid, -720.0, 720.0)
        # Seed mock state to first pose.
        q0_raw = robot.joint_state_to_motor_state(traj.q_deg[0], input_radians=False, output_space="raw")
        now = time.time()
        for mid in MOTOR_IDS:
            st = robot.state[mid]
            st.position_deg = float(q0_raw[mid])
            st.velocity_deg_s = 0.0
            st.torque_nm = 0.0
            st.temp_mos_c = 30.0
            st.stamp = now
            # Also seed bus-internal states so FF command replies keep this pose.
            if mid <= 6:
                bus0._state[mid] = st
            else:
                bus1._state[mid] = st
    else:
        robot = BipedalRobotController(
            control_hz=float(args.control_hz),
            log_path=args.log,
        )

    robot.start(mode="control", auto_enable=True)
    try:
        run_staged_controller(
            robot,
            traj,
            go_to_init_s=float(args.go_to_init_s),
            squat_s=float(args.squat_s),
            loop_count=int(args.loop_count),
            wait_user=not bool(args.no_wait_user),
        )
        print("[info] staged run completed")
    finally:
        robot.stop(disable_motors=not bool(args.mock))


if __name__ == "__main__":
    main()
