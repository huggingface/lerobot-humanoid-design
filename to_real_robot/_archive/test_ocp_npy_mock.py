import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from bipedal_robot import BipedalRobotController
from leg_test.mit import MotorState
from mock_bus import MockBus
from ocp_follower import OCPTrajectory, OCPFollower
from root_constant import JOINT_LIMITS_DEG, MOTORS, MOTOR_IDS, STATE_MARGIN_DEG


def _swap_hx_hy_order(x: np.ndarray) -> np.ndarray:
    """
    OCP order (from robot.yaml):
      [hipz, hipy, hipx, knee, anklex, ankley] x 2
    Controller order:
      [hipz, hipx, hipy, knee, ankle_pitch, ankle_roll] x 2
    """
    idx = [0, 2, 1, 3, 4, 5, 6, 8, 7, 9, 10, 11]
    return x[:, idx]


def load_ocp_npy(path: Path, *, dt_s: float) -> tuple[OCPTrajectory, Dict[str, np.ndarray]]:
    arr = np.load(path, allow_pickle=True)
    if not (isinstance(arr, np.ndarray) and arr.shape == () and arr.dtype == object):
        raise RuntimeError(f"Unsupported npy format in {path}: expected object dict saved by sobec.wwt.save_traj.")

    blob = arr.item()
    if not isinstance(blob, dict):
        raise RuntimeError(f"Unsupported npy payload type in {path}: {type(blob)}")
    if "xs" not in blob or "us" not in blob:
        raise RuntimeError(f"Missing xs/us in {path}. Found keys: {sorted(blob.keys())}")

    xs = np.asarray(blob["xs"], dtype=float)
    us = np.asarray(blob["us"], dtype=float)
    if xs.ndim != 2 or us.ndim != 2:
        raise RuntimeError(f"Invalid shapes: xs={xs.shape}, us={us.shape}. Expected 2D arrays.")
    if xs.shape[1] < 37 or us.shape[1] < 12:
        raise RuntimeError(
            f"Unexpected dimensions for real robot jump: xs={xs.shape}, us={us.shape}. "
            "Need at least 37 state dims and 12 control dims."
        )

    # For this robot setup: nq=19, nv=18, actuated joints are last 12 in q and v.
    nq = xs.shape[1] - 18
    q_act = np.asarray(xs[:, (nq - 12):nq], dtype=float)
    v_act = np.asarray(xs[:, -12:], dtype=float)
    tau = np.asarray(us[:, :12], dtype=float)

    q_act = _swap_hx_hy_order(q_act)
    v_act = _swap_hx_hy_order(v_act)
    tau = _swap_hx_hy_order(tau)

    q_deg = np.rad2deg(q_act)
    qd_deg_s = np.rad2deg(v_act)

    n = xs.shape[0]
    if tau.shape[0] == n - 1:
        tau = np.vstack([tau, tau[-1]])
    elif tau.shape[0] != n:
        raise RuntimeError(f"Control horizon mismatch: xs rows={n}, us rows={tau.shape[0]}.")

    t_s = np.arange(n, dtype=float) * float(dt_s)
    traj = OCPTrajectory(t_s=t_s, q_deg=q_deg, qd_deg_s=qd_deg_s, tau_ff_nm=tau)
    return traj, {"q_deg": q_deg, "qd_deg_s": qd_deg_s, "tau_ff_nm": tau}


def replay_with_mock(traj: OCPTrajectory, *, control_hz: float, log_path: Path) -> None:
    bus0 = MockBus()
    bus1 = MockBus()
    robot = BipedalRobotController(
        bus_can0=bus0,
        bus_can1=bus1,
        control_hz=float(control_hz),
        log_path=log_path,
    )
    # In pure mock replay, start from zero state and avoid false startup/limit trips.
    robot.set_startup_wrap_policy(enabled=False)
    for mid in MOTOR_IDS:
        robot.set_joint_limit(mid, -720.0, 720.0)
    robot.set_max_command_delta(1000.0)

    # Seed mock encoder states at first trajectory frame to remove startup mismatch.
    q0_raw = robot.joint_state_to_motor_state(traj.q_deg[0], input_radians=False, output_space="raw")
    for mid in MOTOR_IDS:
        st = MotorState(
            position_deg=float(q0_raw[mid]),
            velocity_deg_s=0.0,
            torque_nm=0.0,
            temp_mos_c=30.0,
            stamp=time.time(),
        )
        if mid <= 6:
            bus0._state[mid] = st
        else:
            bus1._state[mid] = st

    robot.start(mode="control", auto_enable=True)
    try:
        follower = OCPFollower(robot)
        follower.set_trajectory(traj)
        follower.run()
        # Let the control thread process final action updates before shutdown.
        time.sleep(0.1)
    finally:
        robot.stop(disable_motors=True)


def _read_log(path: Path) -> Dict[str, np.ndarray]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in log: {path}")

    out: Dict[str, np.ndarray] = {}
    out["time_s"] = np.array([float(r["time_s"]) for r in rows], dtype=float)
    out["estop"] = np.array([int(r["estop"]) for r in rows], dtype=int)
    out["missing_ids"] = np.array([r["missing_ids"] for r in rows], dtype=object)
    out["estop_reason"] = np.array([r["estop_reason"] for r in rows], dtype=object)
    out["mode"] = np.array([r["mode"] for r in rows], dtype=object)

    for mid in MOTOR_IDS:
        out[f"m{mid}_pos"] = np.array([float(r[f"m{mid}_pos_deg"]) for r in rows], dtype=float)
        out[f"m{mid}_tgt"] = np.array([float(r[f"m{mid}_target_pos_deg"]) for r in rows], dtype=float)
        out[f"m{mid}_vel"] = np.array([float(r[f"m{mid}_vel_deg_s"]) for r in rows], dtype=float)
        out[f"m{mid}_tau"] = np.array([float(r[f"m{mid}_tau_nm"]) for r in rows], dtype=float)
    return out


def _trajectory_feasibility(raw: Dict[str, np.ndarray]) -> Dict[str, float]:
    tau = raw["tau_ff_nm"]
    qd_deg_s = raw["qd_deg_s"]
    tau_over_naive = 0
    tau_over_motor_mapped = 0
    vel_over = 0
    for j, mid in enumerate(MOTOR_IDS):
        tmax = float(MOTORS[mid].tmax_nm)
        vmax_deg_s = float(np.degrees(MOTORS[mid].vmax_rad_s))
        tau_over_naive += int(np.sum(np.abs(tau[:, j]) > tmax))
        vel_over += int(np.sum(np.abs(qd_deg_s[:, j]) > vmax_deg_s))

    # Correct check in motor space (accounts for ankle coupling).
    robot = BipedalRobotController(
        bus_can0=MockBus(),
        bus_can1=MockBus(),
        control_hz=100.0,
        log_path=Path("/tmp/_traj_feas_tmp.csv"),
    )
    follower = OCPFollower(robot)
    for k in range(tau.shape[0]):
        _, tau_raw = follower._joint_vel_tau_to_motor_raw(qd_deg_s[k], tau[k])
        for mid in MOTOR_IDS:
            if abs(float(tau_raw[mid])) > float(MOTORS[mid].tmax_nm):
                tau_over_motor_mapped += 1

    return {
        "traj_tau_cmd_over_limit_samples_naive": float(tau_over_naive),
        "traj_tau_cmd_over_limit_samples_motor_mapped": float(tau_over_motor_mapped),
        "traj_vel_cmd_over_limit_samples": float(vel_over),
    }


def analyze_log(log_path: Path, raw: Dict[str, np.ndarray]) -> Tuple[bool, Dict[str, float]]:
    data = _read_log(log_path)
    stats: Dict[str, float] = {}

    stats["rows"] = float(data["time_s"].size)
    stats["duration_s"] = float(data["time_s"][-1] - data["time_s"][0]) if data["time_s"].size > 1 else 0.0
    stats["estop_rows"] = float(np.sum(data["estop"] > 0))
    stats["missing_ids_rows"] = float(np.sum(np.array([1 if str(x).strip() else 0 for x in data["missing_ids"]], dtype=int)))

    max_abs_track = 0.0
    limit_violations = 0
    vel_sat = 0
    tau_sat = 0
    tau_hard_over = 0
    vel_hard_over = 0

    for mid in MOTOR_IDS:
        pos = data[f"m{mid}_pos"]
        tgt = data[f"m{mid}_tgt"]
        vel = np.abs(data[f"m{mid}_vel"])
        tau = np.abs(data[f"m{mid}_tau"])
        err = np.abs(tgt - pos)
        max_abs_track = max(max_abs_track, float(np.max(err)))

        lo, hi = JOINT_LIMITS_DEG[mid]
        lo_b = float(lo) - float(STATE_MARGIN_DEG)
        hi_b = float(hi) + float(STATE_MARGIN_DEG)
        limit_violations += int(np.sum((pos < lo_b) | (pos > hi_b)))

        vmax = float(np.degrees(MOTORS[mid].vmax_rad_s))
        tmax = float(MOTORS[mid].tmax_nm)
        vel_sat += int(np.sum(vel >= 0.98 * vmax))
        tau_sat += int(np.sum(tau >= 0.98 * tmax))
        vel_hard_over += int(np.sum(vel > vmax))
        tau_hard_over += int(np.sum(tau > tmax))

    stats["max_abs_tracking_err_deg"] = float(max_abs_track)
    stats["state_limit_violation_samples"] = float(limit_violations)
    stats["vel_near_limit_samples"] = float(vel_sat)
    stats["tau_near_limit_samples"] = float(tau_sat)
    stats["vel_over_limit_samples"] = float(vel_hard_over)
    stats["tau_over_limit_samples"] = float(tau_hard_over)
    stats.update(_trajectory_feasibility(raw))

    feasible = (
        stats["estop_rows"] == 0.0
        and stats["state_limit_violation_samples"] == 0.0
        and stats["vel_over_limit_samples"] == 0.0
        and stats["tau_over_limit_samples"] == 0.0
        and stats["traj_tau_cmd_over_limit_samples_motor_mapped"] == 0.0
        and stats["traj_vel_cmd_over_limit_samples"] == 0.0
    )
    return feasible, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay OCP .npy trajectory on MockBus and evaluate feasibility from logs.")
    parser.add_argument("--traj", type=Path, default=Path("/tmp/real_robot_jump.npy"))
    parser.add_argument("--dt", type=float, default=0.005, help="Trajectory step [s] used for xs/us timeline.")
    parser.add_argument("--control-hz", type=float, default=200.0, help="Controller loop frequency [Hz].")
    parser.add_argument("--log", type=Path, default=Path("/tmp/real_robot_jump_mock_log.csv"))
    args = parser.parse_args()

    traj, raw = load_ocp_npy(args.traj, dt_s=float(args.dt))
    print(f"[INFO] loaded trajectory: N={traj.t_s.size}, duration={traj.duration_s:.3f}s, source={args.traj}")
    replay_with_mock(traj, control_hz=float(args.control_hz), log_path=args.log)
    print(f"[INFO] replay finished, log={args.log}")

    feasible, stats = analyze_log(args.log, raw=raw)
    print("[INFO] feasibility summary")
    for k in sorted(stats.keys()):
        print(f"{k}: {stats[k]}")
    print(f"feasible: {feasible}")


if __name__ == "__main__":
    main()
