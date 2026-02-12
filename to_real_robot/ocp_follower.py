#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import csv
import time

import numpy as np

from bipedal_robot import BipedalRobotController, MotorCommand
from root_constant import ANKLE_COUPLING_CALIBRATION_LEFT, ANKLE_COUPLING_CALIBRATION_RIGHT, MOTOR_IDS


@dataclass
class OCPTrajectory:
    t_s: np.ndarray          # shape [N]
    q_deg: np.ndarray        # shape [N, 12]
    qd_deg_s: np.ndarray     # shape [N, 12]
    tau_ff_nm: np.ndarray    # shape [N, 12]

    @property
    def duration_s(self) -> float:
        if self.t_s.size == 0:
            return 0.0
        return float(self.t_s[-1] - self.t_s[0])


def _find_cols(header, prefix: str) -> list[str]:
    out = []
    for i in range(12):
        candidates = [
            f"{prefix}{i}",
            f"{prefix}_{i}",
            f"{prefix}[{i}]",
            f"{prefix}{i+1}",
            f"{prefix}_{i+1}",
            f"{prefix}[{i+1}]",
        ]
        found = next((c for c in candidates if c in header), None)
        if found is None:
            return []
        out.append(found)
    return out


def load_ocp_csv(path: Path) -> OCPTrajectory:
    """
    Expected columns:
      - time: one of [t, time, timestamp, time_s]
      - q: q0..q11 (or q_0..q_11, q[0]..q[11], optionally 1-based indices)
      - qd: qd0..qd11 (same variants)
      - tau_ff: tau0..tau11 OR tau_ff0..tau_ff11 (same variants)
    Units expected: deg, deg/s, Nm.
    """
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty trajectory CSV: {path}")
    header = list(rows[0].keys())

    t_col = next((c for c in ["t", "time", "timestamp", "time_s"] if c in header), None)
    if t_col is None:
        raise RuntimeError("Missing time column. Expected one of: t, time, timestamp, time_s")

    q_cols = _find_cols(header, "q")
    qd_cols = _find_cols(header, "qd")
    tau_cols = _find_cols(header, "tau_ff")
    if not tau_cols:
        tau_cols = _find_cols(header, "tau")
    if len(q_cols) != 12 or len(qd_cols) != 12 or len(tau_cols) != 12:
        raise RuntimeError(
            "Missing trajectory columns. Need q[12], qd[12], tau_ff[12] "
            "(accepted names: q0..q11, qd0..qd11, tau_ff0..tau_ff11)."
        )

    t = np.array([float(r[t_col]) for r in rows], dtype=float)
    q = np.array([[float(r[c]) for c in q_cols] for r in rows], dtype=float)
    qd = np.array([[float(r[c]) for c in qd_cols] for r in rows], dtype=float)
    tau = np.array([[float(r[c]) for c in tau_cols] for r in rows], dtype=float)

    if np.any(np.diff(t) < 0):
        idx = np.argsort(t)
        t = t[idx]
        q = q[idx]
        qd = qd[idx]
        tau = tau[idx]

    # Normalize timeline to start at 0.
    t = t - float(t[0])
    return OCPTrajectory(t_s=t, q_deg=q, qd_deg_s=qd, tau_ff_nm=tau)


class OCPFollower:
    """
    Real-time OCP follower:
      - read desired q/qd/tau in model joint space
      - map to motor raw position/velocity/torque
      - write commands into controller action buffer
    """

    def __init__(self, robot: BipedalRobotController):
        self.robot = robot
        self.trajectory: Optional[OCPTrajectory] = None
        self._stop = False

    def set_trajectory(self, traj: OCPTrajectory) -> None:
        self.trajectory = traj

    def stop(self) -> None:
        self._stop = True

    def _interp_row(self, t_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        traj = self.trajectory
        if traj is None:
            raise RuntimeError("No trajectory set.")

        t = traj.t_s
        t_query = float(np.clip(t_s, t[0], t[-1]))
        q = np.array([np.interp(t_query, t, traj.q_deg[:, i]) for i in range(12)], dtype=float)
        qd = np.array([np.interp(t_query, t, traj.qd_deg_s[:, i]) for i in range(12)], dtype=float)
        tau = np.array([np.interp(t_query, t, traj.tau_ff_nm[:, i]) for i in range(12)], dtype=float)
        return q, qd, tau

    def _joint_vel_tau_to_motor_raw(self, qd_deg_s: np.ndarray, tau_nm: np.ndarray) -> tuple[Dict[int, float], Dict[int, float]]:
        """
        Map model joint-space (qd, tau) to motor raw-space.
        Direct joints:
          q_cal = sign * q_raw + offset  -> qd_raw = qd_cal / sign
          power consistency -> tau_raw = tau_cal * sign
        Coupled ankles:
          pitch = sp*(a1-a2)/2, roll = sr*(a1+a2)/2
          solve for (a1_dot, a2_dot), and tau_motor = J^T * tau_joint
        """
        qd_raw: Dict[int, float] = {}
        tau_raw: Dict[int, float] = {}

        # Direct joints map: model index -> motor id
        direct = {
            0: 1, 1: 2, 2: 3, 3: 4,
            6: 7, 7: 8, 8: 9, 9: 10,
        }
        for qi, mid in direct.items():
            s = float(self.robot.motor_sign[mid])
            if abs(s) < 1e-9:
                raise ValueError(f"Invalid motor sign for m{mid}: {s}")
            qd_raw[mid] = float(qd_deg_s[qi] / s)
            tau_raw[mid] = float(tau_nm[qi] * s)

        # Left ankle (q indices 4,5 -> motors 5,6)
        sp_l = float(ANKLE_COUPLING_CALIBRATION_LEFT["pitch"]["sign"])
        sr_l = float(ANKLE_COUPLING_CALIBRATION_LEFT["roll"]["sign"])
        u_l = float(qd_deg_s[4]) / sp_l
        v_l = float(qd_deg_s[5]) / sr_l
        qd_raw[5] = u_l + v_l
        qd_raw[6] = v_l - u_l
        tau_raw[5] = 0.5 * sp_l * float(tau_nm[4]) + 0.5 * sr_l * float(tau_nm[5])
        tau_raw[6] = -0.5 * sp_l * float(tau_nm[4]) + 0.5 * sr_l * float(tau_nm[5])

        # Right ankle (q indices 10,11 -> motors 11,12)
        sp_r = float(ANKLE_COUPLING_CALIBRATION_RIGHT["pitch"]["sign"])
        sr_r = float(ANKLE_COUPLING_CALIBRATION_RIGHT["roll"]["sign"])
        u_r = float(qd_deg_s[10]) / sp_r
        v_r = float(qd_deg_s[11]) / sr_r
        qd_raw[11] = u_r + v_r
        qd_raw[12] = v_r - u_r
        tau_raw[11] = 0.5 * sp_r * float(tau_nm[10]) + 0.5 * sr_r * float(tau_nm[11])
        tau_raw[12] = -0.5 * sp_r * float(tau_nm[10]) + 0.5 * sr_r * float(tau_nm[11])

        return qd_raw, tau_raw

    def run(self, *, blocking: bool = True, t_offset_s: float = 0.0) -> None:
        """
        Blocking replay loop.
        Requires controller running in control mode.
        """
        if self.trajectory is None:
            raise RuntimeError("No trajectory set.")
        if self.robot.mode != "control":
            print("[WARN] OCPFollower: robot.mode is not 'control'. Commands may not be sent.")

        self._stop = False
        traj = self.trajectory
        t0_wall = time.perf_counter()
        period = 1.0 / max(1.0, float(self.robot.control_hz))

        while not self._stop:
            t_rel = (time.perf_counter() - t0_wall) + float(t_offset_s)
            if t_rel > float(traj.t_s[-1]):
                break
            q_deg, qd_deg_s, tau_nm = self._interp_row(t_rel)

            # Joint-space desired position -> raw motor desired position.
            pos_raw = self.robot.joint_state_to_motor_state(q_deg, input_radians=False, output_space="raw")
            vel_raw, tau_raw = self._joint_vel_tau_to_motor_raw(qd_deg_s, tau_nm)

            # Update controller action buffer. Existing safety remains in MIT send path.
            with self.robot._action_lock:
                for mid in MOTOR_IDS:
                    prev = self.robot.action[mid]
                    self.robot.action[mid] = MotorCommand(
                        position_deg=float(pos_raw[mid]),
                        velocity_deg_s=float(vel_raw[mid]),
                        torque_nm=float(tau_raw[mid]),
                        kp=prev.kp,
                        kd=prev.kd,
                    )

            time.sleep(period)


def run_from_csv(robot: BipedalRobotController, csv_path: Path) -> OCPFollower:
    follower = OCPFollower(robot)
    follower.set_trajectory(load_ocp_csv(csv_path))
    follower.run()
    return follower

