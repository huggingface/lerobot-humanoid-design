from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sim_robot import SimBipedalRobotController


JOINT_NAMES = [
    "L_hipz",
    "L_hipx",
    "L_hipy",
    "L_knee",
    "L_ank_pitch",
    "L_ank_roll",
    "R_hipz",
    "R_hipx",
    "R_hipy",
    "R_knee",
    "R_ank_pitch",
    "R_ank_roll",
]


@dataclass
class _MockMotorState:
    position_deg: float = 0.0
    velocity_deg_s: float = 0.0
    torque_nm: float = 0.0
    temp_mos_c: float = 35.0
    stamp: float = 0.0


class MockRobotInstant:
    """Simple robot model: position command is reached instantly."""

    def __init__(self, control_hz: float = 200.0):
        self.control_hz = float(control_hz)
        self.dt = 1.0 / max(1.0, self.control_hz)
        self._t = time.time()
        self._q_deg = np.zeros(12, dtype=float)
        self._q_prev_deg = self._q_deg.copy()
        self._qd_deg_s = np.zeros(12, dtype=float)
        self._sim_step_count = 0
        self._motors: Dict[int, _MockMotorState] = {i: _MockMotorState() for i in range(1, 13)}
        self._sync_motor_states()

    def _sync_motor_states(self) -> None:
        now = time.time()
        for i in range(12):
            self._motors[i + 1] = _MockMotorState(
                position_deg=float(self._q_deg[i]),
                velocity_deg_s=float(self._qd_deg_s[i]),
                stamp=now,
            )

    def set_action(
        self,
        *,
        left: Optional[Dict[str, float]] = None,
        right: Optional[Dict[str, float]] = None,
        **_: Any,
    ) -> Dict[int, float]:
        left = left or {}
        right = right or {}
        q_cmd = self._q_deg.copy()
        q_cmd[0] = float(left.get("hipz", q_cmd[0]))
        q_cmd[1] = float(left.get("hipx", q_cmd[1]))
        q_cmd[2] = float(left.get("hipy", q_cmd[2]))
        q_cmd[3] = float(left.get("knee", q_cmd[3]))
        q_cmd[4] = float(left.get("ankle_pitch", q_cmd[4]))
        q_cmd[5] = float(left.get("ankle_roll", q_cmd[5]))
        q_cmd[6] = float(right.get("hipz", q_cmd[6]))
        q_cmd[7] = float(right.get("hipx", q_cmd[7]))
        q_cmd[8] = float(right.get("hipy", q_cmd[8]))
        q_cmd[9] = float(right.get("knee", q_cmd[9]))
        q_cmd[10] = float(right.get("ankle_pitch", q_cmd[10]))
        q_cmd[11] = float(right.get("ankle_roll", q_cmd[11]))

        self._qd_deg_s = (q_cmd - self._q_prev_deg) / max(1e-6, self.dt)
        self._q_prev_deg = self._q_deg.copy()
        self._q_deg = q_cmd
        self._sim_step_count += 1
        self._t += self.dt
        self._sync_motor_states()
        return {i + 1: float(self._q_deg[i]) for i in range(12)}

    def get_combined_state_snapshot(self, *, include_joint_state: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "time_s": float(self._t),
            "sim_step_count": int(self._sim_step_count),
            "sim_timestep_s": float(self.dt),
            "motors": {mid: vars(ms) for mid, ms in self._motors.items()},
        }
        if include_joint_state:
            out["joint_state_deg"] = self._q_deg.tolist()
            out["joint_velocity_deg_s"] = self._qd_deg_s.tolist()
            out["joint_velocity_rad_s"] = np.deg2rad(self._qd_deg_s).tolist()
        return out


def _to_left_right(q_deg: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
    q = np.asarray(q_deg, dtype=float).reshape(12)
    left = {
        "hipz": float(q[0]),
        "hipx": float(q[1]),
        "hipy": float(q[2]),
        "knee": float(q[3]),
        "ankle_pitch": float(q[4]),
        "ankle_roll": float(q[5]),
    }
    right = {
        "hipz": float(q[6]),
        "hipx": float(q[7]),
        "hipy": float(q[8]),
        "knee": float(q[9]),
        "ankle_pitch": float(q[10]),
        "ankle_roll": float(q[11]),
    }
    return left, right


def _build_position_requests(base_q_deg: np.ndarray) -> List[np.ndarray]:
    # Small deterministic sequence around knee-bent pose.
    reqs: List[np.ndarray] = []
    reqs.append(base_q_deg.copy())
    for k in range(1, 13):
        q = base_q_deg.copy()
        s = 1.0 if (k % 2 == 0) else -1.0
        q[1] += 8.0 * s       # L hipx
        q[7] -= 8.0 * s       # R hipx mirror
        q[2] += 6.0 * s       # L hipy
        q[8] -= 6.0 * s       # R hipy mirror
        q[4] += 5.0 * s       # L ankle pitch
        q[10] -= 5.0 * s      # R ankle pitch mirror
        q[3] += 4.0 * (1.0 if k % 3 == 0 else -1.0)
        q[9] += 4.0 * (1.0 if k % 3 == 0 else -1.0)
        reqs.append(q)
    return reqs


def _write_sign_report(csv_path: Path, report_path: Path) -> None:
    rows: Dict[str, List[np.ndarray]] = {"mock": [], "sim": []}
    with csv_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            rows[row["robot"]].append(np.array(json.loads(row["joint_state_deg"]), dtype=float))

    m = rows["mock"]
    s = rows["sim"]
    k = min(len(m), len(s))
    if k < 2:
        report_path.write_text("Not enough rows\n")
        return
    m_arr = np.stack(m[:k])
    s_arr = np.stack(s[:k])
    dm = np.diff(m_arr, axis=0)
    ds = np.diff(s_arr, axis=0)
    eps = 1e-6

    lines = []
    lines.append(f"steps compared: {k}, transitions: {k-1}")
    lines.append("per-joint sign agreement on delta(q_deg):")
    for j, name in enumerate(JOINT_NAMES):
        mask = (np.abs(dm[:, j]) > eps) & (np.abs(ds[:, j]) > eps)
        if np.any(mask):
            agree = float(np.mean(np.sign(dm[mask, j]) == np.sign(ds[mask, j])))
            lines.append(f"  {name:12s} agree={agree:.3f} samples={int(np.sum(mask))}")
        else:
            lines.append(f"  {name:12s} agree=NA samples=0")
    lines.append("")
    lines.append("first transition deltas old/new (deg):")
    for j, name in enumerate(JOINT_NAMES):
        ok = int(np.sign(dm[0, j]) == np.sign(ds[0, j]))
        lines.append(f"  {name:12s} mock={dm[0, j]:+8.3f} sim={ds[0, j]:+8.3f} sign_ok={ok}")

    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    out_csv = Path("RL_policy/compare_simple_position_request.csv")
    out_report = Path("RL_policy/compare_simple_position_request_sign_report.txt")

    # Knee-bent init (deg), matching the usual convention in current scripts.
    base_q_deg = np.array(
        [0.0, 0.0, -20.0535, 40.1070, -20.0535, 0.0, 0.0, 0.0, 20.0535, 40.1070, 20.0535, 0.0],
        dtype=float,
    )
    requests = _build_position_requests(base_q_deg)

    mock = MockRobotInstant(control_hz=200.0)
    sim = SimBipedalRobotController(control_hz=200.0, fixed_base=True, fixed_base_height_m=0.8)
    sim.start(mode="control", auto_enable=True)
    time.sleep(0.03)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "step",
                "robot",
                "time_s",
                "sim_step_count",
                "requested_joint_deg",
                "joint_state_deg",
                "joint_vel_rad_s",
            ]
        )
        for step, q_req in enumerate(requests):
            left, right = _to_left_right(q_req)
            mock.set_action(left=left, right=right)
            sim.set_action(left=left, right=right)
            # Let sim evolve under inertia for one policy period.
            time.sleep(0.02)

            m_snap = mock.get_combined_state_snapshot(include_joint_state=True)
            s_snap = sim.get_combined_state_snapshot(include_joint_state=True)
            for robot_name, snap in (("mock", m_snap), ("sim", s_snap)):
                w.writerow(
                    [
                        step,
                        robot_name,
                        f"{float(snap.get('time_s', time.time())):.6f}",
                        int(snap.get("sim_step_count", 0)),
                        json.dumps(q_req.tolist(), separators=(",", ":")),
                        json.dumps(list(snap.get("joint_state_deg", [0.0] * 12)), separators=(",", ":")),
                        json.dumps(list(snap.get("joint_velocity_rad_s", [0.0] * 12)), separators=(",", ":")),
                    ]
                )

    _write_sign_report(out_csv, out_report)
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_report}")


if __name__ == "__main__":
    main()
