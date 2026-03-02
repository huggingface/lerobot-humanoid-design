from __future__ import annotations

import csv
import json
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from RL_agent import RLAgent as NewRLAgent
from sim_robot import SimBipedalRobotController

# RL_agent_old imports bipedal_robot eagerly; provide a minimal stub.
if "bipedal_robot" not in sys.modules:
    stub = types.ModuleType("bipedal_robot")

    class _StubBipedalRobotController:  # pragma: no cover - runtime shim
        pass

    stub.BipedalRobotController = _StubBipedalRobotController
    sys.modules["bipedal_robot"] = stub

from RL_agent_old import RLAgent as OldRLAgent


@dataclass
class _MockMotorState:
    position_deg: float = 0.0
    velocity_deg_s: float = 0.0
    torque_nm: float = 0.0
    temp_mos_c: float = 35.0
    stamp: float = 0.0


class MockRobotInstant:
    def __init__(self, control_hz: float = 200.0):
        self.control_hz = float(control_hz)
        self.dt = 1.0 / max(1.0, self.control_hz)
        self._t = time.time()
        self._q_deg = np.zeros(12, dtype=float)
        self._q_prev_deg = self._q_deg.copy()
        self._qd_deg_s = np.zeros(12, dtype=float)
        self._sim_step_count = 0
        self._motors = {i: _MockMotorState() for i in range(1, 13)}
        self._sync()

    def _sync(self) -> None:
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
        q = self._q_deg.copy()
        q[0] = float(left.get("hipz", q[0]))
        q[1] = float(left.get("hipx", q[1]))
        q[2] = float(left.get("hipy", q[2]))
        q[3] = float(left.get("knee", q[3]))
        q[4] = float(left.get("ankle_pitch", q[4]))
        q[5] = float(left.get("ankle_roll", q[5]))
        q[6] = float(right.get("hipz", q[6]))
        q[7] = float(right.get("hipx", q[7]))
        q[8] = float(right.get("hipy", q[8]))
        q[9] = float(right.get("knee", q[9]))
        q[10] = float(right.get("ankle_pitch", q[10]))
        q[11] = float(right.get("ankle_roll", q[11]))
        self._qd_deg_s = (q - self._q_prev_deg) / max(1e-6, self.dt)
        self._q_prev_deg = self._q_deg.copy()
        self._q_deg = q
        self._sim_step_count += 1
        self._t += self.dt
        self._sync()
        return {i + 1: float(self._q_deg[i]) for i in range(12)}

    def get_combined_state_snapshot(self, *, include_joint_state: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "time_s": float(self._t),
            "mode": "control",
            "sim_step_count": int(self._sim_step_count),
            "sim_timestep_s": float(self.dt),
            "post_reset_hold_active": False,
            "post_reset_hold_remaining_s": 0.0,
            "motors": {mid: vars(ms) for mid, ms in self._motors.items()},
            "imu": {"gyro_rads": [0.0, 0.0, 0.0], "linear_velocity_mps": [0.0, 0.0, 0.0], "lin_vel_m_s": [0.0, 0.0, 0.0]},
            "orientation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        if include_joint_state:
            out["joint_state_deg"] = self._q_deg.tolist()
            out["joint_state_rad"] = np.deg2rad(self._q_deg).tolist()
            out["joint_velocity_deg_s"] = self._qd_deg_s.tolist()
            out["joint_velocity_rad_s"] = np.deg2rad(self._qd_deg_s).tolist()
        return out


def _qrad_to_left_right(q_rad: np.ndarray) -> tuple[Dict[str, float], Dict[str, float]]:
    q_deg = np.rad2deg(np.asarray(q_rad, dtype=float).reshape(12))
    left = {"hipz": float(q_deg[0]), "hipx": float(q_deg[1]), "hipy": float(q_deg[2]), "knee": float(q_deg[3]), "ankle_pitch": float(q_deg[4]), "ankle_roll": float(q_deg[5])}
    right = {"hipz": float(q_deg[6]), "hipx": float(q_deg[7]), "hipy": float(q_deg[8]), "knee": float(q_deg[9]), "ankle_pitch": float(q_deg[10]), "ankle_roll": float(q_deg[11])}
    return left, right


def _manual_step(agent: Any) -> Dict[str, Any]:
    obs_now = agent._build_obs_now()
    obs_hist = agent._build_history_obs(obs_now)
    obs_in = agent._adapt_obs_dim_for_policy(obs_hist)
    action = agent.policy.infer(obs_in)
    agent._last_obs = obs_in
    agent._last_action = action
    agent._apply_action(action)
    snap = agent.robot.get_combined_state_snapshot(include_joint_state=True)
    return {
        "obs": np.asarray(obs_in, dtype=float),
        "act": np.asarray(action, dtype=float),
        "q_deg": np.asarray(snap.get("joint_state_deg", [0.0] * 12), dtype=float),
    }


def _load_mjlab_first3(path: Path) -> List[Dict[str, np.ndarray]]:
    out: List[Dict[str, np.ndarray]] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            out.append(
                {
                    "obs": np.array([float(row[f"obs_{k}"]) for k in range(45)], dtype=float),
                    "act": np.array([float(row[f"action_{k}"]) for k in range(12)], dtype=float),
                }
            )
            if i >= 2:
                break
    return out


def _diff_line(a: np.ndarray, b: np.ndarray) -> str:
    d = np.abs(a - b)
    return f"mean={float(np.mean(d)):.6f} max={float(np.max(d)):.6f}"


def main() -> None:
    config_path = Path("RL_policy/config.yaml")
    policy_path = Path("RL_policy/2026-02-26_20-56-19.onnx")
    mjlab_log = Path("/home/virgile/devel/lerobot_legged_robots/logs/csv/lerobot_humanoid_no_arms_flat_play_20260227_172658.csv")
    out_csv = Path("RL_policy/rl_scale1_matrix_first3.csv")
    out_report = Path("RL_policy/rl_scale1_matrix_first3_report.txt")

    mj = _load_mjlab_first3(mjlab_log)

    # Build robots.
    mock_robot = MockRobotInstant(control_hz=200.0)
    sim_robot = SimBipedalRobotController(control_hz=200.0, fixed_base=True, fixed_base_height_m=0.8)
    sim_robot.start(mode="control", auto_enable=True)
    time.sleep(0.03)

    # Build controller matrix where possible.
    controllers: List[Tuple[str, Any]] = []
    old_mock = OldRLAgent.from_files(mock_robot, config_path=str(config_path), policy_path=str(policy_path), log_observation=False, log_action=False)
    old_mock.spec.action_scale = 1.0
    controllers.append(("old_mock", old_mock))

    old_sim = OldRLAgent.from_files(sim_robot, config_path=str(config_path), policy_path=str(policy_path), log_observation=False, log_action=False)
    old_sim.spec.action_scale = 1.0
    controllers.append(("old_sim", old_sim))

    new_mock = NewRLAgent.from_files(mock_robot, config_path=str(config_path), policy_path=str(policy_path), log_observation=False, log_action=False)
    new_mock.spec.action_scale = 1.0
    controllers.append(("new_mock", new_mock))

    new_sim = NewRLAgent.from_files(sim_robot, config_path=str(config_path), policy_path=str(policy_path), log_observation=False, log_action=False)
    new_sim.spec.action_scale = 1.0
    controllers.append(("new_sim", new_sim))

    # Initialize in reference pose.
    q_ref = np.asarray(new_sim._default_joint_pos_rad, dtype=float).reshape(12)
    left_ref, right_ref = _qrad_to_left_right(q_ref)
    mock_robot.set_action(left=left_ref, right=right_ref)
    sim_robot.set_action(left=left_ref, right=right_ref)
    time.sleep(0.05)

    # Run first 3 manual inference steps.
    rows = []
    for step in range(3):
        for name, agent in controllers:
            r = _manual_step(agent)
            rows.append((step, name, r))
        time.sleep(0.02)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "controller", "observation", "action_pre_scale", "joint_state_deg"])
        for step, name, r in rows:
            w.writerow(
                [
                    step,
                    name,
                    json.dumps(r["obs"].tolist(), separators=(",", ":")),
                    json.dumps(r["act"].tolist(), separators=(",", ":")),
                    json.dumps(r["q_deg"].tolist(), separators=(",", ":")),
                ]
            )

    # Report row-by-row against MJLab.
    lines = []
    lines.append("Scale=1 matrix comparison vs MJLab (first 3 inferences)")
    lines.append(f"MJLab source: {mjlab_log}")
    for step in range(3):
        lines.append("")
        lines.append(f"[row {step}]")
        m_obs = mj[step]["obs"]
        m_act = mj[step]["act"]
        for name, _ in controllers:
            r = [x for s, n, x in rows if s == step and n == name][0]
            lines.append(
                f"{name:8s} obs_diff({_diff_line(r['obs'], m_obs)}), act_diff({_diff_line(r['act'], m_act)})"
            )
            lines.append(
                f"{name:8s} base_ang={np.round(r['obs'][:3],4).tolist()} "
                f"joint_pos0..5={np.round(r['obs'][6:12],4).tolist()}"
            )

    out_report.write_text("\n".join(lines) + "\n")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_report}")


if __name__ == "__main__":
    main()
