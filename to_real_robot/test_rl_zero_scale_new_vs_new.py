from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from RL_agent import RLAgent
from sim_robot import SimBipedalRobotController


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


def _manual_step(agent: RLAgent) -> Dict[str, Any]:
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
        "qd_rad": np.asarray(snap.get("joint_velocity_rad_s", [0.0] * 12), dtype=float),
        "sim_step_count": int(snap.get("sim_step_count", 0)),
    }


def _qrad_to_left_right(q_rad: np.ndarray) -> tuple[Dict[str, float], Dict[str, float]]:
    q_deg = np.rad2deg(np.asarray(q_rad, dtype=float).reshape(12))
    left = {"hipz": float(q_deg[0]), "hipx": float(q_deg[1]), "hipy": float(q_deg[2]), "knee": float(q_deg[3]), "ankle_pitch": float(q_deg[4]), "ankle_roll": float(q_deg[5])}
    right = {"hipz": float(q_deg[6]), "hipx": float(q_deg[7]), "hipy": float(q_deg[8]), "knee": float(q_deg[9]), "ankle_pitch": float(q_deg[10]), "ankle_roll": float(q_deg[11])}
    return left, right


def main() -> None:
    config_path = Path("RL_policy/config.yaml")
    policy_path = Path("RL_policy/2026-02-26_20-56-19.onnx")
    out_csv = Path("RL_policy/rl_zero_scale_first3_new_vs_new.csv")
    out_report = Path("RL_policy/rl_zero_scale_first3_new_vs_new_report.txt")
    n_steps = 3

    mock = MockRobotInstant(control_hz=200.0)
    sim = SimBipedalRobotController(control_hz=200.0, fixed_base=True, fixed_base_height_m=0.8)
    sim.start(mode="control", auto_enable=True)
    time.sleep(0.03)

    mock_agent = RLAgent.from_files(mock, config_path=str(config_path), policy_path=str(policy_path), log_observation=False, log_action=False)
    sim_agent = RLAgent.from_files(sim, config_path=str(config_path), policy_path=str(policy_path), log_observation=False, log_action=False)
    mock_agent.spec.action_scale = 0.0
    sim_agent.spec.action_scale = 0.0

    q_ref = np.asarray(sim_agent._default_joint_pos_rad, dtype=float).reshape(12)
    left_ref, right_ref = _qrad_to_left_right(q_ref)
    mock.set_action(left=left_ref, right=right_ref)
    sim.set_action(left=left_ref, right=right_ref)
    time.sleep(0.05)

    rows = []
    for step in range(n_steps):
        m = _manual_step(mock_agent)
        s = _manual_step(sim_agent)
        rows.append((step, "new_mock", m))
        rows.append((step, "new_sim", s))
        time.sleep(0.02)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "controller", "sim_step_count", "observation", "action_pre_scale", "joint_state_deg", "joint_vel_rad_s"])
        for step, name, r in rows:
            w.writerow(
                [
                    step,
                    name,
                    r["sim_step_count"],
                    json.dumps(r["obs"].tolist(), separators=(",", ":")),
                    json.dumps(r["act"].tolist(), separators=(",", ":")),
                    json.dumps(r["q_deg"].tolist(), separators=(",", ":")),
                    json.dumps(r["qd_rad"].tolist(), separators=(",", ":")),
                ]
            )

    lines = []
    lines.append("RL zero-scale first-3 inference report (new controller on both mock and sim)")
    for step in range(n_steps):
        m = [r for s, n, r in rows if s == step and n == "new_mock"][0]
        s = [r for s, n, r in rows if s == step and n == "new_sim"][0]
        lines.append(
            f"step {step}: mean|obs|={float(np.mean(np.abs(m['obs']-s['obs']))):.6f}, "
            f"mean|act|={float(np.mean(np.abs(m['act']-s['act']))):.6f}, "
            f"mean|q_deg|={float(np.mean(np.abs(m['q_deg']-s['q_deg']))):.6f}, "
            f"mean|qd|={float(np.mean(np.abs(m['qd_rad']-s['qd_rad']))):.6f}"
        )
        lines.append(
            f"step {step}: mock act_abs_max={float(np.max(np.abs(m['act']))):.6f}, "
            f"sim act_abs_max={float(np.max(np.abs(s['act']))):.6f}"
        )
    out_report.write_text("\n".join(lines) + "\n")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_report}")


if __name__ == "__main__":
    main()
