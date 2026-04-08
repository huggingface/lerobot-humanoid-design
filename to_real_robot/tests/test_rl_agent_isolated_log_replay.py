from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from RL_agent_isolated import RLAgent, _load_config, _extract_default_joint_pos_rad_from_cfg


DEFAULT_LOG_PATH = Path(
    "/home/virgile/devel/lerobot_legged_robots/logs/csv/lerobot_humanoid_no_arms_flat_play_20260228_110054.csv"
)
DEFAULT_CONFIG_PATH = Path("RL_policy/config.yaml")
DEFAULT_POLICY_PATH = Path("RL_policy/2026-02-26_20-56-19.onnx")
DEFAULT_REPORT_PATH = Path("RL_policy/rl_agent_isolated_log_replay_report.txt")

# log/policy order = right(6), left(6)
POLICY_TO_SNAPSHOT_IDX = np.array([6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5], dtype=np.int64)


@dataclass
class ReplayStep:
    obs: np.ndarray
    action: np.ndarray
    q_policy_rad: np.ndarray
    qd_policy_rad_s: np.ndarray


class LogReplayRobot:
    """
    Minimal robot API shim for RLAgent:
      - get_combined_state_snapshot(include_joint_state=True)
      - set_action(left=..., right=...)
    """

    def __init__(self, rows: List[ReplayStep]) -> None:
        self.rows = rows
        self.idx = 0
        self.last_q_cmd_policy_rad = np.zeros(12, dtype=np.float32)

    def get_combined_state_snapshot(self, *, include_joint_state: bool = True) -> Dict[str, Any]:
        i = min(self.idx, len(self.rows) - 1)
        row = self.rows[i]
        q_snap_rad = row.q_policy_rad[POLICY_TO_SNAPSHOT_IDX]
        qd_snap_rad_s = row.qd_policy_rad_s[POLICY_TO_SNAPSHOT_IDX]
        snap: Dict[str, Any] = {
            "time_s": float(i),
            "imu": {"gyro_rads": row.obs[0:3].tolist()},
            "projected_gravity": row.obs[3:6].tolist(),
            "orientation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        if include_joint_state:
            snap["joint_state_deg"] = np.rad2deg(q_snap_rad).astype(np.float32).tolist()
            snap["joint_velocity_rad_s"] = qd_snap_rad_s.astype(np.float32).tolist()
        return snap

    def set_action(self, *, left: Dict[str, float], right: Dict[str, float]) -> Dict[int, float]:
        q_deg_policy = np.array(
            [
                right.get("hipz", 0.0),
                right.get("hipx", 0.0),
                right.get("hipy", 0.0),
                right.get("knee", 0.0),
                right.get("ankle_pitch", 0.0),
                right.get("ankle_roll", 0.0),
                left.get("hipz", 0.0),
                left.get("hipx", 0.0),
                left.get("hipy", 0.0),
                left.get("knee", 0.0),
                left.get("ankle_pitch", 0.0),
                left.get("ankle_roll", 0.0),
            ],
            dtype=np.float32,
        )
        self.last_q_cmd_policy_rad = np.deg2rad(q_deg_policy).astype(np.float32, copy=False)
        self.idx = min(self.idx + 1, len(self.rows) - 1)
        return {}


def _detect_raw_to_policy_index(
    q_raw_rows: List[np.ndarray],
    obs_rows: List[np.ndarray],
    q_ref_rad: np.ndarray,
    obs_slice: slice,
) -> np.ndarray:
    candidates = [np.arange(12, dtype=np.int64), POLICY_TO_SNAPSHOT_IDX]
    best_rmse = float("inf")
    best_idx = candidates[0]
    n = min(len(q_raw_rows), len(obs_rows), 8)
    for idx in candidates:
        errs = []
        for i in range(n):
            q_pol = q_raw_rows[i][idx]
            errs.append(float(np.sqrt(np.mean((q_pol - q_ref_rad - obs_rows[i][obs_slice]) ** 2))))
        rmse = float(np.mean(errs)) if errs else float("inf")
        if rmse < best_rmse:
            best_rmse = rmse
            best_idx = idx
    return best_idx


def _load_replay_rows(log_path: Path, max_steps: int, q_ref_rad: np.ndarray) -> List[ReplayStep]:
    with log_path.open() as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError(f"No header in log: {log_path}")
        fieldnames = list(r.fieldnames)
        obs_cols = sorted([k for k in fieldnames if k.startswith("obs_")], key=lambda x: int(x.split("_")[1]))
        action_cols = sorted(
            [k for k in fieldnames if k.startswith("action_") and k[7:].isdigit()],
            key=lambda x: int(x.split("_")[1]),
        )
        q_cols = sorted(
            [k for k in fieldnames if k.startswith("robot_state_joint_pos_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        qd_cols = sorted(
            [k for k in fieldnames if k.startswith("robot_state_joint_vel_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        if len(obs_cols) < 45 or len(action_cols) < 12 or len(q_cols) < 12 or len(qd_cols) < 12:
            raise ValueError(
                f"Unexpected columns in {log_path}: obs={len(obs_cols)}, action={len(action_cols)}, "
                f"q={len(q_cols)}, qd={len(qd_cols)}"
            )

        raw_obs: List[np.ndarray] = []
        raw_act: List[np.ndarray] = []
        raw_q: List[np.ndarray] = []
        raw_qd: List[np.ndarray] = []
        for row in r:
            if row.get("env_index", "0") != "0":
                continue
            raw_obs.append(np.array([float(row[k]) for k in obs_cols[:45]], dtype=np.float32))
            raw_act.append(np.array([float(row[k]) for k in action_cols[:12]], dtype=np.float32))
            raw_q.append(np.array([float(row[k]) for k in q_cols[:12]], dtype=np.float32))
            raw_qd.append(np.array([float(row[k]) for k in qd_cols[:12]], dtype=np.float32))
            if len(raw_obs) >= max_steps:
                break
    if not raw_obs:
        raise ValueError(f"No env_index=0 rows in {log_path}")

    q_unit_scale = np.deg2rad(1.0) if float(np.nanmax(np.abs(raw_q[0]))) > 6.5 else 1.0
    qd_unit_scale = np.deg2rad(1.0) if float(np.nanmax(np.abs(raw_qd[0]))) > 30.0 else 1.0
    q_raw_rows = [q * q_unit_scale for q in raw_q]
    qd_raw_rows = [qd * qd_unit_scale for qd in raw_qd]

    q_raw_to_policy = _detect_raw_to_policy_index(q_raw_rows, raw_obs, q_ref_rad, slice(6, 18))
    qd_raw_to_policy = _detect_raw_to_policy_index(qd_raw_rows, raw_obs, np.zeros(12, dtype=np.float32), slice(18, 30))

    rows: List[ReplayStep] = []
    for i in range(len(raw_obs)):
        rows.append(
            ReplayStep(
                obs=raw_obs[i],
                action=raw_act[i],
                q_policy_rad=q_raw_rows[i][q_raw_to_policy],
                qd_policy_rad_s=qd_raw_rows[i][qd_raw_to_policy],
            )
        )
    return rows


def run_replay_test(
    *,
    log_path: Path,
    config_path: Path,
    policy_path: Path,
    report_path: Path,
    steps: int,
) -> Dict[str, float]:
    cfg = _load_config(config_path)
    q_ref = _extract_default_joint_pos_rad_from_cfg(cfg)
    if q_ref is None or q_ref.size < 12:
        raise RuntimeError("Could not extract init_state.joint_pos from config.")

    replay_rows = _load_replay_rows(log_path, max_steps=max(1, int(steps)), q_ref_rad=np.asarray(q_ref, dtype=np.float32))
    robot = LogReplayRobot(replay_rows)
    agent = RLAgent.from_files(
        robot,
        config_path=str(config_path),
        policy_path=str(policy_path),
        log_observation=False,
        log_action=False,
    )
    agent.spec.action_scale = 1.0
    # Prime one-step delayed observation buffers like runtime start().
    agent.obs_history.clear()
    agent._prev_q_rad = None
    agent._prev_q_t_s = None
    agent._prev_obs_joint_pos = (-agent._default_joint_pos_rad).astype(np.float32, copy=True)
    agent._prev_obs_joint_vel = np.zeros(12, dtype=np.float32)
    agent._curr_obs_joint_pos = None
    agent._curr_obs_joint_vel = None

    obs_err: List[float] = []
    act_err: List[float] = []
    cmd_err_vs_log_action: List[float] = []
    per_step: List[Dict[str, Any]] = []

    for i, row in enumerate(replay_rows):
        snap = robot.get_combined_state_snapshot(include_joint_state=True)
        obs_now = agent._build_obs_now(snap)
        obs_hist = agent._build_history_obs(obs_now)
        obs_in = agent._adapt_obs_dim_for_policy(obs_hist)
        act = agent.policy.infer(obs_in)
        agent._apply_action(act)

        exp_cmd_from_inferred = np.asarray(q_ref, dtype=np.float32).copy()
        n = min(12, act.size, len(agent.spec.action_scales_rad), len(agent.spec.encoder_bias_rad))
        exp_cmd_from_inferred[:n] += (
            act[:n] * np.asarray(agent.spec.action_scales_rad[:n], dtype=np.float32) * float(agent.spec.action_scale)
            - np.asarray(agent.spec.encoder_bias_rad[:n], dtype=np.float32)
        )
        exp_cmd_from_log = np.asarray(q_ref, dtype=np.float32).copy()
        exp_cmd_from_log[:n] += (
            row.action[:n]
            * np.asarray(agent.spec.action_scales_rad[:n], dtype=np.float32)
            * float(agent.spec.action_scale)
            - np.asarray(agent.spec.encoder_bias_rad[:n], dtype=np.float32)
        )

        e_obs = float(np.sqrt(np.mean((obs_in[:45] - row.obs[:45]) ** 2)))
        e_act = float(np.sqrt(np.mean((act[:12] - row.action[:12]) ** 2)))
        e_cmd_vs_log = float(np.sqrt(np.mean((robot.last_q_cmd_policy_rad[:12] - exp_cmd_from_log[:12]) ** 2)))
        obs_err.append(e_obs)
        act_err.append(e_act)
        cmd_err_vs_log_action.append(e_cmd_vs_log)
        per_step.append(
            {
                "step": i,
                "obs_rmse": e_obs,
                "action_rmse": e_act,
                "cmd_rmse_vs_log_action": e_cmd_vs_log,
                "obs_local": obs_in[:45].tolist(),
                "obs_log": row.obs[:45].tolist(),
                "action_local": act[:12].tolist(),
                "action_log": row.action[:12].tolist(),
                "cmd_local": robot.last_q_cmd_policy_rad[:12].tolist(),
                "cmd_formula_from_inferred_action": exp_cmd_from_inferred[:12].tolist(),
                "cmd_formula_from_log_action": exp_cmd_from_log[:12].tolist(),
            }
        )

    summary = {
        "steps": float(len(per_step)),
        "obs_rmse_mean": float(np.mean(obs_err)),
        "obs_rmse_max": float(np.max(obs_err)),
        "action_rmse_mean": float(np.mean(act_err)),
        "action_rmse_max": float(np.max(act_err)),
        "cmd_rmse_vs_log_action_mean": float(np.mean(cmd_err_vs_log_action)),
        "cmd_rmse_vs_log_action_max": float(np.max(cmd_err_vs_log_action)),
    }

    # Action->state proxy transform diagnosis from replay rows only.
    # proxy_q[t] is reconstructed from obs[t+1] because joint_pos obs is one-step delayed.
    if len(replay_rows) >= 3:
        actions = np.stack([r.action[:12] for r in replay_rows], axis=0)
        obs = np.stack([r.obs[:45] for r in replay_rows], axis=0)
        proxy_q = np.zeros((len(replay_rows), 12), dtype=np.float32)
        proxy_q[0] = np.asarray(q_ref, dtype=np.float32)
        proxy_q[1:] = obs[1:, 6:18] + np.asarray(q_ref, dtype=np.float32)[None, :]

        fit_lines: List[str] = []
        fit_lines.append("Action->proxy-state affine fit: proxy_q[t+lag] ~= gain * action[t] + offset")
        best_lag = 0
        best_mean_rmse = float("inf")
        for lag in (0, 1, 2):
            if len(replay_rows) - lag <= 2:
                continue
            x = actions[: len(replay_rows) - lag]
            y = proxy_q[lag:]
            gains: List[float] = []
            offsets: List[float] = []
            rmses: List[float] = []
            for j in range(12):
                xx = x[:, j]
                yy = y[:, j]
                m = np.vstack([xx, np.ones_like(xx)]).T
                g, o = np.linalg.lstsq(m, yy, rcond=None)[0]
                pred = g * xx + o
                rmse = float(np.sqrt(np.mean((pred - yy) ** 2)))
                gains.append(float(g))
                offsets.append(float(o))
                rmses.append(rmse)
            lag_mean_rmse = float(np.mean(rmses))
            if lag_mean_rmse < best_mean_rmse:
                best_mean_rmse = lag_mean_rmse
                best_lag = lag
                summary["fit_best_lag"] = float(lag)
                summary["fit_mean_rmse"] = lag_mean_rmse
                summary["fit_max_rmse"] = float(np.max(rmses))
                summary["fit_gain_mean"] = float(np.mean(gains))
                summary["fit_offset_mean"] = float(np.mean(offsets))
                summary["fit_gains"] = gains
                summary["fit_offsets"] = offsets
                summary["fit_rmses"] = rmses
        fit_lines.append(
            f"Best lag={best_lag}, mean_rmse={summary.get('fit_mean_rmse', 0.0):.6f}, "
            f"max_rmse={summary.get('fit_max_rmse', 0.0):.6f}"
        )
    else:
        fit_lines = ["Action->proxy-state affine fit: skipped (not enough rows)"]

    report_lines = [
        "RL_agent_isolated replay test (log-driven robot API)",
        f"log_path: {log_path}",
        f"config_path: {config_path}",
        f"policy_path: {policy_path}",
        "",
        "Summary:",
        json.dumps(summary, indent=2),
        "",
        *fit_lines,
        "",
        "First 3 steps:",
    ]
    for item in per_step[:3]:
        report_lines.append(
            f"step={item['step']} obs_rmse={item['obs_rmse']:.6f} "
            f"action_rmse={item['action_rmse']:.6f} cmd_rmse_vs_log_action={item['cmd_rmse_vs_log_action']:.6f}"
        )
    report_path.write_text("\n".join(report_lines))
    print("\n".join(report_lines[:20]))
    print(f"\nReport written to: {report_path}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay MJLab CSV through RL_agent_isolated with robot-like API mock.")
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    ap.add_argument("--steps", type=int, default=100000)
    args = ap.parse_args()

    run_replay_test(
        log_path=args.log,
        config_path=args.config,
        policy_path=args.policy,
        report_path=args.report,
        steps=max(1, int(args.steps)),
    )


if __name__ == "__main__":
    main()
