from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from RL_agent import (
    _default_action_keys,
    _extract_default_joint_pos_rad_from_cfg,
    _load_config,
    infer_agent_spec,
)


JOINT_NAMES_12 = [
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


def _load_mjlab_rows(path: Path, nrows: int = 64) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            action = np.array([float(row[f"action_{k}"]) for k in range(12)], dtype=float)
            action_scale = None
            if "action_scale_0" in row:
                action_scale = np.array([float(row[f"action_scale_{k}"]) for k in range(12)], dtype=float)
            q = None
            if "robot_state_joint_pos_0" in row:
                q = np.array([float(row[f"robot_state_joint_pos_{k}"]) for k in range(12)], dtype=float)
            out.append(
                {
                    "action": action,
                    "action_scale": action_scale,
                    "q": q,
                    "row": i,
                }
            )
            if i + 1 >= nrows:
                break
    return out


def _load_local_policy_rows(path: Path, nrows: int = 64) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            out.append(
                {
                    "time_s": float(row["time_s"]),
                    "action": np.array(json.loads(row["action_pre_scale"]), dtype=float),
                    "row": i,
                }
            )
            if i + 1 >= nrows:
                break
    return out


def _load_local_trace(path: Path, nrows: int = 10000) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            q_deg = np.array([float(row[f"state_joint_deg_{k}"]) for k in range(12)], dtype=float)
            cmd_deg = np.array([float(row[f"cmd_joint_deg_{k}"]) for k in range(12)], dtype=float)
            out.append(
                {
                    "time_s": float(row["time_s"]),
                    "q_rad": np.deg2rad(q_deg),
                    "cmd_deg": cmd_deg,
                }
            )
            if i + 1 >= nrows:
                break
    return out


def _action_to_q_cmd_rad(
    action_raw: np.ndarray,
    action_keys: Sequence[str],
    action_scales_rad: Sequence[float],
    q_ref_rad: np.ndarray,
    action_scale_global: float = 1.0,
) -> np.ndarray:
    key_to_idx = {k: i for i, k in enumerate(_default_action_keys())}
    q_cmd = q_ref_rad.copy()
    n = min(len(action_raw), len(action_keys), len(action_scales_rad))
    for i in range(n):
        key = action_keys[i]
        idx = key_to_idx.get(key)
        if idx is None:
            continue
        q_cmd[idx] = q_ref_rad[idx] + float(action_raw[i]) * float(action_scales_rad[i]) * float(action_scale_global)
    return q_cmd


def _nearest_trace_index(trace: Sequence[Dict[str, object]], t: float) -> int:
    return min(range(len(trace)), key=lambda i: abs(float(trace[i]["time_s"]) - t))


def _find_first_cmd_match_index(trace: Sequence[Dict[str, object]], q_cmd_deg: np.ndarray, tol_deg: float = 1e-3) -> Optional[int]:
    for i, row in enumerate(trace):
        cmd = np.asarray(row["cmd_deg"], dtype=float)
        if float(np.max(np.abs(cmd - q_cmd_deg))) <= float(tol_deg):
            return i
    return None


def _fmt(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:+.4f}" for x in v.tolist()) + "]"


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare first-action requested/achieved joint positions: MJLab vs local sim.")
    ap.add_argument("--mjlab-log", required=True, type=Path)
    ap.add_argument("--sim-debug-log", default=Path("RL_policy/sim_robot_debug_ctrl.csv"), type=Path)
    ap.add_argument("--sim-trace-log", default=Path("RL_policy/sim_robot_trace.csv"), type=Path)
    ap.add_argument("--config", default=Path("RL_policy/config.yaml"), type=Path)
    ap.add_argument("--n-achieved", type=int, default=4)
    args = ap.parse_args()

    cfg = _load_config(args.config)
    spec = infer_agent_spec(cfg)
    q_ref = _extract_default_joint_pos_rad_from_cfg(cfg)
    if q_ref is None or q_ref.size < 12:
        raise RuntimeError("Could not load default 12-joint reference from config.")
    q_ref = q_ref[:12].astype(float, copy=True)

    mj = _load_mjlab_rows(args.mjlab_log, nrows=max(8, args.n_achieved + 2))
    sm_pol = _load_local_policy_rows(args.sim_debug_log, nrows=max(8, args.n_achieved + 2))
    sm_trace = _load_local_trace(args.sim_trace_log)
    if not mj or not sm_pol or not sm_trace:
        raise RuntimeError("One of the logs is empty.")

    mj_a0 = np.asarray(mj[0]["action"], dtype=float)
    mj_sc = mj[0]["action_scale"]
    if mj_sc is None:
        mj_sc_arr = np.asarray(spec.action_scales_rad[:12], dtype=float)
    else:
        mj_sc_arr = np.asarray(mj_sc, dtype=float)
    mj_q_cmd = _action_to_q_cmd_rad(mj_a0, spec.action_keys, mj_sc_arr, q_ref, action_scale_global=1.0)

    sm_a0 = np.asarray(sm_pol[0]["action"], dtype=float)
    sm_q_cmd = _action_to_q_cmd_rad(sm_a0, spec.action_keys, spec.action_scales_rad, q_ref, action_scale_global=1.0)
    sm_q_cmd_deg = np.rad2deg(sm_q_cmd)

    # MJLab achieved (policy-rate only, from robot_state_joint_pos if available).
    mj_achieved: List[np.ndarray] = []
    for i in range(1, min(len(mj), args.n_achieved + 1)):
        q = mj[i]["q"]
        if q is None:
            break
        mj_achieved.append(np.asarray(q, dtype=float))

    # Local achieved: find first trace row where command equals first command target, then next N rows.
    start_idx = _find_first_cmd_match_index(sm_trace, sm_q_cmd_deg, tol_deg=1e-2)
    if start_idx is None:
        # Fallback: nearest to first policy timestamp.
        start_idx = _nearest_trace_index(sm_trace, float(sm_pol[0]["time_s"]))
    sm_achieved: List[np.ndarray] = []
    for k in range(1, args.n_achieved + 1):
        j = start_idx + k
        if j >= len(sm_trace):
            break
        sm_achieved.append(np.asarray(sm_trace[j]["q_rad"], dtype=float))

    print("=== First Action Window ===")
    print(f"MJLab log: {args.mjlab_log}")
    print(f"Local policy log: {args.sim_debug_log}")
    print(f"Local trace log: {args.sim_trace_log}")
    print("")
    print("First requested joint target q_cmd (rad):")
    print("MJLab:", _fmt(mj_q_cmd))
    print("Local:", _fmt(sm_q_cmd))
    print("mean|Δ|:", float(np.mean(np.abs(mj_q_cmd - sm_q_cmd))))
    print("")

    print(f"Next {args.n_achieved} achieved joint positions (rad):")
    print("MJLab (policy-rate robot_state_joint_pos):")
    for i, q in enumerate(mj_achieved, start=1):
        print(f"  t+{i}: {_fmt(q)}")
    print("Local (sim trace after first cmd application):")
    print(f"  start_trace_idx={start_idx}")
    for i, q in enumerate(sm_achieved, start=1):
        print(f"  t+{i}: {_fmt(q)}")
    print("")

    # Per-joint compact summary.
    print("Per-joint summary (first achieved sample only):")
    if mj_achieved and sm_achieved:
        mj1, sm1 = mj_achieved[0], sm_achieved[0]
        print("joint, q_cmd_mj, q_cmd_sm, q1_mj, q1_sm, dq_to_cmd_mj, dq_to_cmd_sm")
        for j, name in enumerate(JOINT_NAMES_12):
            dq_mj = mj_q_cmd[j] - mj1[j]
            dq_sm = sm_q_cmd[j] - sm1[j]
            print(
                f"{name},"
                f"{mj_q_cmd[j]:+.6f},{sm_q_cmd[j]:+.6f},"
                f"{mj1[j]:+.6f},{sm1[j]:+.6f},"
                f"{dq_mj:+.6f},{dq_sm:+.6f}"
            )


if __name__ == "__main__":
    main()
