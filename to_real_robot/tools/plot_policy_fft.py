from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


JOINT_NAMES_6 = ["hipz", "hipx", "hipy", "knee", "ankle_pitch", "ankle_roll"]


def _parse_vec(text: str) -> np.ndarray:
    if text is None:
        return np.zeros(0, dtype=np.float32)
    s = str(text).strip()
    if not s:
        return np.zeros(0, dtype=np.float32)
    try:
        v = json.loads(s)
        return np.asarray(v, dtype=np.float32).reshape(-1)
    except Exception:
        return np.zeros(0, dtype=np.float32)


def _parse_cmd_indices(spec: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated indices, got: {spec}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _fft_amp(signal: np.ndarray, fs_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    n = int(x.size)
    if n < 4 or fs_hz <= 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    x = x - float(np.mean(x))
    win = np.hanning(n)
    xw = x * win
    spec = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    amp = (2.0 / max(1, n)) * np.abs(spec)
    return freqs, amp


def _load_policy_csv(
    policy_csv: Path,
    *,
    cmd_indices: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    times: List[float] = []
    actions: List[np.ndarray] = []
    commands: List[np.ndarray] = []

    with policy_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        has_action = "action_pre_scale" in cols or "action" in cols
        has_obs = "observation" in cols
        if not has_action:
            raise ValueError("CSV missing action column. Expected 'action_pre_scale' or 'action'.")
        if not has_obs:
            raise ValueError("CSV missing 'observation' column (needed for command extraction).")

        action_col = "action_pre_scale" if "action_pre_scale" in cols else "action"
        ci0, ci1, ci2 = cmd_indices

        for row in reader:
            try:
                t = float(row.get("time_s", "nan"))
            except Exception:
                continue
            act = _parse_vec(row.get(action_col, ""))
            obs = _parse_vec(row.get("observation", ""))
            if not np.isfinite(t) or act.size == 0 or obs.size == 0:
                continue

            if act.size < 12:
                act = np.pad(act, (0, 12 - act.size), mode="constant")
            else:
                act = act[:12]

            idxs = [ci0, ci1, ci2]
            cmd_vals = []
            ok = True
            for idx in idxs:
                ridx = idx if idx >= 0 else (obs.size + idx)
                if ridx < 0 or ridx >= obs.size:
                    ok = False
                    break
                cmd_vals.append(float(obs[ridx]))
            if not ok:
                cmd_vals = [0.0, 0.0, 0.0]

            times.append(t)
            actions.append(act.astype(np.float32, copy=False))
            commands.append(np.asarray(cmd_vals, dtype=np.float32))

    if not times:
        raise ValueError("No valid rows loaded from policy CSV.")

    t = np.asarray(times, dtype=np.float64)
    a = np.vstack(actions).astype(np.float64, copy=False)
    c = np.vstack(commands).astype(np.float64, copy=False)

    order = np.argsort(t)
    return t[order], a[order], c[order]


def plot_policy_fft(
    policy_csv: Path,
    *,
    cmd_indices: Tuple[int, int, int] = (-3, -2, -1),
    max_freq_hz: float = 20.0,
    include_command_trace: bool = True,
    out_png: Path | None = None,
) -> None:
    t, actions, cmd = _load_policy_csv(policy_csv, cmd_indices=cmd_indices)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 1e-6)]
    if dt.size == 0:
        raise ValueError("Cannot estimate sample rate from time_s.")
    fs = 1.0 / float(np.median(dt))

    t_rel = t - t[0]
    if include_command_trace:
        fig = plt.figure(figsize=(16, 20))
        gs = fig.add_gridspec(
            nrows=7,
            ncols=2,
            height_ratios=[1, 1, 1, 1, 1, 1, 1.4],
            hspace=0.4,
            wspace=0.22,
        )
    else:
        fig = plt.figure(figsize=(16, 16))
        gs = fig.add_gridspec(
            nrows=6,
            ncols=2,
            hspace=0.4,
            wspace=0.22,
        )
    fig.suptitle(
        f"Policy Action FFT + Command Trace\nfile={policy_csv.name} | fs~{fs:.2f} Hz | N={actions.shape[0]}",
        fontsize=13,
    )

    for j, joint in enumerate(JOINT_NAMES_6):
        ax_l = fig.add_subplot(gs[j, 0])
        ax_r = fig.add_subplot(gs[j, 1])

        f_l, a_l = _fft_amp(actions[:, j], fs_hz=fs)
        f_r, a_r = _fft_amp(actions[:, 6 + j], fs_hz=fs)

        if f_l.size:
            m_l = f_l <= max_freq_hz
            ax_l.plot(f_l[m_l], a_l[m_l], linewidth=1.2)
        if f_r.size:
            m_r = f_r <= max_freq_hz
            ax_r.plot(f_r[m_r], a_r[m_r], linewidth=1.2)

        ax_l.set_title(f"Left {joint}")
        ax_r.set_title(f"Right {joint}")
        ax_l.set_xlabel("Frequency [Hz]")
        ax_r.set_xlabel("Frequency [Hz]")
        ax_l.set_ylabel("Amplitude")
        ax_r.set_ylabel("Amplitude")
        ax_l.grid(True, alpha=0.3)
        ax_r.grid(True, alpha=0.3)

    if include_command_trace:
        ax_cmd = fig.add_subplot(gs[6, :])
        ax_cmd.plot(t_rel, cmd[:, 0], label="cmd_x", linewidth=1.2)
        ax_cmd.plot(t_rel, cmd[:, 1], label="cmd_y", linewidth=1.2)
        ax_cmd.plot(t_rel, cmd[:, 2], label="cmd_yaw", linewidth=1.2)
        ax_cmd.set_title("Command Components Over Time (from observation)")
        ax_cmd.set_xlabel("Time [s]")
        ax_cmd.set_ylabel("Command")
        ax_cmd.grid(True, alpha=0.3)
        ax_cmd.legend(loc="upper right")

    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_png}")
    else:
        plt.show()


def _main() -> None:
    p = argparse.ArgumentParser(description="Plot FFT of policy actions (12 joints) + command trace from observation.")
    p.add_argument("policy_csv", type=Path, help="Path to policy CSV log (from RL agent logger).")
    p.add_argument(
        "--cmd-indices",
        type=str,
        default="-3,-2,-1",
        help="Indices in observation vector for command x,y,yaw (default: last 3).",
    )
    p.add_argument("--max-freq-hz", type=float, default=20.0, help="Max frequency displayed for FFT plots.")
    p.add_argument(
        "--only-joints",
        action="store_true",
        help="Plot only the 12 joint FFT subplots (no command trace).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path. Default: <policy_csv_parent>/<policy_csv_stem>_fft.png",
    )
    args = p.parse_args()

    out_png = args.out
    if out_png is None:
        out_png = args.policy_csv.parent / f"{args.policy_csv.stem}_fft.png"

    plot_policy_fft(
        args.policy_csv,
        cmd_indices=_parse_cmd_indices(args.cmd_indices),
        max_freq_hz=float(max(0.1, args.max_freq_hz)),
        include_command_trace=not args.only_joints,
        out_png=out_png,
    )


if __name__ == "__main__":
    _main()
