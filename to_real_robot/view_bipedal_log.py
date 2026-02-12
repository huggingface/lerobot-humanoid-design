#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from root_constant import MOTORS


def _to_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def load_log(path: Path) -> Dict[str, np.ndarray]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in log: {path}")

    data: Dict[str, List[float]] = {}
    keys = list(rows[0].keys())
    for k in keys:
        data[k] = []
    for r in rows:
        for k in keys:
            data[k].append(r.get(k, ""))

    out: Dict[str, np.ndarray] = {}
    out["time_s"] = np.array([_to_float({"v": v}, "v") for v in data["time_s"]], dtype=float)
    out["mode_control"] = np.array([1.0 if v == "control" else 0.0 for v in data.get("mode", [])], dtype=float)
    out["estop"] = np.array([_to_float({"v": v}, "v") for v in data.get("estop", [])], dtype=float)
    out["damping_active"] = np.array([_to_float({"v": v}, "v") for v in data.get("damping_active", [])], dtype=float)

    for mid in range(1, 13):
        out[f"m{mid}_pos_deg"] = np.array([_to_float({"v": v}, "v") for v in data.get(f"m{mid}_pos_deg", [])], dtype=float)
        out[f"m{mid}_target_pos_deg"] = np.array(
            [_to_float({"v": v}, "v") for v in data.get(f"m{mid}_target_pos_deg", [])], dtype=float
        )
        out[f"m{mid}_vel_deg_s"] = np.array([_to_float({"v": v}, "v") for v in data.get(f"m{mid}_vel_deg_s", [])], dtype=float)
        out[f"m{mid}_tau_nm"] = np.array([_to_float({"v": v}, "v") for v in data.get(f"m{mid}_tau_nm", [])], dtype=float)
        out[f"m{mid}_temp_c"] = np.array([_to_float({"v": v}, "v") for v in data.get(f"m{mid}_temp_c", [])], dtype=float)

    out["missing_ids"] = np.array(data.get("missing_ids", []), dtype=object)
    out["mode"] = np.array(data.get("mode", []), dtype=object)
    out["estop_reason"] = np.array(data.get("estop_reason", []), dtype=object)
    return out


def print_summary(data: Dict[str, np.ndarray]) -> None:
    t = data["time_s"]
    dt = np.diff(t)
    dt = dt[dt > 0]
    hz = (1.0 / float(np.mean(dt))) if dt.size else 0.0
    print(f"rows: {len(t)}")
    print(f"time span: {t[0]:.3f} -> {t[-1]:.3f} ({t[-1] - t[0]:.2f}s)")
    if dt.size:
        print(f"loop dt: mean={np.mean(dt):.4f}s min={np.min(dt):.4f}s max={np.max(dt):.4f}s (~{hz:.2f} Hz)")
    print(f"control rows: {int(np.sum(data['mode_control']))}")
    print(f"damping rows: {int(np.sum(data['damping_active']))}")
    print(f"estop rows: {int(np.sum(data['estop']))}")
    missing = int(np.sum(np.array([1 if str(v).strip() else 0 for v in data["missing_ids"]], dtype=int)))
    print(f"rows with missing_ids: {missing}")

    for mid in range(1, 13):
        pos = data[f"m{mid}_pos_deg"]
        tgt = data[f"m{mid}_target_pos_deg"]
        err = tgt - pos
        print(
            f"m{mid:02d}: pos_span={np.max(pos)-np.min(pos):7.2f} deg | "
            f"mean_abs_err={np.mean(np.abs(err)):7.2f} deg | max_abs_err={np.max(np.abs(err)):7.2f} deg"
        )


def _save_fig(fig: plt.Figure, save_prefix: Path | None, suffix: str) -> None:
    if save_prefix is None:
        return
    out = save_prefix.with_name(f"{save_prefix.stem}_{suffix}.png")
    fig.savefig(out, dpi=140)
    print(f"saved figure: {out}")


def _overlay_mode_bands(ax, x: np.ndarray, data: Dict[str, np.ndarray]) -> None:
    # Light background hints to quickly spot state/control and damping windows.
    ctrl = data["mode_control"] > 0.5
    damp = data["damping_active"] > 0.5
    y0, y1 = ax.get_ylim()
    ax.fill_between(x, y0, y1, where=ctrl, color="#d9eefc", alpha=0.18, linewidth=0.0)
    ax.fill_between(x, y0, y1, where=damp, color="#ffd8a8", alpha=0.18, linewidth=0.0)


def plot_integrated_dashboard(data: Dict[str, np.ndarray], save_prefix: Path | None = None) -> None:
    t = data["time_s"]
    t0 = t[0]
    x = t - t0

    fig, axes = plt.subplots(6, 3, figsize=(18, 14), sharex=True)
    left_c = "#1565c0"
    right_c = "#c62828"
    tgt_alpha = 0.55
    for row in range(6):
        i = row + 1
        j = i + 6
        ax_pos, ax_tau, ax_tmp = axes[row, 0], axes[row, 1], axes[row, 2]
        show_col_legend = (row == 0)

        # Position: current + target
        ax_pos.plot(x, data[f"m{i}_pos_deg"], color=left_c, linewidth=1.2, label=("left cur" if show_col_legend else None))
        ax_pos.plot(x, data[f"m{j}_pos_deg"], color=right_c, linewidth=1.2, label=("right cur" if show_col_legend else None))
        ax_pos.plot(
            x,
            data[f"m{i}_target_pos_deg"],
            color=left_c,
            linestyle="--",
            alpha=tgt_alpha,
            linewidth=1.0,
            label=("left tgt" if show_col_legend else None),
        )
        ax_pos.plot(
            x,
            data[f"m{j}_target_pos_deg"],
            color=right_c,
            linestyle="--",
            alpha=tgt_alpha,
            linewidth=1.0,
            label=("right tgt" if show_col_legend else None),
        )
        ax_pos.set_ylabel(f"Pair {i}/{j}\nPos [deg]")
        ax_pos.grid(True, alpha=0.2)
        _overlay_mode_bands(ax_pos, x, data)

        # Torque: absolute, scaled by pair limit
        tau_i = np.abs(data[f"m{i}_tau_nm"])
        tau_j = np.abs(data[f"m{j}_tau_nm"])
        ax_tau.plot(x, tau_i, color=left_c, linewidth=1.2, label=("left" if show_col_legend else None))
        ax_tau.plot(x, tau_j, color=right_c, linewidth=1.2, label=("right" if show_col_legend else None))
        tmax = max(float(MOTORS[i].tmax_nm), float(MOTORS[j].tmax_nm))
        ax_tau.set_ylim(0.0, tmax)
        ax_tau.set_ylabel("|Tau| [Nm]")
        ax_tau.grid(True, alpha=0.2)
        _overlay_mode_bands(ax_tau, x, data)

        # Temperature
        ax_tmp.plot(x, data[f"m{i}_temp_c"], color=left_c, linewidth=1.2, label=("left" if show_col_legend else None))
        ax_tmp.plot(x, data[f"m{j}_temp_c"], color=right_c, linewidth=1.2, label=("right" if show_col_legend else None))
        ax_tmp.set_ylim(15.0, 70.0)
        ax_tmp.set_ylabel("Temp [°C]")
        ax_tmp.grid(True, alpha=0.2)
        _overlay_mode_bands(ax_tmp, x, data)

        if row == 0:
            ax_pos.set_title("Position (current + target)")
            ax_tau.set_title("Absolute Torque")
            ax_tmp.set_title("Temperature")
            ax_pos.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
            ax_tau.legend(loc="upper right", fontsize=8, framealpha=0.9)
            ax_tmp.legend(loc="upper right", fontsize=8, framealpha=0.9)

    for c in range(3):
        axes[-1, c].set_xlabel("Time [s]")

    fig.suptitle("Bipedal Log Dashboard", y=0.995, fontsize=16, fontweight="bold")
    fig.text(0.5, 1.01, "Blue: left leg  |  Red: right leg", ha="center", va="center", fontsize=10, color="#444444")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    _save_fig(fig, save_prefix, "dashboard")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="View and summarize bipedal robot CSV logs.")
    p.add_argument("--log", type=Path, default=Path("bipedal_state_log.csv"), help="path to CSV log")
    p.add_argument("--save-prefix", type=Path, default=None, help="optional output prefix for png files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = load_log(args.log)
    print_summary(data)
    with plt.style.context("seaborn-v0_8-whitegrid"):
        plot_integrated_dashboard(data, save_prefix=args.save_prefix)
        plt.show()


if __name__ == "__main__":
    main()
