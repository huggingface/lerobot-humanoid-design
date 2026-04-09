from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from bipedal_robot import BipedalRobotController


HIP_JOINTS: tuple[str, ...] = ("hipz", "hipx", "hipy")
HIP_COLUMN_KEYS: tuple[str, ...] = (
    "left_hipz",
    "left_hipx",
    "left_hipy",
    "right_hipz",
    "right_hipx",
    "right_hipy",
)


def _joint_array_to_command(q_deg: Sequence[float]) -> tuple[Dict[str, float], Dict[str, float]]:
    q = np.asarray(q_deg, dtype=float).reshape(-1)
    if q.size < 12:
        raise ValueError(f"Expected 12 joints, got {q.size}")
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


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return False


def _extract_imu_gyro(imu_state: Any) -> tuple[float, float, float]:
    if not isinstance(imu_state, dict):
        return (float("nan"), float("nan"), float("nan"))
    for key in ("gyro_rads", "ang_vel_rad_s"):
        raw = imu_state.get(key)
        if isinstance(raw, (list, tuple)) and len(raw) >= 3:
            return (_safe_float(raw[0]), _safe_float(raw[1]), _safe_float(raw[2]))
    return (float("nan"), float("nan"), float("nan"))


def _phase_name(t_rel_s: float, *, hold_s: float, duration_s: float, settle_s: float) -> str:
    if t_rel_s < hold_s:
        return "hold"
    if t_rel_s < hold_s + duration_s:
        return "oscillate"
    if t_rel_s < hold_s + duration_s + settle_s:
        return "settle"
    return "done"


def _projected_gravity_from_quat_xyzw(quat_xyzw: Sequence[float]) -> tuple[float, float, float]:
    x, y, z, w = [float(v) for v in quat_xyzw]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rot = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=float,
    )
    projected = rot.T @ np.array([0.0, 0.0, -1.0], dtype=float)
    return (float(projected[0]), float(projected[1]), float(projected[2]))


def run_hip_imu_debug(
    robot: "BipedalRobotController",
    *,
    joint: str = "hipx",
    amplitude_deg: float = 3.0,
    frequency_hz: float = 0.5,
    duration_s: float = 10.0,
    hold_s: float = 1.0,
    settle_s: float = 1.0,
    command_hz: float = 100.0,
    log_path: str | Path = "debug_logs/hip_imu_debug.csv",
    prepare_control: bool = False,
    enable_motors: bool = False,
) -> Path:
    """
    Drive one hip joint in opposite phase on left/right and log a focused CSV.

    Safety defaults are conservative:
      - Uses the current measured pose as the base pose.
      - Refuses to switch out of state_only unless prepare_control=True.
      - Does not auto-enable motors unless enable_motors=True.
    """
    joint = str(joint).strip().lower()
    if joint not in HIP_JOINTS:
        raise ValueError(f"joint must be one of {HIP_JOINTS}, got {joint!r}")
    if amplitude_deg < 0.0:
        raise ValueError("amplitude_deg must be >= 0")
    if frequency_hz < 0.0:
        raise ValueError("frequency_hz must be >= 0")
    if duration_s <= 0.0:
        raise ValueError("duration_s must be > 0")
    if command_hz <= 0.0:
        raise ValueError("command_hz must be > 0")

    robot.request_state_once()
    snap0 = robot.get_combined_state_snapshot(include_joint_state=True)
    q_base_deg = np.asarray(snap0.get("joint_state_deg", [0.0] * 12), dtype=float).reshape(-1)
    if q_base_deg.size < 12:
        raise RuntimeError(f"Expected 12 joint states from robot, got {q_base_deg.size}")

    if getattr(robot, "mode", None) != "control":
        if not prepare_control:
            raise RuntimeError(
                "Robot is not in control mode. Set robot.set_mode('control') first, "
                "or call run_hip_imu_debug(..., prepare_control=True)."
            )
        robot.set_mode("control")
        if enable_motors:
            robot.enable_all()
        time.sleep(0.2)

    base_left, base_right = _joint_array_to_command(q_base_deg)
    total_s = float(hold_s + duration_s + settle_s)
    period_s = 1.0 / float(command_hz)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: List[str] = [
        "time_s",
        "t_rel_s",
        "phase",
        "command_joint",
        "command_amplitude_deg",
        "command_frequency_hz",
        "command_delta_deg",
        f"target_left_{joint}_deg",
        f"target_right_{joint}_deg",
        "orientation_timestamp_s",
        "orientation_quat_x",
        "orientation_quat_y",
        "orientation_quat_z",
        "orientation_quat_w",
        "projected_gravity_x",
        "projected_gravity_y",
        "projected_gravity_z",
        "imu_gyro_x_rad_s",
        "imu_gyro_y_rad_s",
        "imu_gyro_z_rad_s",
        "imu_error",
        "estop",
        "estop_reason",
    ]
    for key in HIP_COLUMN_KEYS:
        fieldnames.append(f"{key}_pos_deg")
    for key in HIP_COLUMN_KEYS:
        fieldnames.append(f"{key}_vel_rad_s")

    t_start = time.perf_counter()
    next_tick = t_start

    with log_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()

        while True:
            now = time.perf_counter()
            t_rel_s = now - t_start
            if t_rel_s > total_s:
                break

            phase = _phase_name(t_rel_s, hold_s=hold_s, duration_s=duration_s, settle_s=settle_s)
            if phase == "oscillate":
                osc_t = t_rel_s - hold_s
                delta_deg = float(amplitude_deg) * math.sin(2.0 * math.pi * float(frequency_hz) * osc_t)
            else:
                delta_deg = 0.0

            left_cmd = dict(base_left)
            right_cmd = dict(base_right)
            left_cmd[joint] = float(base_left[joint] + delta_deg)
            right_cmd[joint] = float(base_right[joint] - delta_deg)
            robot.set_action(left=left_cmd, right=right_cmd)

            snap = robot.get_combined_state_snapshot(include_joint_state=True)
            q_deg = np.asarray(snap.get("joint_state_deg", [0.0] * 12), dtype=float).reshape(-1)
            qd_rad_s = np.asarray(snap.get("joint_velocity_rad_s", [0.0] * 12), dtype=float).reshape(-1)
            imu_state = snap.get("imu") or {}
            quat = snap.get("orientation_quaternion_xyzw")
            gyro = _extract_imu_gyro(imu_state)
            projected_gravity = (None, None, None)
            if isinstance(quat, (list, tuple)) and len(quat) == 4:
                projected_gravity = _projected_gravity_from_quat_xyzw(tuple(float(v) for v in quat))

            row: Dict[str, Any] = {
                "time_s": _safe_float(snap.get("time_s", time.time())),
                "t_rel_s": float(t_rel_s),
                "phase": phase,
                "command_joint": joint,
                "command_amplitude_deg": float(amplitude_deg),
                "command_frequency_hz": float(frequency_hz),
                "command_delta_deg": float(delta_deg),
                f"target_left_{joint}_deg": float(left_cmd[joint]),
                f"target_right_{joint}_deg": float(right_cmd[joint]),
                "orientation_timestamp_s": _safe_float(snap.get("orientation_timestamp_s", float("nan"))),
                "orientation_quat_x": _safe_float(quat[0]) if isinstance(quat, (list, tuple)) and len(quat) == 4 else "",
                "orientation_quat_y": _safe_float(quat[1]) if isinstance(quat, (list, tuple)) and len(quat) == 4 else "",
                "orientation_quat_z": _safe_float(quat[2]) if isinstance(quat, (list, tuple)) and len(quat) == 4 else "",
                "orientation_quat_w": _safe_float(quat[3]) if isinstance(quat, (list, tuple)) and len(quat) == 4 else "",
                "projected_gravity_x": projected_gravity[0] if projected_gravity[0] is not None else "",
                "projected_gravity_y": projected_gravity[1] if projected_gravity[1] is not None else "",
                "projected_gravity_z": projected_gravity[2] if projected_gravity[2] is not None else "",
                "imu_gyro_x_rad_s": gyro[0],
                "imu_gyro_y_rad_s": gyro[1],
                "imu_gyro_z_rad_s": gyro[2],
                "imu_error": "" if not isinstance(imu_state, dict) else str(imu_state.get("error", "")),
                "estop": int(_safe_bool(snap.get("estop", False))),
                "estop_reason": str(snap.get("estop_reason", "")),
            }
            hip_pos = {
                "left_hipz": q_deg[0],
                "left_hipx": q_deg[1],
                "left_hipy": q_deg[2],
                "right_hipz": q_deg[6],
                "right_hipx": q_deg[7],
                "right_hipy": q_deg[8],
            }
            hip_vel = {
                "left_hipz": qd_rad_s[0],
                "left_hipx": qd_rad_s[1],
                "left_hipy": qd_rad_s[2],
                "right_hipz": qd_rad_s[6],
                "right_hipx": qd_rad_s[7],
                "right_hipy": qd_rad_s[8],
            }
            for key, value in hip_pos.items():
                row[f"{key}_pos_deg"] = float(value)
            for key, value in hip_vel.items():
                row[f"{key}_vel_rad_s"] = float(value)
            writer.writerow(row)
            fp.flush()

            next_tick += period_s
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_tick = time.perf_counter()

    robot.set_action(left=base_left, right=base_right)
    return log_path


def _load_debug_csv(log_path: str | Path) -> tuple[List[Dict[str, str]], np.ndarray]:
    rows = list(csv.DictReader(Path(log_path).open()))
    if not rows:
        raise RuntimeError(f"No rows found in {log_path}")
    t_rel = np.array([_safe_float(row.get("t_rel_s", float("nan"))) for row in rows], dtype=float)
    return rows, t_rel


def _column(rows: Sequence[Dict[str, str]], key: str) -> np.ndarray:
    return np.array([_safe_float(row.get(key, float("nan"))) for row in rows], dtype=float)


def _first_finite(values: Iterable[float], default: float = float("nan")) -> float:
    for value in values:
        if np.isfinite(value):
            return float(value)
    return float(default)


def _nan_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(np.count_nonzero(mask)) < 3:
        return float("nan")
    aa = a[mask]
    bb = b[mask]
    if float(np.std(aa)) < 1e-9 or float(np.std(bb)) < 1e-9:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def _nanmean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _nanstd(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.std(finite))


def _nanrms(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(finite ** 2)))


def summarize_hip_imu_debug(log_path: str | Path) -> Dict[str, float]:
    rows, t_rel = _load_debug_csv(log_path)
    t = np.asarray(t_rel, dtype=float)
    dt = np.diff(t)
    sample_hz = float("nan")
    if dt.size > 0:
        finite_dt = dt[np.isfinite(dt) & (dt > 0.0)]
        if finite_dt.size > 0:
            sample_hz = float(1.0 / np.median(finite_dt))

    joint = rows[0].get("command_joint", "hipx")
    left_target = _column(rows, f"target_left_{joint}_deg")
    right_target = _column(rows, f"target_right_{joint}_deg")
    left_meas = _column(rows, f"left_{joint}_pos_deg")
    right_meas = _column(rows, f"right_{joint}_pos_deg")

    quat_x = _column(rows, "orientation_quat_x")
    quat_y = _column(rows, "orientation_quat_y")
    quat_z = _column(rows, "orientation_quat_z")
    quat_w = _column(rows, "orientation_quat_w")
    quat_norm = np.sqrt(quat_x ** 2 + quat_y ** 2 + quat_z ** 2 + quat_w ** 2)

    pg_x = _column(rows, "projected_gravity_x")
    pg_y = _column(rows, "projected_gravity_y")
    pg_z = _column(rows, "projected_gravity_z")
    pg_norm = np.sqrt(pg_x ** 2 + pg_y ** 2 + pg_z ** 2)

    mean_vel = 0.5 * (
        _column(rows, f"left_{joint}_vel_rad_s") - _column(rows, f"right_{joint}_vel_rad_s")
    )
    gyro_x = _column(rows, "imu_gyro_x_rad_s")
    gyro_y = _column(rows, "imu_gyro_y_rad_s")
    gyro_z = _column(rows, "imu_gyro_z_rad_s")

    stats: Dict[str, float] = {
        "duration_s": float(t[-1] - t[0]) if t.size > 1 else 0.0,
        "sample_hz": sample_hz,
        "quat_norm_mean": _nanmean(quat_norm),
        "quat_norm_std": _nanstd(quat_norm),
        "projected_gravity_norm_mean": _nanmean(pg_norm),
        "projected_gravity_norm_std": _nanstd(pg_norm),
        "left_target_meas_corr": _nan_corr(left_target, left_meas),
        "right_target_meas_corr": _nan_corr(right_target, right_meas),
        "joint_vel_vs_gyro_x_corr": _nan_corr(mean_vel, gyro_x),
        "joint_vel_vs_gyro_y_corr": _nan_corr(mean_vel, gyro_y),
        "joint_vel_vs_gyro_z_corr": _nan_corr(mean_vel, gyro_z),
        "imu_gyro_x_rms": _nanrms(gyro_x),
        "imu_gyro_y_rms": _nanrms(gyro_y),
        "imu_gyro_z_rms": _nanrms(gyro_z),
    }
    return stats


def plot_hip_imu_debug(
    log_path: str | Path,
    *,
    out_path: Optional[str | Path] = None,
    show: bool = False,
) -> tuple[Path, Dict[str, float]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, t_rel = _load_debug_csv(log_path)
    joint = rows[0].get("command_joint", "hipx")

    left_target = _column(rows, f"target_left_{joint}_deg")
    right_target = _column(rows, f"target_right_{joint}_deg")
    left_meas = _column(rows, f"left_{joint}_pos_deg")
    right_meas = _column(rows, f"right_{joint}_pos_deg")

    phase = [row.get("phase", "") for row in rows]
    osc_start = _first_finite(
        [float(t) for t, ph in zip(t_rel, phase) if ph == "oscillate"],
        default=float("nan"),
    )
    osc_end = _first_finite(
        reversed([float(t) for t, ph in zip(t_rel, phase) if ph == "oscillate"]),
        default=float("nan"),
    )

    all_hip_pos = {f"{key}_pos_deg": _column(rows, f"{key}_pos_deg") for key in HIP_COLUMN_KEYS}
    all_hip_vel = {f"{key}_vel_rad_s": _column(rows, f"{key}_vel_rad_s") for key in HIP_COLUMN_KEYS}
    gyro_cols = {
        "imu_gyro_x_rad_s": _column(rows, "imu_gyro_x_rad_s"),
        "imu_gyro_y_rad_s": _column(rows, "imu_gyro_y_rad_s"),
        "imu_gyro_z_rad_s": _column(rows, "imu_gyro_z_rad_s"),
    }
    pg_cols = {
        "projected_gravity_x": _column(rows, "projected_gravity_x"),
        "projected_gravity_y": _column(rows, "projected_gravity_y"),
        "projected_gravity_z": _column(rows, "projected_gravity_z"),
    }
    quat_cols = {
        "orientation_quat_x": _column(rows, "orientation_quat_x"),
        "orientation_quat_y": _column(rows, "orientation_quat_y"),
        "orientation_quat_z": _column(rows, "orientation_quat_z"),
        "orientation_quat_w": _column(rows, "orientation_quat_w"),
    }
    has_pg = any(np.any(np.isfinite(values)) for values in pg_cols.values())

    nrows = 6 if has_pg else 5
    fig, axes = plt.subplots(nrows, 1, figsize=(14, 2.7 * nrows), sharex=True, constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    def _decorate_axis(ax: Any, title: str, ylabel: str) -> None:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if np.isfinite(osc_start):
            ax.axvline(float(osc_start), color="0.5", linestyle="--", linewidth=1.0)
        if np.isfinite(osc_end):
            ax.axvline(float(osc_end), color="0.5", linestyle="--", linewidth=1.0)

    axes[0].plot(t_rel, left_target, label=f"left {joint} target", color="tab:blue", linestyle="--")
    axes[0].plot(t_rel, left_meas, label=f"left {joint} meas", color="tab:blue")
    axes[0].plot(t_rel, right_target, label=f"right {joint} target", color="tab:orange", linestyle="--")
    axes[0].plot(t_rel, right_meas, label=f"right {joint} meas", color="tab:orange")
    _decorate_axis(axes[0], f"Opposite-Phase {joint.upper()} Command", "deg")
    axes[0].legend(loc="upper right", ncol=2)

    for key, values in all_hip_pos.items():
        rel = values - values[0] if np.any(np.isfinite(values)) else values
        axes[1].plot(t_rel, rel, label=key.replace("_pos_deg", ""))
    _decorate_axis(axes[1], "All Hip Position Deviations", "deg")
    axes[1].legend(loc="upper right", ncol=3, fontsize=8)

    for key, values in all_hip_vel.items():
        axes[2].plot(t_rel, values, label=key.replace("_vel_rad_s", ""))
    _decorate_axis(axes[2], "All Hip Velocities", "rad/s")
    axes[2].legend(loc="upper right", ncol=3, fontsize=8)

    for key, values in gyro_cols.items():
        axes[3].plot(t_rel, values, label=key.replace("imu_gyro_", "gyro_").replace("_rad_s", ""))
    _decorate_axis(axes[3], "IMU Angular Velocity", "rad/s")
    axes[3].legend(loc="upper right")

    next_axis = 4
    if has_pg:
        for key, values in pg_cols.items():
            axes[next_axis].plot(t_rel, values, label=key.replace("projected_gravity_", "g_"))
        _decorate_axis(axes[next_axis], "Projected Gravity", "unitless")
        axes[next_axis].legend(loc="upper right")
        next_axis += 1

    for key, values in quat_cols.items():
        axes[next_axis].plot(t_rel, values, label=key.replace("orientation_quat_", "q_"))
    quat_norm = np.sqrt(sum(values ** 2 for values in quat_cols.values()))
    axes[next_axis].plot(t_rel, quat_norm, label="|q|", color="black", linewidth=1.2, alpha=0.8)
    _decorate_axis(axes[next_axis], "Quaternion", "unitless")
    axes[next_axis].legend(loc="upper right")
    axes[next_axis].set_xlabel("t_rel_s")

    out_path = Path(out_path) if out_path is not None else Path(log_path).with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)

    stats = summarize_hip_imu_debug(log_path)
    print(f"[hip_imu_debug] wrote plot: {out_path}")
    print(
        "[hip_imu_debug] summary: "
        f"duration_s={stats['duration_s']:.3f}, sample_hz={stats['sample_hz']:.2f}, "
        f"|q| mean={stats['quat_norm_mean']:.5f} std={stats['quat_norm_std']:.5f}, "
        f"|g_proj| mean={stats['projected_gravity_norm_mean']:.5f} std={stats['projected_gravity_norm_std']:.5f}"
    )
    print(
        "[hip_imu_debug] tracking corr: "
        f"left={stats['left_target_meas_corr']:.3f}, right={stats['right_target_meas_corr']:.3f}"
    )
    print(
        "[hip_imu_debug] gyro rms: "
        f"x={stats['imu_gyro_x_rms']:.4f}, y={stats['imu_gyro_y_rms']:.4f}, z={stats['imu_gyro_z_rms']:.4f}"
    )
    print(
        "[hip_imu_debug] joint-vel vs gyro corr: "
        f"x={stats['joint_vel_vs_gyro_x_corr']:.3f}, "
        f"y={stats['joint_vel_vs_gyro_y_corr']:.3f}, "
        f"z={stats['joint_vel_vs_gyro_z_corr']:.3f}"
    )
    return out_path, stats


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot a focused hip/IMU debug CSV.")
    parser.add_argument("log_path", help="CSV written by run_hip_imu_debug(...)")
    parser.add_argument("--out", dest="out_path", default=None, help="Optional output PNG path")
    parser.add_argument("--show", action="store_true", help="Display the figure after saving")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    plot_hip_imu_debug(args.log_path, out_path=args.out_path, show=bool(args.show))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
