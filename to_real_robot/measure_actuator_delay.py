from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np

from bipedal_robot import BipedalRobotController
from mock_bus import MockBus
from root_constant import MOTOR_IDS


JOINT_ORDER = (
    "left_hipz",
    "left_hipx",
    "left_hipy",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hipz",
    "right_hipx",
    "right_hipy",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
)
JOINT_INDEX = {name: idx for idx, name in enumerate(JOINT_ORDER)}
SIDE_JOINT_KEYS = ("hipz", "hipx", "hipy", "knee", "ankle_pitch", "ankle_roll")
MetricValue = Union[int, float]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Measure actuator/joint delay on BipedalRobotController using a sinusoidal "
            "joint-space command and the measured joint observation."
        )
    )
    p.add_argument("--joint", type=str, default="left_knee", choices=JOINT_ORDER)
    p.add_argument("--all-joints", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--fps", type=int, default=100)
    p.add_argument("--duration-s", type=float, default=2.0)
    p.add_argument("--freq-hz", type=float, default=1.0)
    p.add_argument("--amp-deg", type=float, default=10.0)
    p.add_argument("--pre-roll-s", type=float, default=1.0)
    p.add_argument("--between-s", type=float, default=0.5)
    p.add_argument("--save-csv", type=str, default="")
    p.add_argument("--save-png", type=str, default="")
    p.add_argument("--use-mock-bus", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--control-hz", type=float, default=200.0)
    p.add_argument("--recv-timeout-s", type=float, default=0.001)
    p.add_argument("--max-command-delta-deg", type=float, default=60.0)
    p.add_argument("--disable-command-limits", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--can0-port", type=str, default="can0")
    p.add_argument("--can1-port", type=str, default="can1")
    p.add_argument("--log-path", type=str, default="bipedal_state_log.csv")
    return p.parse_args()


def _joint_vector_to_action(q_deg: Sequence[float]) -> tuple[dict[str, float], dict[str, float]]:
    q = np.asarray(q_deg, dtype=float).reshape(-1)
    if q.size < len(JOINT_ORDER):
        raise ValueError(f"Expected at least {len(JOINT_ORDER)} joint values, got {q.size}")

    left = {key: float(q[idx]) for idx, key in enumerate(SIDE_JOINT_KEYS)}
    right = {key: float(q[idx + 6]) for idx, key in enumerate(SIDE_JOINT_KEYS)}
    return left, right


def _send_joint_vector(robot: BipedalRobotController, q_deg: Sequence[float]) -> None:
    left, right = _joint_vector_to_action(q_deg)
    robot.set_action(left=left, right=right)


def _capture_joint_state_deg(robot: BipedalRobotController) -> np.ndarray:
    snap = robot.get_combined_state_snapshot(include_joint_state=True)
    q_deg = np.asarray(snap.get("joint_state_deg", []), dtype=float).reshape(-1)
    if q_deg.size < len(JOINT_ORDER):
        raise RuntimeError(
            f"Joint-state snapshot has size {q_deg.size}, expected at least {len(JOINT_ORDER)}."
        )
    q_deg = q_deg[: len(JOINT_ORDER)].copy()
    if not np.all(np.isfinite(q_deg)):
        raise RuntimeError("Joint-state snapshot contains non-finite values.")
    return q_deg


def _ensure_valid_state(
    robot: BipedalRobotController,
    *,
    retries: int = 5,
    sleep_s: float = 0.05,
) -> None:
    wait_s = max(float(sleep_s), 1.0 / max(1.0, float(robot.control_hz)))
    for _ in range(max(1, int(retries))):
        state = robot.get_state_snapshot()
        if state and all(float(st.stamp) > 0.0 for st in state.values()):
            _capture_joint_state_deg(robot)
            return
        robot.request_state_once()
        time.sleep(wait_s)
    raise RuntimeError("No valid motor state available. Check feedback and E-STOP status.")


def _estimate_delay_xcorr(
    *,
    cmd_deg: np.ndarray,
    obs_deg: np.ndarray,
    dt_s: float,
) -> tuple[int, float, float]:
    cmd0 = np.asarray(cmd_deg, dtype=float) - float(np.mean(cmd_deg))
    obs0 = np.asarray(obs_deg, dtype=float) - float(np.mean(obs_deg))
    denom = float(np.linalg.norm(cmd0) * np.linalg.norm(obs0))
    if denom < 1e-9:
        return 0, float("nan"), float("nan")

    corr = np.correlate(obs0, cmd0, mode="full")
    lags = np.arange(-len(cmd0) + 1, len(cmd0), dtype=int)
    i = int(np.argmax(corr))
    lag_samples = int(lags[i])
    delay_s = float(lag_samples) * float(dt_s)
    corr_coeff = float(corr[i] / denom)
    return lag_samples, delay_s, corr_coeff


def _estimate_delay_phase(
    *,
    t_s: np.ndarray,
    obs_deg: np.ndarray,
    freq_hz: float,
) -> tuple[float, float, float, float]:
    omega = 2.0 * math.pi * float(freq_hz)
    y = np.asarray(obs_deg, dtype=float)
    t = np.asarray(t_s, dtype=float)
    if y.size < 3 or abs(omega) < 1e-12:
        return float("nan"), float("nan"), float("nan"), float("nan")

    s = np.sin(omega * t)
    c = np.cos(omega * t)
    X = np.column_stack([s, c, np.ones_like(t)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b, offset = float(beta[0]), float(beta[1]), float(beta[2])
    amp = float(math.hypot(a, b))
    phi = float(math.atan2(b, a))
    raw_delay_s = float(-phi / omega)
    period_s = 1.0 / float(freq_hz)
    wrapped_delay_s = ((raw_delay_s + 0.5 * period_s) % period_s) - 0.5 * period_s
    return raw_delay_s, wrapped_delay_s, amp, offset


def _analyze_trace(
    *,
    t_log: list[float],
    cmd_log: list[float],
    obs_log: list[float],
    fps: int,
    freq_hz: float,
    center_deg: float,
) -> tuple[dict[str, MetricValue], np.ndarray, np.ndarray, np.ndarray]:
    t_arr = np.asarray(t_log, dtype=float)
    cmd_arr = np.asarray(cmd_log, dtype=float)
    obs_arr = np.asarray(obs_log, dtype=float)
    valid = np.isfinite(t_arr) & np.isfinite(cmd_arr) & np.isfinite(obs_arr)
    if not np.any(valid):
        raise RuntimeError("No valid samples collected.")

    t_arr = t_arr[valid]
    cmd_arr = cmd_arr[valid]
    obs_arr = obs_arr[valid]

    if t_arr.size > 1:
        dt_eff = float(np.median(np.diff(t_arr)))
    else:
        dt_eff = 1.0 / float(fps)
    achieved_hz_dt = (1.0 / dt_eff) if dt_eff > 1e-12 else float("nan")
    if t_arr.size > 1:
        span_s = float(t_arr[-1] - t_arr[0])
        achieved_hz_avg = float((t_arr.size - 1) / span_s) if span_s > 1e-12 else float("nan")
    else:
        achieved_hz_avg = float("nan")

    lag_samples, delay_s_xcorr, corr_coeff = _estimate_delay_xcorr(
        cmd_deg=cmd_arr,
        obs_deg=obs_arr,
        dt_s=dt_eff,
    )
    raw_delay_s_phase, wrapped_delay_s_phase, amp_fit, offset_fit = _estimate_delay_phase(
        t_s=t_arr,
        obs_deg=obs_arr,
        freq_hz=float(freq_hz),
    )
    metrics: dict[str, MetricValue] = {
        "samples": int(t_arr.size),
        "target_hz": float(fps),
        "dt_eff_s": float(dt_eff),
        "achieved_hz_dt": float(achieved_hz_dt),
        "achieved_hz_avg": float(achieved_hz_avg),
        "lag_samples": int(lag_samples),
        "delay_s_xcorr": float(delay_s_xcorr),
        "peak_corr": float(corr_coeff),
        "delay_s_phase_raw": float(raw_delay_s_phase),
        "delay_s_phase_wrapped": float(wrapped_delay_s_phase),
        "fit_amp_deg": float(amp_fit),
        "fit_offset_deg": float(offset_fit),
        "center_deg": float(center_deg),
    }
    return metrics, t_arr, cmd_arr, obs_arr


def _print_delay_summary(joint: str, metrics: dict[str, MetricValue]) -> None:
    print("\n[delay estimate]")
    print(f"  joint: {joint}")
    print(
        f"  samples: {int(metrics['samples'])} "
        f"(dt_eff={float(metrics['dt_eff_s']):.6f}s, center={float(metrics['center_deg']):+.4f}deg)"
    )
    print(
        f"  loop freq: target={float(metrics['target_hz']):.2f}Hz "
        f"achieved_dt={float(metrics['achieved_hz_dt']):.2f}Hz "
        f"achieved_avg={float(metrics['achieved_hz_avg']):.2f}Hz"
    )
    print(
        f"  xcorr lag: {int(metrics['lag_samples']):+d} samples  "
        f"({float(metrics['delay_s_xcorr']):+.6f}s), peak_corr={float(metrics['peak_corr']):.4f}"
    )
    print(f"  phase delay raw: {float(metrics['delay_s_phase_raw']):+.6f}s")
    print(f"  phase delay wrapped[-T/2,T/2): {float(metrics['delay_s_phase_wrapped']):+.6f}s")
    print(
        f"  fitted obs sinus amp={float(metrics['fit_amp_deg']):.4f}deg "
        f"offset={float(metrics['fit_offset_deg']):+.4f}deg"
    )
    if abs(float(metrics["fit_amp_deg"])) < 0.2:
        print("  [warn] very low observed amplitude; actuator may be disconnected or not tracking.")


def _save_trace_csv(path: Path, t_arr: np.ndarray, cmd_arr: np.ndarray, obs_arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "cmd_deg", "obs_deg"])
        for t, c, o in zip(t_arr.tolist(), cmd_arr.tolist(), obs_arr.tolist()):
            w.writerow([f"{t:.9f}", f"{c:.9f}", f"{o:.9f}"])
    print(f"[save] trace written to {path}")


def _save_trace_plot(
    path: Path,
    *,
    joint: str,
    t_arr: np.ndarray,
    cmd_arr: np.ndarray,
    obs_arr: np.ndarray,
    metrics: dict[str, MetricValue],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"Could not import matplotlib to save plot: {exc}") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(t_arr, cmd_arr, label="cmd_deg", linewidth=2.0)
    ax.plot(t_arr, obs_arr, label="obs_deg", linestyle="--", linewidth=2.0)
    ax.set_title(joint)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    summary = (
        f"xcorr={float(metrics['delay_s_xcorr']):+.4f}s\n"
        f"phase={float(metrics['delay_s_phase_wrapped']):+.4f}s\n"
        f"peak_corr={float(metrics['peak_corr']):.3f}"
    )
    ax.text(
        0.99,
        0.02,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[save] figure written to {path}")


def _output_path_for_joint(
    base_path: str,
    joint: str,
    multi_joint: bool,
    *,
    default_suffix: str,
) -> Path:
    out = Path(base_path).expanduser()
    if not multi_joint:
        if out.suffix or not default_suffix:
            return out
        return out.with_suffix(default_suffix)
    if out.suffix:
        return out.with_name(f"{out.stem}_{joint}{out.suffix}")
    return out / f"{joint}{default_suffix}"


def measure_actuator_delay(
    robot: BipedalRobotController,
    *,
    joint: str = "left_knee",
    fps: int = 100,
    duration_s: float = 2.0,
    freq_hz: float = 1.0,
    amp_deg: float = 10.0,
    pre_roll_s: float = 1.0,
    between_s: float = 0.5,
) -> tuple[dict[str, MetricValue], np.ndarray, np.ndarray, np.ndarray]:
    if joint not in JOINT_INDEX:
        raise ValueError(f"Unsupported joint '{joint}', expected one of {JOINT_ORDER}")
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if duration_s <= 0:
        raise ValueError("duration_s must be > 0")
    if freq_hz <= 0:
        raise ValueError("freq_hz must be > 0")
    if robot.mode != "control":
        raise RuntimeError("Robot must already be running in control mode before measuring delay.")

    _ensure_valid_state(robot)
    estop_reason = robot.get_estop_reason()
    if estop_reason:
        raise RuntimeError(f"E-STOP active before replay. Reason: {estop_reason}")

    hold_q_deg = _capture_joint_state_deg(robot)
    joint_idx = JOINT_INDEX[joint]
    center_deg = float(hold_q_deg[joint_idx])

    _send_joint_vector(robot, hold_q_deg)
    if pre_roll_s > 0.0:
        time.sleep(float(pre_roll_s))

    estop_reason = robot.get_estop_reason()
    if estop_reason:
        raise RuntimeError(f"E-STOP triggered during pre-roll. Reason: {estop_reason}")

    dt = 1.0 / float(fps)

    t_log: list[float] = []
    cmd_log: list[float] = []
    obs_log: list[float] = []
    t0 = time.perf_counter()
    next_tick = t0

    while True:
        now = time.perf_counter()
        sleep_s = next_tick - now
        if sleep_s > 0.0:
            time.sleep(sleep_s)
        else:
            next_tick = now

        sample_t = float(time.perf_counter() - t0)
        if sample_t > float(duration_s):
            break

        obs_q_deg = _capture_joint_state_deg(robot)
        cmd_deg = center_deg + float(amp_deg) * math.sin(2.0 * math.pi * float(freq_hz) * sample_t)
        q_cmd_deg = hold_q_deg.copy()
        q_cmd_deg[joint_idx] = float(cmd_deg)
        _send_joint_vector(robot, q_cmd_deg)

        t_log.append(sample_t)
        cmd_log.append(float(cmd_deg))
        obs_log.append(float(obs_q_deg[joint_idx]))
        next_tick += dt

        estop_reason = robot.get_estop_reason()
        if estop_reason:
            raise RuntimeError(f"E-STOP triggered during replay. Reason: {estop_reason}")

    _send_joint_vector(robot, hold_q_deg)
    if between_s > 0.0:
        time.sleep(float(between_s))

    return _analyze_trace(
        t_log=t_log,
        cmd_log=cmd_log,
        obs_log=obs_log,
        fps=int(fps),
        freq_hz=float(freq_hz),
        center_deg=center_deg,
    )


def measure_actuator_delays(
    robot: BipedalRobotController,
    *,
    joint: str = "left_knee",
    all_joints: bool = False,
    fps: int = 100,
    duration_s: float = 2.0,
    freq_hz: float = 1.0,
    amp_deg: float = 10.0,
    pre_roll_s: float = 1.0,
    between_s: float = 0.5,
    save_csv: str = "",
    save_png: str = "",
    print_summary: bool = True,
) -> dict[str, dict[str, Any]]:
    joints_to_run = list(JOINT_ORDER) if bool(all_joints) else [str(joint)]
    multi_joint = len(joints_to_run) > 1
    results: dict[str, dict[str, Any]] = {}

    for idx, joint_name in enumerate(joints_to_run):
        if print_summary:
            print(
                f"[record] joint={joint_name} profile=sinus duration={float(duration_s):.2f}s "
                f"fps={int(fps)} ({idx + 1}/{len(joints_to_run)})"
            )
        metrics, t_arr, cmd_arr, obs_arr = measure_actuator_delay(
            robot,
            joint=joint_name,
            fps=int(fps),
            duration_s=float(duration_s),
            freq_hz=float(freq_hz),
            amp_deg=float(amp_deg),
            pre_roll_s=float(pre_roll_s),
            between_s=float(between_s),
        )
        results[joint_name] = {
            "metrics": metrics,
            "t_arr": t_arr,
            "cmd_arr": cmd_arr,
            "obs_arr": obs_arr,
        }

        if print_summary:
            _print_delay_summary(joint_name, metrics)

        if str(save_csv).strip():
            out = _output_path_for_joint(
                str(save_csv),
                joint_name,
                multi_joint=multi_joint,
                default_suffix=".csv",
            )
            _save_trace_csv(out, t_arr, cmd_arr, obs_arr)
        if str(save_png).strip():
            out = _output_path_for_joint(
                str(save_png),
                joint_name,
                multi_joint=multi_joint,
                default_suffix=".png",
            )
            _save_trace_plot(
                out,
                joint=joint_name,
                t_arr=t_arr,
                cmd_arr=cmd_arr,
                obs_arr=obs_arr,
                metrics=metrics,
            )

    if print_summary and multi_joint:
        print("\n[summary] wrapped phase delay by joint")
        for joint_name in joints_to_run:
            metrics = results[joint_name]["metrics"]
            print(
                f"  {joint_name:>18s}: {float(metrics['delay_s_phase_wrapped']):+.6f}s "
                f"(xcorr={float(metrics['delay_s_xcorr']):+.6f}s, peak_corr={float(metrics['peak_corr']):.4f})"
            )

    return results


def _build_robot_from_args(args: argparse.Namespace) -> BipedalRobotController:
    if bool(args.use_mock_bus):
        robot = BipedalRobotController(
            bus_can0=MockBus(),
            bus_can1=MockBus(),
            control_hz=float(args.control_hz),
            recv_timeout_s=float(args.recv_timeout_s),
            log_path=str(args.log_path),
        )
        for mid in MOTOR_IDS:
            robot.set_joint_limit(mid, -720.0, 720.0)
    else:
        robot = BipedalRobotController(
            control_hz=float(args.control_hz),
            recv_timeout_s=float(args.recv_timeout_s),
            log_path=str(args.log_path),
            channel_can0=str(args.can0_port),
            channel_can1=str(args.can1_port),
        )

    robot.set_max_command_delta(float(args.max_command_delta_deg))
    robot.enforce_command_limits = not bool(args.disable_command_limits)
    return robot


def main() -> None:
    args = _parse_args()

    robot = _build_robot_from_args(args)
    print(
        f"[setup] joints={'all' if bool(args.all_joints) else args.joint} "
        f"fps={args.fps} duration={args.duration_s:.3f}s "
        f"freq={args.freq_hz:.3f}Hz amp={args.amp_deg:.3f}deg "
        f"mock_bus={bool(args.use_mock_bus)}"
    )
    print("[setup] starting BipedalRobotController")
    robot.start(mode="control", auto_enable=True)

    try:
        _ensure_valid_state(robot)
        measure_actuator_delays(
            robot,
            joint=str(args.joint),
            all_joints=bool(args.all_joints),
            fps=int(args.fps),
            duration_s=float(args.duration_s),
            freq_hz=float(args.freq_hz),
            amp_deg=float(args.amp_deg),
            pre_roll_s=float(args.pre_roll_s),
            between_s=float(args.between_s),
            save_csv=str(args.save_csv),
            save_png=str(args.save_png),
            print_summary=True,
        )
    finally:
        try:
            _ensure_valid_state(robot, retries=1, sleep_s=0.0)
            _send_joint_vector(robot, _capture_joint_state_deg(robot))
        except Exception:
            pass
        try:
            robot.stop(disable_motors=True)
        except Exception as exc:
            print(f"[warn] robot stop raised: {exc}")
        print("[teardown] robot stopped")


if __name__ == "__main__":
    main()
