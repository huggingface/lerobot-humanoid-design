#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import inspect
import queue
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import gradio as gr
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from typing import get_type_hints

from bipedal_robot import BipedalRobotController
from mock_bus import MockBus
from ocp_controller_staged import go_to_pose, load_ocp_npy
from root_constant import JOINT_LIMITS_DEG, MOTOR_IDS


def _gradio_major_version() -> int:
    try:
        return int(str(getattr(gr, "__version__", "0")).split(".")[0])
    except Exception:
        return 0


def _patch_gradio_py39_special_args() -> None:
    """
    Gradio 5.x currently uses `Type | None` in special_args, which breaks on Python 3.9.
    Patch only that helper with a py3.9-safe implementation (no OAuth typed injection needed here).
    """
    try:
        from gradio import helpers as gh
        from gradio import blocks as gr_blocks
        from gradio import routes as gr_routes
        from gradio.events import EventData
        from gradio.helpers import Progress
        from gradio import processing_utils
    except Exception:
        return

    orig = gh.special_args

    def _py39_special_args(fn, inputs=None, request=None, event_data=None):
        try:
            return orig(fn, inputs=inputs, request=request, event_data=event_data)
        except TypeError as exc:
            if "unsupported operand type(s) for |" not in str(exc):
                raise

        # Fallback equivalent behavior without OAuth annotation branch.
        try:
            signature = inspect.signature(fn)
        except ValueError:
            return inputs or [], None, None

        type_hints = get_type_hints(fn)
        positional_args = []
        for param in signature.parameters.values():
            if param.kind not in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                break
            positional_args.append(param)

        progress_index = None
        event_data_index = None
        for i, param in enumerate(positional_args):
            type_hint = type_hints.get(param.name)
            if isinstance(param.default, Progress):
                progress_index = i
                if inputs is not None:
                    inputs.insert(i, param.default)
            elif type_hint in (gr_routes.Request, Optional[gr_routes.Request]):
                if inputs is not None:
                    inputs.insert(i, request)
            elif type_hint and inspect.isclass(type_hint) and issubclass(type_hint, EventData):
                event_data_index = i
                if inputs is not None and event_data is not None:
                    processing_utils.check_all_files_in_cache(event_data._data)
                    inputs.insert(i, type_hint(event_data.target, event_data._data))
            elif param.default is not param.empty and inputs is not None and len(inputs) <= i:
                inputs.insert(i, param.default)

        if inputs is not None:
            while len(inputs) < len(positional_args):
                i = len(inputs)
                param = positional_args[i]
                if param.default == param.empty:
                    warnings.warn("Unexpected argument. Filling with None.")
                    inputs.append(None)
                else:
                    inputs.append(param.default)
        return inputs or [], progress_index, event_data_index

    gh.special_args = _py39_special_args
    try:
        gr_blocks.special_args = _py39_special_args
    except Exception:
        pass


def _patch_zip_strict_py39() -> None:
    """
    Python 3.9 compatibility for Gradio internals using zip(..., strict=...).
    Patch only gradio modules, not global builtins.
    """
    try:
        zip([1], [1], strict=True)  # type: ignore[arg-type]
        return
    except TypeError:
        pass

    try:
        import gradio.blocks as gr_blocks
    except Exception:
        return

    orig_zip = zip
    _sentinel = object()

    def zip_compat(*iterables, **kwargs):
        strict = bool(kwargs.pop("strict", False))
        if kwargs:
            raise TypeError("zip() takes no keyword arguments")
        if not strict:
            return orig_zip(*iterables)

        from itertools import zip_longest

        for tup in zip_longest(*iterables, fillvalue=_sentinel):
            if _sentinel in tup:
                raise ValueError("zip() argument lengths differ")
            yield tup

    gr_blocks.zip = zip_compat


# Gradio v4/v5 needs py3.9 shims. Gradio v3 should keep default behavior.
if _gradio_major_version() >= 4:
    _patch_gradio_py39_special_args()
    _patch_zip_strict_py39()


@dataclass
class AppState:
    lock: threading.Lock
    robot: Optional[BipedalRobotController] = None
    traj = None
    traj_path: Optional[Path] = None
    squat_idx: Optional[int] = None
    status: str = "idle"
    # (timestamp_s, {motor_id: {"pos": deg, "vel": deg/s, "tau": Nm}})
    state_history: deque = field(default_factory=lambda: deque(maxlen=4000))
    last_plot_img: Optional[np.ndarray] = None
    last_render_ts: float = 0.0
    last_sample_ts: float = 0.0
    plot_lock: threading.Lock = field(default_factory=threading.Lock)
    cmd_queue: queue.Queue = field(default_factory=queue.Queue)
    workers_running: bool = False
    cmd_thread: Optional[threading.Thread] = None
    plot_thread: Optional[threading.Thread] = None


APP = AppState(lock=threading.Lock())
PLOT_WORKER_PERIOD_S = 0.2
UI_REFRESH_PERIOD_S_V4 = 0.2
UI_REFRESH_PERIOD_S_V3 = 0.4


def _safe_status(msg: str) -> str:
    ts = time.strftime("%H:%M:%S")
    return f"[{ts}] {msg}"


def _status_only(status: str):
    return status


def _enqueue_action(action_fn, *args):
    APP.cmd_queue.put((action_fn, args))
    return _safe_status(f"queued: {action_fn.__name__}")


def _load_traj(path_txt: str, dt_s: float) -> str:
    p = Path(path_txt).expanduser()
    if not p.exists():
        return _safe_status(f"trajectory not found: {p}")
    traj = load_ocp_npy(p, dt_s=float(dt_s))
    dev = np.linalg.norm(np.asarray(traj.q_deg, dtype=float) - np.asarray(traj.q_deg[0], dtype=float)[None, :], axis=1)
    squat_idx = int(np.argmax(dev))
    with APP.lock:
        APP.traj = traj
        APP.traj_path = p
        APP.squat_idx = squat_idx
        APP.status = _safe_status(f"trajectory loaded: {p.name}, N={traj.t_s.size}, squat_idx={squat_idx}")
        return APP.status


def _build_robot(use_mock: bool, control_hz: float, log_path: str) -> BipedalRobotController:
    if use_mock:
        robot = BipedalRobotController(
            bus_can0=MockBus(),
            bus_can1=MockBus(),
            control_hz=float(control_hz),
            log_path=Path(log_path),
        )
        robot.set_startup_wrap_policy(enabled=False)
        for mid in MOTOR_IDS:
            robot.set_joint_limit(mid, -720.0, 720.0)
        robot.set_max_command_delta(1000.0)
    else:
        robot = BipedalRobotController(
            control_hz=float(control_hz),
            log_path=Path(log_path),
        )
    return robot


def start_controller(use_mock: bool, control_hz: float, log_path: str) -> str:
    with APP.lock:
        if APP.robot is not None:
            return _safe_status("controller already running")
    try:
        robot = _build_robot(bool(use_mock), float(control_hz), log_path)
        robot.start(mode="control", auto_enable=False)
    except Exception as exc:
        return _safe_status(f"start failed: {exc}")
    with APP.lock:
        APP.robot = robot
        APP.state_history.clear()
        APP.status = _safe_status(f"controller started (mock={bool(use_mock)}, hz={float(control_hz):.1f})")
        return APP.status


def stop_controller() -> str:
    with APP.lock:
        robot = APP.robot
        APP.robot = None
    if robot is not None:
        try:
            robot.stop(disable_motors=True)
        except Exception as exc:
            return _safe_status(f"stop error: {exc}")
    with APP.lock:
        APP.status = _safe_status("controller stopped")
        return APP.status


def enable_motors() -> str:
    with APP.lock:
        robot = APP.robot
    if robot is None:
        return _safe_status("controller not started")
    robot.enable_all()
    return _safe_status("motors enabled")


def disable_motors() -> str:
    with APP.lock:
        robot = APP.robot
    if robot is None:
        return _safe_status("controller not started")
    robot.disable_all()
    return _safe_status("motors disabled")


def go_to_ocp_zero(duration_s: float) -> str:
    with APP.lock:
        robot = APP.robot
        traj = APP.traj
    if robot is None:
        return _safe_status("controller not started")
    if traj is None:
        return _safe_status("trajectory not loaded")
    go_to_pose(robot, np.asarray(traj.q_deg[0], dtype=float), duration_s=float(duration_s))
    return _safe_status(f"moved to OCP zero pose in {float(duration_s):.2f}s")


def one_squat(duration_s: float) -> str:
    with APP.lock:
        robot = APP.robot
        traj = APP.traj
        squat_idx = APP.squat_idx
    if robot is None:
        return _safe_status("controller not started")
    if traj is None or squat_idx is None:
        return _safe_status("trajectory not loaded")
    q0 = np.asarray(traj.q_deg[0], dtype=float)
    qsq = np.asarray(traj.q_deg[int(squat_idx)], dtype=float)
    half = max(0.1, float(duration_s) * 0.5)
    go_to_pose(robot, qsq, duration_s=half)
    go_to_pose(robot, q0, duration_s=half)
    return _safe_status(f"one squat done (idx={squat_idx}, total={float(duration_s):.2f}s)")


def _empty_plot(title: str):
    fig, ax = plt.subplots(figsize=(3.6, 2.2), dpi=70)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def _fig_to_rgb(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = buf.reshape(h, w, 3).copy()
    plt.close(fig)
    return img


def _build_dashboard_plot(history_items):
    if not history_items:
        return _empty_plot("motor states")
    t0 = history_items[0][0]
    t = np.array([it[0] - t0 for it in history_items], dtype=float)
    fig, axes = plt.subplots(3, 2, figsize=(10.2, 5.8), dpi=60, sharex=True)
    left_ids = list(MOTOR_IDS[: len(MOTOR_IDS) // 2])
    right_ids = list(MOTOR_IDS[len(MOTOR_IDS) // 2 :])
    left_colors = plt.cm.Blues(np.linspace(0.45, 0.95, len(left_ids)))
    right_colors = plt.cm.Oranges(np.linspace(0.45, 0.95, len(right_ids)))

    for i, mid in enumerate(left_ids):
        pos = np.array([it[1].get(mid, {}).get("pos", np.nan) for it in history_items], dtype=float)
        vel = np.array([it[1].get(mid, {}).get("vel", np.nan) for it in history_items], dtype=float)
        tau = np.array([it[1].get(mid, {}).get("tau", np.nan) for it in history_items], dtype=float)
        axes[0, 0].plot(t, pos, color=left_colors[i], linewidth=0.9, label=f"m{mid}")
        axes[1, 0].plot(t, vel, color=left_colors[i], linewidth=0.9, label=f"m{mid}")
        axes[2, 0].plot(t, tau, color=left_colors[i], linewidth=0.9, label=f"m{mid}")

    for i, mid in enumerate(right_ids):
        pos = np.array([it[1].get(mid, {}).get("pos", np.nan) for it in history_items], dtype=float)
        vel = np.array([it[1].get(mid, {}).get("vel", np.nan) for it in history_items], dtype=float)
        tau = np.array([it[1].get(mid, {}).get("tau", np.nan) for it in history_items], dtype=float)
        axes[0, 1].plot(t, pos, color=right_colors[i], linewidth=0.9, label=f"m{mid}")
        axes[1, 1].plot(t, vel, color=right_colors[i], linewidth=0.9, label=f"m{mid}")
        axes[2, 1].plot(t, tau, color=right_colors[i], linewidth=0.9, label=f"m{mid}")

    axes[0, 0].set_title("Left leg: position")
    axes[0, 1].set_title("Right leg: position")
    axes[1, 0].set_title("Left leg: velocity")
    axes[1, 1].set_title("Right leg: velocity")
    axes[2, 0].set_title("Left leg: torque")
    axes[2, 1].set_title("Right leg: torque")

    axes[0, 0].set_ylabel("deg")
    axes[1, 0].set_ylabel("deg/s")
    axes[2, 0].set_ylabel("Nm")
    axes[2, 0].set_xlabel("time [s]")
    axes[2, 1].set_xlabel("time [s]")

    for ax in axes.reshape(-1):
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", fontsize=6, ncol=2, frameon=False)

    fig.suptitle("Motor states (last 2s)", y=0.995)
    fig.tight_layout()
    return fig


def get_state_image():
    with APP.lock:
        status = APP.status
        last_img = APP.last_plot_img
    if last_img is not None:
        return status, last_img
    with APP.plot_lock:
        img = _fig_to_rgb(_empty_plot("motor states"))
    with APP.lock:
        APP.last_plot_img = img
    return status, img


def _cleanup() -> None:
    with APP.lock:
        APP.workers_running = False
    with APP.lock:
        robot = APP.robot
        APP.robot = None
    if robot is not None:
        try:
            robot.stop(disable_motors=True)
        except Exception:
            pass


def _command_worker_loop() -> None:
    while True:
        with APP.lock:
            running = APP.workers_running
        if not running:
            return
        try:
            fn, args = APP.cmd_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            status = fn(*args)
        except Exception as exc:
            status = _safe_status(f"command error ({fn.__name__}): {exc}")
        with APP.lock:
            APP.status = status
        APP.cmd_queue.task_done()


def _plot_worker_loop() -> None:
    while True:
        with APP.lock:
            running = APP.workers_running
            robot = APP.robot
        if not running:
            return
        if robot is None:
            time.sleep(0.2)
            continue
        try:
            snap = robot.get_state_snapshot()
            now = time.time()
            sample_ts = max(float(snap[mid].stamp) for mid in MOTOR_IDS)
            state_map = {
                mid: {
                    "pos": float(snap[mid].position_deg),
                    "vel": float(snap[mid].velocity_deg_s),
                    "tau": float(snap[mid].torque_nm),
                }
                for mid in MOTOR_IDS
            }
            with APP.lock:
                APP.state_history.append((now, state_map))
                while APP.state_history and (now - APP.state_history[0][0]) > 2.0:
                    APP.state_history.popleft()
                history_items = list(APP.state_history)
            # Always refresh from current 2s buffer. Some buses can expose non-monotonic or flat stamps.
            with APP.plot_lock:
                img = _fig_to_rgb(_build_dashboard_plot(history_items))
            with APP.lock:
                APP.last_plot_img = img
                APP.last_render_ts = now
                APP.last_sample_ts = sample_ts
        except Exception as exc:
            with APP.lock:
                APP.status = _safe_status(f"plot warning: {exc}")
        time.sleep(PLOT_WORKER_PERIOD_S)


def _start_workers_once() -> None:
    with APP.lock:
        if APP.workers_running:
            return
        APP.workers_running = True
        APP.cmd_thread = threading.Thread(target=_command_worker_loop, daemon=True)
        APP.plot_thread = threading.Thread(target=_plot_worker_loop, daemon=True)
        APP.cmd_thread.start()
        APP.plot_thread.start()


atexit.register(_cleanup)


def build_ui(default_traj: str, default_log: str, default_hz: float, default_dt: float) -> gr.Blocks:
    _start_workers_once()
    with gr.Blocks(title="Raspi Robot Controller") as demo:
        gr.Markdown("## Raspi Robot Controller")
        with gr.Row():
            use_mock = gr.Checkbox(label="Use Mock Bus", value=True)
            control_hz = gr.Number(label="Control Hz", value=default_hz, precision=2)
            log_path = gr.Textbox(label="Log Path", value=default_log)

        with gr.Row():
            traj_path = gr.Textbox(label="Trajectory .npy", value=default_traj)
            traj_dt = gr.Number(label="Trajectory dt [s]", value=default_dt, precision=4)
            load_btn = gr.Button("Load Trajectory", variant="secondary")

        with gr.Row():
            start_btn = gr.Button("Start Controller", variant="primary")
            stop_btn = gr.Button("Stop Controller", variant="secondary")
            enable_btn = gr.Button("Enable Motors")
            disable_btn = gr.Button("Disable Motors")

        with gr.Row():
            go_zero_s = gr.Number(label="Go-To-Zero Duration [s]", value=2.0, precision=2)
            squat_s = gr.Number(label="Squat Duration [s]", value=2.0, precision=2)
            zero_btn = gr.Button("Go To OCP Zero")
            squat_btn = gr.Button("One Squat")

        status = gr.Textbox(label="Status", value="idle")
        state_plot = gr.Image(
            label="Motor states (last 2s)",
            value=_fig_to_rgb(_empty_plot("motor states")),
            interactive=False,
        )

        def _load_cb(path_txt, dt):
            _enqueue_action(_load_traj, path_txt, dt)
            return _safe_status("queued: load trajectory")

        def _start_cb(use_mock_v, hz_v, log_v):
            _enqueue_action(start_controller, use_mock_v, hz_v, log_v)
            return _safe_status("queued: start controller")

        def _stop_cb():
            _enqueue_action(stop_controller)
            return _safe_status("queued: stop controller")

        def _enable_cb():
            _enqueue_action(enable_motors)
            return _safe_status("queued: enable motors")

        def _disable_cb():
            _enqueue_action(disable_motors)
            return _safe_status("queued: disable motors")

        def _zero_cb(dur):
            _enqueue_action(go_to_ocp_zero, dur)
            return _safe_status("queued: go to zero")

        def _squat_cb(dur):
            _enqueue_action(one_squat, dur)
            return _safe_status("queued: one squat")

        load_btn.click(fn=_load_cb, inputs=[traj_path, traj_dt], outputs=[status], queue=False)
        start_btn.click(fn=_start_cb, inputs=[use_mock, control_hz, log_path], outputs=[status], queue=False)
        stop_btn.click(fn=_stop_cb, inputs=None, outputs=[status], queue=False)
        enable_btn.click(fn=_enable_cb, inputs=None, outputs=[status], queue=False)
        disable_btn.click(fn=_disable_cb, inputs=None, outputs=[status], queue=False)
        zero_btn.click(fn=_zero_cb, inputs=[go_zero_s], outputs=[status], queue=False)
        squat_btn.click(fn=_squat_cb, inputs=[squat_s], outputs=[status], queue=False)

        # Gradio v4/v5: Timer component. Gradio v3: periodic demo.load fallback.
        if hasattr(gr, "Timer"):
            refresh_timer = gr.Timer(value=UI_REFRESH_PERIOD_S_V4)
            refresh_timer.tick(fn=get_state_image, inputs=None, outputs=[status, state_plot], queue=False)
        else:
            # Gradio v3 periodic polling path: keep queue-enabled behavior for reliability.
            demo.load(fn=get_state_image, inputs=None, outputs=[status, state_plot], every=UI_REFRESH_PERIOD_S_V3)

    return demo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gradio UI for Raspberry Pi robot control.")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Create Gradio share link if localhost is not reachable.")
    p.add_argument("--traj", type=str, default="/tmp/real_robot_jump_mirrored.npy")
    p.add_argument("--log", type=str, default="bipedal_state_log.csv")
    p.add_argument("--control-hz", type=float, default=100.0)
    p.add_argument("--dt", type=float, default=0.005)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_ui(
        default_traj=args.traj,
        default_log=args.log,
        default_hz=float(args.control_hz),
        default_dt=float(args.dt),
    )
    demo.queue(concurrency_count=8).launch(
        server_name=args.host,
        server_port=int(args.port),
        show_error=True,
        share=bool(args.share),
    )


if __name__ == "__main__":
    main()
