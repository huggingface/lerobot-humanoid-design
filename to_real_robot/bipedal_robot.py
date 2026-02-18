from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import csv
import threading
import time

import can
import numpy as np

from leg_test.mit import MotorState, decode_state_frame, pack_mit_command
from root_constant import (
    MODEL_URDF_DIR,
    MODEL_URDF_PATH,
    MOTOR_IDS,
    MOTORS,
    CAN0_MOTOR_IDS,
    CAN1_MOTOR_IDS,
    MOTOR_SIGN,
    MOTOR_OFFSET_DEG,
    ANKLE_COUPLING_CALIBRATION_LEFT,
    ANKLE_COUPLING_CALIBRATION_RIGHT,
    DEFAULT_GAINS,
    JOINT_LIMITS_DEG,
    COMMAND_MARGIN_DEG,
    STATE_MARGIN_DEG,
    NEAR_STOP_MARGIN_DEG,
    DAMPING_KD,
    CAN_CMD_CLEAR_FAULT,
    CAN_CMD_ZERO,
    CAN_CMD_ENABLE,
    CAN_CMD_DISABLE,
)

try:
    import pinocchio as pin
    import meshcat
    from pinocchio.robot_wrapper import RobotWrapper
    from pinocchio.visualize import MeshcatVisualizer
except Exception:  # pragma: no cover - optional runtime dependencies
    pin = None
    meshcat = None
    RobotWrapper = None
    MeshcatVisualizer = None


@dataclass
class JointGains:
    kp: float
    kd: float


@dataclass
class MotorCommand:
    position_deg: float = 0.0
    velocity_deg_s: float = 0.0
    torque_nm: float = 0.0
    kp: Optional[float] = None
    kd: Optional[float] = None


class BipedalRobotController:
    MODES = ("state_only", "control")

    def __init__(
        self,
        *,
        interface: str = "socketcan",
        channel_can0: str = "can0",
        channel_can1: str = "can1",
        bus_can0: Optional[Any] = None,
        bus_can1: Optional[Any] = None,
        control_hz: float = 100.0,
        recv_timeout_s: float = 0.001,
        log_path: Union[str, Path] = "bipedal_state_log.csv",
        imu: Optional[Any] = None,
    ):
        self.bus_can0 = bus_can0 if bus_can0 is not None else can.interface.Bus(interface=interface, channel=channel_can0)
        self.bus_can1 = bus_can1 if bus_can1 is not None else can.interface.Bus(interface=interface, channel=channel_can1)
        self.control_hz = float(control_hz)
        self.recv_timeout_s = float(recv_timeout_s)
        self.mode = "state_only"

        self.state: Dict[int, MotorState] = {mid: MotorState() for mid in MOTOR_IDS}
        self.gains: Dict[int, JointGains] = {
            mid: JointGains(*DEFAULT_GAINS.get(mid, (5.0, 0.5))) for mid in MOTOR_IDS
        }
        self.motor_sign: Dict[int, float] = {mid: float(MOTOR_SIGN[mid]) for mid in MOTOR_IDS}
        self.motor_offset_deg: Dict[int, float] = {mid: float(MOTOR_OFFSET_DEG[mid]) for mid in MOTOR_IDS}
        self.joint_limits_deg: Dict[int, tuple[float, float]] = {
            mid: tuple(JOINT_LIMITS_DEG[mid]) for mid in MOTOR_IDS if mid in JOINT_LIMITS_DEG
        }
        self.command_limits_cal_deg: Dict[int, tuple[float, float]] = {}
        self.action: Dict[int, MotorCommand] = {mid: MotorCommand() for mid in MOTOR_IDS}
        self.protocol_usage = {
            "clear_fault": 0,
            "zero": 0,
            "enable": 0,
            "disable": 0,
            "mit_control": 0,
            "damping": 0,
        }

        self._state_lock = threading.Lock()
        self._action_lock = threading.Lock()
        self._loop_thread: Optional[threading.Thread] = None
        self._running = False
        self._estop = False
        self._estop_reason = ""
        self._damping_active = False
        self._action_initialized = False
        self._state_only_enforced = False
        self._cycle_idx = 0

        self._viz = None
        self._model_nq = 12
        self._viz_thread: Optional[threading.Thread] = None
        self._viz_running = False
        self._viz_hz = 60.0
        self._viz_data_lock = threading.Lock()
        self._viz_raw_latest: Optional[Dict[int, float]] = None
        self._limit_source = "raw"
        self._startup_wrap_checked = False
        self._auto_shift_limits_with_wrap = False
        self.max_cmd_delta_deg = 20.0
        self._last_jump_warn_s: Dict[int, float] = {mid: 0.0 for mid in MOTOR_IDS}
        self.reset_estop_each_state_cycle = True
        self._recompute_command_limits()

        self.log_path = Path(log_path)
        self._log_fp = None
        self._log_writer = None

        self._imu_lock = threading.Lock()
        self._imu = imu
        self._imu_state: Optional[Dict[str, Any]] = None
        self._orientation_quaternion_xyzw: Optional[tuple[float, float, float, float]] = None
        self._orientation_stamp_s: float = 0.0

    # -------------------------
    # Public API
    # -------------------------
    def set_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unsupported mode '{mode}', expected one of {self.MODES}")
        if mode == "control" and self.mode != "control":
            # Re-latch the next measured state as initial hold target.
            self._action_initialized = False
            self._state_only_enforced = False
        if mode == "state_only" and self.mode != "state_only":
            self._state_only_enforced = False
        self.mode = mode

    def set_joint_gains(self, motor_id: int, *, kp: Optional[float] = None, kd: Optional[float] = None) -> None:
        g = self.gains[motor_id]
        self.gains[motor_id] = JointGains(kp if kp is not None else g.kp, kd if kd is not None else g.kd)

    def set_limit_source(self, source: str) -> None:
        if source not in ("raw", "calibrated"):
            raise ValueError("source must be 'raw' or 'calibrated'")
        self._limit_source = source
        self._recompute_command_limits()

    def set_motor_calibration(self, motor_id: int, *, sign: Optional[float] = None, offset_deg: Optional[float] = None) -> None:
        if sign is not None:
            self.motor_sign[motor_id] = float(sign)
        if offset_deg is not None:
            self.motor_offset_deg[motor_id] = float(offset_deg)
        self._recompute_command_limits()

    def set_joint_limit(self, motor_id: int, lo_deg: float, hi_deg: float) -> None:
        self.joint_limits_deg[motor_id] = (float(lo_deg), float(hi_deg))
        self._recompute_command_limits()

    def set_startup_wrap_policy(self, *, enabled: bool = True, shift_limits: bool = True) -> None:
        self._startup_wrap_checked = not bool(enabled)
        self._auto_shift_limits_with_wrap = bool(shift_limits)

    def set_max_command_delta(self, delta_deg: float) -> None:
        self.max_cmd_delta_deg = float(delta_deg)

    def set_state_only_estop_reset(self, enabled: bool = True) -> None:
        self.reset_estop_each_state_cycle = bool(enabled)

    def set_action(
        self,
        *,
        left: Optional[Dict[str, float]] = None,
        right: Optional[Dict[str, float]] = None,
        velocity_deg_s: Union[float, int] = 0.0,
        torque_nm: Union[float, int] = 0.0,
        left_velocity_deg_s: Optional[Dict[str, float]] = None,
        right_velocity_deg_s: Optional[Dict[str, float]] = None,
        left_torque_nm: Optional[Dict[str, float]] = None,
        right_torque_nm: Optional[Dict[str, float]] = None,
        kp: Optional[float] = None,
        kd: Optional[float] = None,
    ) -> Dict[int, float]:
        # Joint-space public API.
        return self.set_joint_action(
            left=left,
            right=right,
            velocity_deg_s=velocity_deg_s,
            torque_nm=torque_nm,
            left_velocity_deg_s=left_velocity_deg_s,
            right_velocity_deg_s=right_velocity_deg_s,
            left_torque_nm=left_torque_nm,
            right_torque_nm=right_torque_nm,
            kp=kp,
            kd=kd,
        )

    def set_joint_action(
        self,
        *,
        left: Optional[Dict[str, float]] = None,
        right: Optional[Dict[str, float]] = None,
        velocity_deg_s: Union[float, int] = 0.0,
        torque_nm: Union[float, int] = 0.0,
        left_velocity_deg_s: Optional[Dict[str, float]] = None,
        right_velocity_deg_s: Optional[Dict[str, float]] = None,
        left_torque_nm: Optional[Dict[str, float]] = None,
        right_torque_nm: Optional[Dict[str, float]] = None,
        kp: Optional[float] = None,
        kd: Optional[float] = None,
    ) -> Dict[int, float]:
        """
        Update command targets in joint space.
        Joint keys per side: hipz, hipx, hipy, knee, ankle_pitch, ankle_roll.
        Velocity and torque inputs are interpreted in joint space as well and
        converted to motor-space using per-motor signs and ankle coupling.
        Returns effective raw motor-space send targets (deg), i.e. after clamp/wrap.
        """
        left = left or {}
        right = right or {}
        left_velocity_deg_s = left_velocity_deg_s or {}
        right_velocity_deg_s = right_velocity_deg_s or {}
        left_torque_nm = left_torque_nm or {}
        right_torque_nm = right_torque_nm or {}

        with self._action_lock:
            proposed = {mid: cmd for mid, cmd in self.action.items()}

            # Build current command in joint space from existing motor commands.
            cur_cmd_raw = {mid: float(proposed[mid].position_deg) for mid in MOTOR_IDS}
            q_cmd_deg = self.motor_state_to_joint_state(cur_cmd_raw, output_radians=False, nq=12)
            qd_cmd_deg_s = np.full(12, float(velocity_deg_s), dtype=float)
            tau_cmd_nm = np.full(12, float(torque_nm), dtype=float)

            # Apply partial updates in model joint space.
            self._apply_joint_side_updates(q_cmd_deg, left=left, right=right)
            self._apply_joint_side_updates(qd_cmd_deg_s, left=left_velocity_deg_s, right=right_velocity_deg_s)
            self._apply_joint_side_updates(tau_cmd_nm, left=left_torque_nm, right=right_torque_nm)

            # Convert desired joint state back to per-motor raw commands.
            target_raw = self.joint_state_to_motor_state(q_cmd_deg, input_radians=False, output_space="raw")
            vel_raw, tau_raw = self._joint_vel_tau_to_motor_raw(qd_cmd_deg_s, tau_cmd_nm)
            if not self._passes_action_update_jump_guard(target_raw):
                req_raw = {mid: float(proposed[mid].position_deg) for mid in MOTOR_IDS}
            else:
                for mid in MOTOR_IDS:
                    proposed[mid] = self._with_fields(
                        proposed[mid],
                        target_raw[mid],
                        vel_raw[mid],
                        tau_raw[mid],
                        kp,
                        kd,
                    )
                self.action = proposed
                req_raw = {mid: float(proposed[mid].position_deg) for mid in MOTOR_IDS}
        return self._effective_send_raw_map(req_raw)

    def get_state_snapshot(self) -> Dict[int, MotorState]:
        with self._state_lock:
            return {mid: st for mid, st in self.state.items()}

    def attach_imu(self, imu: Any) -> None:
        with self._imu_lock:
            self._imu = imu
            self._imu_state = None

    def attach_bno085_i2c_imu(self, *, address: int = 0x4A) -> None:
        from IMU_integration import BNO085IMU

        self.attach_imu(BNO085IMU(address=address))

    def get_imu_snapshot(self) -> Optional[Dict[str, Any]]:
        with self._imu_lock:
            if self._imu_state is None:
                return None
            return dict(self._imu_state)

    def get_orientation_quaternion(self) -> Optional[tuple[float, float, float, float]]:
        with self._imu_lock:
            if self._orientation_quaternion_xyzw is None:
                return None
            return tuple(self._orientation_quaternion_xyzw)

    def get_orientation_timestamp(self) -> float:
        with self._imu_lock:
            return float(self._orientation_stamp_s)

    def get_combined_state_snapshot(self, *, include_joint_state: bool = True) -> Dict[str, Any]:
        with self._state_lock:
            motor_snapshot = {mid: asdict(st) for mid, st in self.state.items()}
        out: Dict[str, Any] = {
            "time_s": time.time(),
            "mode": self.mode,
            "estop": bool(self._estop),
            "estop_reason": str(self._estop_reason),
            "motors": motor_snapshot,
            "imu": self.get_imu_snapshot(),
            "orientation_quaternion_xyzw": self.get_orientation_quaternion(),
            "orientation_timestamp_s": self.get_orientation_timestamp(),
        }
        if include_joint_state:
            raw = {mid: float(motor_snapshot[mid]["position_deg"]) for mid in MOTOR_IDS}
            out["joint_state_rad"] = self.motor_state_to_joint_state(raw, output_radians=True, nq=12).tolist()
            out["joint_state_deg"] = self.motor_state_to_joint_state(raw, output_radians=False, nq=12).tolist()
        return out

    def get_protocol_usage(self) -> Dict[str, int]:
        return dict(self.protocol_usage)

    def get_unused_protocols(self) -> list[str]:
        return [name for name, count in self.protocol_usage.items() if count <= 0]

    def print_raw_motor_positions(self) -> None:
        """
        Print current raw motor positions (deg) for m1..m12.
        Useful in state_only mode to inspect encoder values directly.
        """
        with self._state_lock:
            snapshot = {mid: st for mid, st in self.state.items()}
        for mid in MOTOR_IDS:
            st = snapshot[mid]
            stamp_txt = "n/a" if st.stamp <= 0.0 else f"{st.stamp:.3f}"
            print(f"m{mid:02d} | raw={float(st.position_deg):+8.3f} deg | stamp={stamp_txt}")

    def preview_action_send(
        self,
        *,
        left: Optional[Dict[str, float]] = None,
        right: Optional[Dict[str, float]] = None,
        velocity_deg_s: Union[float, int] = 0.0,
        torque_nm: Union[float, int] = 0.0,
        left_velocity_deg_s: Optional[Dict[str, float]] = None,
        right_velocity_deg_s: Optional[Dict[str, float]] = None,
        left_torque_nm: Optional[Dict[str, float]] = None,
        right_torque_nm: Optional[Dict[str, float]] = None,
        kp: Optional[float] = None,
        kd: Optional[float] = None,
    ) -> Dict[int, Dict[str, float]]:
        """
        Dry-run conversion for a prospective joint-space action.
        Returns per-motor values:
          - req_raw_deg: requested raw target
          - tgt_raw_deg: raw target after command clamp
          - send_raw_deg: raw value encoded in MIT command
        """
        with self._action_lock:
            prev_action = {mid: cmd for mid, cmd in self.action.items()}
        try:
            self.set_joint_action(
                left=left,
                right=right,
                velocity_deg_s=velocity_deg_s,
                torque_nm=torque_nm,
                left_velocity_deg_s=left_velocity_deg_s,
                right_velocity_deg_s=right_velocity_deg_s,
                left_torque_nm=left_torque_nm,
                right_torque_nm=right_torque_nm,
                kp=kp,
                kd=kd,
            )
            with self._action_lock:
                cur_action = {mid: cmd for mid, cmd in self.action.items()}
        finally:
            with self._action_lock:
                self.action = prev_action

        out: Dict[int, Dict[str, float]] = {}
        with self._state_lock:
            state_snapshot = {mid: st for mid, st in self.state.items()}
        for mid in MOTOR_IDS:
            req_raw = float(cur_action[mid].position_deg)
            st = state_snapshot[mid]
            ref_raw = float(st.position_deg) if st.stamp > 0.0 else None
            tgt_raw = req_raw
            if st.stamp > 0.0:
                cur_raw = float(st.position_deg)
                send_raw = self._resolve_raw_target_near_current(mid, tgt_raw, cur_raw)
            else:
                send_raw = tgt_raw
            out[mid] = {
                "req_raw_deg": req_raw,
                "tgt_raw_deg": float(tgt_raw),
                "send_raw_deg": float(send_raw),
            }
        return out

    def start(self, *, mode: str = "state_only", auto_enable: bool = False) -> None:
        self.set_mode(mode)
        self._estop = False
        self._estop_reason = ""
        self._startup_wrap_checked = False
        self._state_only_enforced = False
        self._open_log()
        if auto_enable and mode == "control":
            self.enable_all()
        if self._loop_thread is None:
            self._running = True
            self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
            self._loop_thread.start()
        self._start_viz_thread()

    def stop(self, *, disable_motors: bool = True) -> None:
        self._running = False
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=2.0)
        self._loop_thread = None
        self._stop_viz_thread()
        if disable_motors:
            self.disable_all()
        self._close_log()
        self._shutdown_buses()
        with self._imu_lock:
            imu = self._imu
        if imu is not None and hasattr(imu, "stop"):
            try:
                imu.stop()
            except Exception:
                pass

    def clear_estop(self) -> None:
        self._estop = False
        self._estop_reason = ""

    def get_estop_reason(self) -> str:
        return str(self._estop_reason)

    def request_state_once(self, motor_ids: Iterable[int] = MOTOR_IDS) -> list[int]:
        return self._request_state(motor_ids)

    # -------------------------
    # CAN protocol wrappers
    # -------------------------
    def enable(self, motor_id: int) -> bool:
        if self._estop:
            return False
        self._send_cmd_ff(motor_id, CAN_CMD_ENABLE)
        self.protocol_usage["enable"] += 1
        rx = self._recv_any(self.recv_timeout_s)
        if rx is not None:
            self._try_update_state_from_msg(rx)
        return rx is not None

    def disable(self, motor_id: int) -> bool:
        self._send_cmd_ff(motor_id, CAN_CMD_DISABLE)
        self.protocol_usage["disable"] += 1
        rx = self._recv_any(self.recv_timeout_s)
        if rx is not None:
            self._try_update_state_from_msg(rx)
        return rx is not None

    def set_zero(self, motor_id: int, *, wait_s: float = 0.5):
        if self._estop:
            return None
        data = [0xFF] * 7 + [CAN_CMD_ZERO]
        self._send8(motor_id, data)
        self.protocol_usage["zero"] += 1
        rx = self._recv_any(wait_s)
        if rx is not None:
            self._try_update_state_from_msg(rx)
        return rx

    def enable_all(self) -> None:
        for mid in MOTOR_IDS:
            self.enable(mid)

    def disable_all(self) -> None:
        for mid in MOTOR_IDS:
            self.disable(mid)

    def set_zero_all(self) -> None:
        for mid in MOTOR_IDS:
            self.set_zero(mid)

    # -------------------------
    # Optional meshcat
    # -------------------------
    def attach_meshcat(self, viz) -> None:
        self._viz = viz
        try:
            self._model_nq = int(viz.model.nq)
        except Exception:
            self._model_nq = 12
        self._start_viz_thread()

    def attach_default_meshcat(self, *, zmq_url: str = "tcp://127.0.0.1:6000") -> None:
        if RobotWrapper is None or MeshcatVisualizer is None or meshcat is None or pin is None:
            raise RuntimeError("Meshcat/Pinocchio dependencies are not available in this environment.")

        urdf_candidates = [MODEL_URDF_DIR / "robit.urdf", MODEL_URDF_PATH]
        urdf_path = None
        for p in urdf_candidates:
            if p.exists():
                urdf_path = p
                break
        if urdf_path is None:
            raise FileNotFoundError(
                f"Could not find URDF. Expected one of: {urdf_candidates[0]} or {urdf_candidates[1]}"
            )

        robot_wrapper = RobotWrapper.BuildFromURDF(str(urdf_path), [str(MODEL_URDF_DIR)])
        viz = MeshcatVisualizer(
            robot_wrapper.model,
            robot_wrapper.collision_model,
            robot_wrapper.visual_model,
        )
        viz.viewer = meshcat.Visualizer(zmq_url=zmq_url)
        viz.clean()
        viz.loadViewerModel(rootNodeName="universe")
        viz.display(pin.neutral(robot_wrapper.model))
        self.attach_meshcat(viz)

    # -------------------------
    # Internal loop
    # -------------------------
    def _run_loop(self) -> None:
        period = 1.0 / max(1.0, self.control_hz)
        next_tick = time.perf_counter()

        while self._running:
            self._cycle_idx += 1
            missing: list[int] = []

            if self.mode == "state_only":
                if self.reset_estop_each_state_cycle:
                    self.clear_estop()
                # State-only mode explicitly polls state by CLEAR_FAULT.
                missing = self._request_state(MOTOR_IDS)
                if not self._startup_wrap_checked:
                    self._startup_wrap_checked = self._auto_correct_startup_wrap()
                if not self._state_only_enforced:
                    # Hard safety: in state-only mode, keep actuators disabled.
                    self.disable_all()
                    self._state_only_enforced = True
                self._damping_active = False
                self._action_initialized = False
            elif not self._estop:
                self._state_only_enforced = False
                # Control mode: use command replies as primary state source.
                # Poll only when bootstrapping or when some motors are stale.
                if not self._has_any_valid_state():
                    missing = self._request_state(MOTOR_IDS)
                elif (self._cycle_idx % 50) == 0:
                    stale = self._stale_motor_ids(stale_s=0.5)
                    if stale:
                        missing = self._request_state(stale)

                if not self._startup_wrap_checked and self._has_any_valid_state():
                    self._startup_wrap_checked = self._auto_correct_startup_wrap()
                if not self._action_initialized:
                    # Safe startup: hold current measured pose before accepting motion updates.
                    self._latch_action_from_state()
                    self._action_initialized = True
                if self._is_near_stop():
                    self._damping_active = True
                    self._send_damping_all()
                else:
                    self._damping_active = False
                    self._send_action_all()

            self._update_imu_state()
            self._publish_viz_state()
            self._log_state(missing)

            next_tick += period
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.perf_counter()

    def _has_any_valid_state(self) -> bool:
        with self._state_lock:
            return any(st.stamp > 0.0 for st in self.state.values())

    def _update_imu_state(self) -> None:
        with self._imu_lock:
            imu = self._imu
        if imu is None:
            return
        try:
            if hasattr(imu, "read_dict"):
                imu_state = imu.read_dict()
            elif hasattr(imu, "read"):
                raw = imu.read()
                imu_state = raw if isinstance(raw, dict) else {"data": raw}
            else:
                imu_state = {"error": "IMU object has no read/read_dict method"}
            q_xyzw: Optional[tuple[float, float, float, float]] = None
            q_raw = imu_state.get("quaternion_xyzw")
            if isinstance(q_raw, (list, tuple)) and len(q_raw) == 4:
                try:
                    q_xyzw = (float(q_raw[0]), float(q_raw[1]), float(q_raw[2]), float(q_raw[3]))
                except Exception:
                    q_xyzw = None
            stamp = imu_state.get("timestamp_s", time.time())
            try:
                stamp_s = float(stamp)
            except Exception:
                stamp_s = time.time()
            with self._imu_lock:
                self._imu_state = dict(imu_state)
                self._orientation_quaternion_xyzw = q_xyzw
                self._orientation_stamp_s = stamp_s
        except Exception as exc:
            with self._imu_lock:
                self._imu_state = {"time_s": time.time(), "error": f"{type(exc).__name__}: {exc}"}
                self._orientation_quaternion_xyzw = None
                self._orientation_stamp_s = time.time()

    def _stale_motor_ids(self, *, stale_s: float = 0.5) -> list[int]:
        now = time.time()
        stale: list[int] = []
        with self._state_lock:
            snapshot = {mid: st for mid, st in self.state.items()}
        for mid in MOTOR_IDS:
            st = snapshot[mid]
            if st.stamp <= 0.0 or (now - st.stamp) > float(stale_s):
                stale.append(mid)
        return stale

    # -------------------------
    # State request / decode
    # -------------------------
    def _request_state(self, motor_ids: Iterable[int], *, settle_s: float = 0.001) -> list[int]:
        ids = list(motor_ids)
        touched = set()

        for mid in ids:
            data = [0xFF] * 7 + [CAN_CMD_CLEAR_FAULT]
            self._send8(mid, data)
            self.protocol_usage["clear_fault"] += 1

        time.sleep(settle_s)

        max_msgs = max(4 * len(ids), 32)
        for _ in range(max_msgs):
            rx = self._recv_any(self.recv_timeout_s)
            if rx is None:
                break
            rec_mid = self._try_update_state_from_msg(rx)
            if rec_mid is not None:
                touched.add(rec_mid)
                if len(touched) == len(ids):
                    break

        return [mid for mid in ids if mid not in touched]

    def _try_update_state_from_msg(self, msg: can.Message) -> Optional[int]:
        try:
            raw = bytes(msg.data)
            mid = int(raw[0])
            spec = MOTORS.get(mid)
            if spec is None:
                return None
            motor_id, st = decode_state_frame(raw, pmax=spec.pmax_rad, vmax=spec.vmax_rad_s, tmax=spec.tmax_nm)
            st.stamp = time.time()

            with self._state_lock:
                self.state[motor_id] = st

            # Avoid premature estop before startup wrap correction has run once.
            if (not self._startup_wrap_checked):
                return motor_id

            if not self._state_in_bounds(motor_id, st.position_deg):
                reason = (
                    f"State out of limits on motor {motor_id}: {st.position_deg:.2f} deg, "
                    f"limits={self.joint_limits_deg.get(motor_id)}, margin={STATE_MARGIN_DEG}"
                )
                if not self._estop:
                    print(f"[WARN] E-STOP triggered: {reason}")
                self._estop = True
                self._estop_reason = reason
                self.disable_all()
            return motor_id
        except Exception:
            return None

    # -------------------------
    # MIT control / damping
    # -------------------------
    def _send_action_all(self) -> None:
        with self._action_lock:
            commands = {mid: cmd for mid, cmd in self.action.items()}

        send_raw = self._build_send_raw_from_joint_error(commands)
        for mid in MOTOR_IDS:
            cmd = commands[mid]
            cmd_eff = MotorCommand(
                position_deg=float(send_raw[mid]),
                velocity_deg_s=float(cmd.velocity_deg_s),
                torque_nm=float(cmd.torque_nm),
                kp=cmd.kp,
                kd=cmd.kd,
            )
            self._mit_command(mid, cmd_eff, clamp=False)
        self._drain_rx_states(max_wait_s=max(0.001, self.recv_timeout_s * len(MOTOR_IDS)), max_msgs=4 * len(MOTOR_IDS))

    def _build_send_raw_from_joint_error(self, commands: Dict[int, MotorCommand]) -> Dict[int, float]:
        """
        Build raw motor targets from joint-space error:
          - current joint state from measured raw motor state (same conversion as meshcat)
          - requested joint state from requested raw command
          - delta joint = requested - current
          - target joint = current + delta
          - convert target joint back to raw motor targets
        """
        with self._state_lock:
            snapshot = {mid: st for mid, st in self.state.items()}

        cur_raw = {mid: float(snapshot[mid].position_deg) for mid in MOTOR_IDS}
        req_raw = {mid: float(commands[mid].position_deg) for mid in MOTOR_IDS}

        q_cur_deg = self.motor_state_to_joint_state(cur_raw, output_radians=False, nq=12)
        q_req_deg = self.motor_state_to_joint_state(req_raw, output_radians=False, nq=12)
        dq_deg = q_req_deg - q_cur_deg
        q_tgt_deg = q_cur_deg + dq_deg

        tgt_raw = self.joint_state_to_motor_state(q_tgt_deg, input_radians=False, output_space="raw")
        out: Dict[int, float] = {}
        for mid in MOTOR_IDS:
            st = snapshot[mid]
            raw = float(tgt_raw[mid])
            if st.stamp > 0.0:
                raw = self._resolve_raw_target_near_current(mid, raw, float(st.position_deg))
            out[mid] = float(raw)

        return out

    def _latch_action_from_state(self) -> None:
        with self._state_lock:
            snapshot = {mid: st for mid, st in self.state.items()}
        with self._action_lock:
            for mid in MOTOR_IDS:
                prev = self.action[mid]
                st = snapshot[mid]
                self.action[mid] = MotorCommand(
                    position_deg=float(st.position_deg),
                    velocity_deg_s=0.0,
                    torque_nm=0.0,
                    kp=prev.kp,
                    kd=prev.kd,
                )

    def _with_fields(
        self,
        prev: MotorCommand,
        position_deg: float,
        velocity_deg_s: float,
        torque_nm: float,
        kp: Optional[float],
        kd: Optional[float],
    ) -> MotorCommand:
        return MotorCommand(
            position_deg=float(position_deg),
            velocity_deg_s=float(velocity_deg_s),
            torque_nm=float(torque_nm),
            kp=prev.kp if kp is None else kp,
            kd=prev.kd if kd is None else kd,
        )

    def _joint_index_map(self) -> Dict[str, Dict[str, int]]:
        return {
            "left": {"hipz": 0, "hipx": 1, "hipy": 2, "knee": 3, "ankle_pitch": 4, "ankle_roll": 5},
            "right": {"hipz": 6, "hipx": 7, "hipy": 8, "knee": 9, "ankle_pitch": 10, "ankle_roll": 11},
        }

    def _apply_joint_side_updates(
        self,
        values: np.ndarray,
        *,
        left: Optional[Dict[str, float]] = None,
        right: Optional[Dict[str, float]] = None,
    ) -> None:
        joint_index = self._joint_index_map()
        left = left or {}
        right = right or {}
        for side_name, side_cmd in (("left", left), ("right", right)):
            for key, value in side_cmd.items():
                idx = joint_index[side_name].get(key)
                if idx is None:
                    continue
                values[idx] = float(value)

    def _joint_vel_tau_to_motor_raw(
        self,
        qd_deg_s: np.ndarray,
        tau_nm: np.ndarray,
    ) -> tuple[Dict[int, float], Dict[int, float]]:
        """
        Map model joint-space (qd, tau) to motor raw-space.
        Direct joints:
          q_cal = sign * q_raw + offset  -> qd_raw = qd_cal / sign
          power consistency -> tau_raw = tau_cal * sign
        Coupled ankles:
          pitch = sp*(a1-a2)/2, roll = sr*(a1+a2)/2
          solve for (a1_dot, a2_dot), and tau_motor = J^T * tau_joint
        """
        qd_raw: Dict[int, float] = {}
        tau_raw: Dict[int, float] = {}

        direct = {
            0: 1, 1: 2, 2: 3, 3: 4,
            6: 7, 7: 8, 8: 9, 9: 10,
        }
        for qi, mid in direct.items():
            s = float(self.motor_sign[mid])
            if abs(s) < 1e-9:
                raise ValueError(f"Invalid motor sign for m{mid}: {s}")
            qd_raw[mid] = float(qd_deg_s[qi] / s)
            tau_raw[mid] = float(tau_nm[qi] * s)

        sp_l = float(ANKLE_COUPLING_CALIBRATION_LEFT["pitch"]["sign"])
        sr_l = float(ANKLE_COUPLING_CALIBRATION_LEFT["roll"]["sign"])
        u_l = float(qd_deg_s[4]) / sp_l
        v_l = float(qd_deg_s[5]) / sr_l
        qd_raw[5] = u_l + v_l
        qd_raw[6] = v_l - u_l
        tau_raw[5] = 0.5 * sp_l * float(tau_nm[4]) + 0.5 * sr_l * float(tau_nm[5])
        tau_raw[6] = -0.5 * sp_l * float(tau_nm[4]) + 0.5 * sr_l * float(tau_nm[5])

        sp_r = float(ANKLE_COUPLING_CALIBRATION_RIGHT["pitch"]["sign"])
        sr_r = float(ANKLE_COUPLING_CALIBRATION_RIGHT["roll"]["sign"])
        u_r = float(qd_deg_s[10]) / sp_r
        v_r = float(qd_deg_s[11]) / sr_r
        qd_raw[11] = u_r + v_r
        qd_raw[12] = v_r - u_r
        tau_raw[11] = 0.5 * sp_r * float(tau_nm[10]) + 0.5 * sr_r * float(tau_nm[11])
        tau_raw[12] = -0.5 * sp_r * float(tau_nm[10]) + 0.5 * sr_r * float(tau_nm[11])
        return qd_raw, tau_raw

    def _ankle_pitch_roll_from_actions(self, a1_cmd: MotorCommand, a2_cmd: MotorCommand, *, side_name: str) -> tuple[float, float]:
        cfg = ANKLE_COUPLING_CALIBRATION_LEFT if side_name == "left" else ANKLE_COUPLING_CALIBRATION_RIGHT
        sign_p = float(cfg["pitch"]["sign"])
        off_p = float(cfg["pitch"]["offset_deg"])
        sign_r = float(cfg["roll"]["sign"])
        off_r = float(cfg["roll"]["offset_deg"])
        a1 = float(a1_cmd.position_deg)
        a2 = float(a2_cmd.position_deg)
        pitch = sign_p * ((a1 - a2) / 2.0) + off_p
        roll = sign_r * ((a1 + a2) / 2.0) + off_r
        return pitch, roll

    def _ankle_pitch_roll_from_cal_values(self, a1: float, a2: float, *, side_name: str) -> tuple[float, float]:
        cfg = ANKLE_COUPLING_CALIBRATION_LEFT if side_name == "left" else ANKLE_COUPLING_CALIBRATION_RIGHT
        sign_p = float(cfg["pitch"]["sign"])
        off_p = float(cfg["pitch"]["offset_deg"])
        sign_r = float(cfg["roll"]["sign"])
        off_r = float(cfg["roll"]["offset_deg"])
        pitch = sign_p * ((float(a1) - float(a2)) / 2.0) + off_p
        roll = sign_r * ((float(a1) + float(a2)) / 2.0) + off_r
        return pitch, roll

    def _ankle_motors_from_pitch_roll(self, pitch: float, roll: float, *, side_name: str) -> tuple[float, float]:
        cfg = ANKLE_COUPLING_CALIBRATION_LEFT if side_name == "left" else ANKLE_COUPLING_CALIBRATION_RIGHT
        sign_p = float(cfg["pitch"]["sign"])
        off_p = float(cfg["pitch"]["offset_deg"])
        sign_r = float(cfg["roll"]["sign"])
        off_r = float(cfg["roll"]["offset_deg"])
        p = (float(pitch) - off_p) / sign_p
        r = (float(roll) - off_r) / sign_r
        a1 = p + r
        a2 = r - p
        return a1, a2

    def _current_ankle_pitch(self, *, side_name: str) -> Optional[float]:
        pr = self._ankle_pitch_roll_from_state(side_name=side_name)
        if pr is None:
            return None
        return pr[0]

    def _ankle_pitch_roll_from_state(self, *, side_name: str) -> Optional[tuple[float, float]]:
        """
        Compute ankle joint-space values from current measured states.
        Must stay consistent with visualization path.
        """
        ids = (5, 6) if side_name == "left" else (11, 12)
        with self._state_lock:
            s1 = self.state[ids[0]]
            s2 = self.state[ids[1]]
        if s1.stamp <= 0.0 or s2.stamp <= 0.0:
            return None
        a1 = self._raw_to_cal(ids[0], s1.position_deg)
        a2 = self._raw_to_cal(ids[1], s2.position_deg)
        return self._ankle_pitch_roll_from_cal_values(a1, a2, side_name=side_name)

    def _send_damping_all(self) -> None:
        with self._state_lock:
            snapshot = {mid: st for mid, st in self.state.items()}
        for mid in MOTOR_IDS:
            st = snapshot[mid]
            damping_cmd = MotorCommand(
                position_deg=float(st.position_deg),
                velocity_deg_s=0.0,
                torque_nm=0.0,
                kp=0.0,
                kd=DAMPING_KD,
            )
            self._mit_command(mid, damping_cmd, clamp=False)
            self.protocol_usage["damping"] += 1
        self._drain_rx_states(max_wait_s=max(0.001, self.recv_timeout_s * len(MOTOR_IDS)), max_msgs=4 * len(MOTOR_IDS))

    def _mit_command(self, motor_id: int, cmd: MotorCommand, *, clamp: bool) -> None:
        if self._estop:
            return

        spec = MOTORS[motor_id]
        gains = self.gains[motor_id]
        kp = gains.kp if cmd.kp is None else float(cmd.kp)
        kd = gains.kd if cmd.kd is None else float(cmd.kd)
        req_raw = float(cmd.position_deg)

        # Safety gate: reject large command jumps from current measured state.
        with self._state_lock:
            st = self.state[motor_id]
        if st.stamp <= 0.0:
            now = time.time()
            if now - self._last_jump_warn_s[motor_id] > 0.5:
                print(f"[WARN] m{motor_id}: no valid state yet, skipping command send")
                self._last_jump_warn_s[motor_id] = now
            return

        cur_raw = float(st.position_deg)
        pos_raw = req_raw
        pos_raw = self._resolve_raw_target_near_current(motor_id, pos_raw, cur_raw)
        if clamp and (not self._raw_command_in_limits(motor_id, pos_raw, ref_raw_deg=cur_raw)):
            now = time.time()
            if now - self._last_jump_warn_s[motor_id] > 0.5:
                lo, hi = self._raw_limits_near_ref(motor_id, self._raw_command_limits().get(motor_id, (-np.inf, np.inf)), cur_raw)
                print(
                    f"[WARN] m{motor_id}: command out of limits, req_raw={req_raw:.1f}, tgt_raw={pos_raw:.1f}, "
                    f"allowed=({lo + COMMAND_MARGIN_DEG:.1f}, {hi - COMMAND_MARGIN_DEG:.1f}), skipping send"
                )
                self._last_jump_warn_s[motor_id] = now
            return
        delta = abs(pos_raw - cur_raw)
        if delta > self.max_cmd_delta_deg:
            now = time.time()
            if now - self._last_jump_warn_s[motor_id] > 0.5:
                print(
                    f"[WARN] m{motor_id}: command jump too large ({delta:.1f} deg > {self.max_cmd_delta_deg:.1f} deg), "
                    f"cur_raw={cur_raw:.1f}, req_raw={req_raw:.1f}, tgt_raw={pos_raw:.1f}, skipping send"
                )
                self._last_jump_warn_s[motor_id] = now
            return

        payload = pack_mit_command(
            position_deg=pos_raw,
            velocity_deg_s=float(cmd.velocity_deg_s),
            kp=kp,
            kd=kd,
            torque_nm=float(cmd.torque_nm),
            pmax=spec.pmax_rad,
            vmax=spec.vmax_rad_s,
            tmax=spec.tmax_nm,
        )
        self._send8(motor_id, payload)
        self.protocol_usage["mit_control"] += 1

    def _passes_action_update_jump_guard(self, target_raw: Dict[int, float]) -> bool:
        """
        Action-update jump check against current global measured state.
        """
        with self._state_lock:
            snapshot = {mid: st for mid, st in self.state.items()}
        for mid in MOTOR_IDS:
            st = snapshot[mid]
            if st.stamp <= 0.0:
                continue
            cur_raw = float(st.position_deg)
            tgt_raw = self._resolve_raw_target_near_current(mid, float(target_raw[mid]), cur_raw)
            if not self._raw_command_in_limits(mid, tgt_raw, ref_raw_deg=cur_raw):
                lo, hi = self._raw_limits_near_ref(mid, self._raw_command_limits().get(mid, (-np.inf, np.inf)), cur_raw)
                print(
                    f"[WARN] m{mid}: action update out of limits, tgt_raw={tgt_raw:.1f}, "
                    f"allowed=({lo + COMMAND_MARGIN_DEG:.1f}, {hi - COMMAND_MARGIN_DEG:.1f}), rejecting action update"
                )
                return False
            delta = abs(tgt_raw - cur_raw)
            if delta > self.max_cmd_delta_deg:
                print(
                    f"[WARN] m{mid}: action update jump too large ({delta:.1f} deg > {self.max_cmd_delta_deg:.1f} deg), "
                    f"cur_raw={cur_raw:.1f}, tgt_raw={tgt_raw:.1f}, rejecting action update"
                )
                return False
        return True

    def _current_action_motor_raw(self) -> Dict[int, float]:
        """
        Current action expressed in raw motor-space (deg).
        """
        with self._action_lock:
            action_snapshot = {mid: cmd for mid, cmd in self.action.items()}
        return {mid: float(action_snapshot[mid].position_deg) for mid in MOTOR_IDS}

    def _effective_send_raw_map(self, req_raw_map: Dict[int, float]) -> Dict[int, float]:
        """
        Effective raw send targets from requested raw targets, applying:
          1) raw clamp
          2) wrap branch resolution near current measured raw state
        """
        out: Dict[int, float] = {}
        with self._state_lock:
            snapshot = {mid: st for mid, st in self.state.items()}
        for mid in MOTOR_IDS:
            req_raw = float(req_raw_map[mid])
            st = snapshot[mid]
            tgt_raw = req_raw
            if st.stamp > 0.0:
                tgt_raw = self._resolve_raw_target_near_current(mid, tgt_raw, float(st.position_deg))
            out[mid] = float(tgt_raw)
        return out

    def _raw_command_in_limits(self, motor_id: int, pos_raw_deg: float, ref_raw_deg: Optional[float] = None) -> bool:
        lim_raw = self._raw_command_limits().get(int(motor_id))
        if lim_raw is None:
            return True
        lo, hi = self._raw_limits_near_ref(int(motor_id), lim_raw, ref_raw_deg)
        lo_safe = lo + COMMAND_MARGIN_DEG
        hi_safe = hi - COMMAND_MARGIN_DEG
        if lo_safe > hi_safe:
            lo_safe, hi_safe = lo, hi
        x = float(pos_raw_deg)
        return (lo_safe <= x <= hi_safe)

    def _drain_rx_states(self, *, max_wait_s: float, max_msgs: int = 64) -> int:
        """
        Drain and decode state frames after a batched TX phase.

        Non-blocking behavior:
          - stop immediately when no frame is ready
          - leave late frames for next control iteration
        """
        n = 0
        deadline = time.perf_counter() + max(0.0, float(max_wait_s))
        while n < int(max_msgs):
            if time.perf_counter() >= deadline:
                break
            rx = self._recv_any(0.0)
            if rx is None:
                break
            self._try_update_state_from_msg(rx)
            n += 1
        return n

    # -------------------------
    # Safety
    # -------------------------
    def _state_in_bounds(self, motor_id: int, pos_deg: float) -> bool:
        lim = self.command_limits_cal_deg.get(motor_id)
        if lim is None:
            return True
        lo, hi = lim
        value = self._raw_to_cal_near_interval(motor_id, float(pos_deg), lo, hi)
        return (lo - STATE_MARGIN_DEG) <= value <= (hi + STATE_MARGIN_DEG)

    def _clamp_command(self, motor_id: int, pos_deg: float) -> float:
        lim = self.command_limits_cal_deg.get(motor_id)
        if lim is None:
            return float(pos_deg)
        lo, hi = lim
        lo_safe = lo + COMMAND_MARGIN_DEG
        hi_safe = hi - COMMAND_MARGIN_DEG
        if lo_safe > hi_safe:
            lo_safe, hi_safe = lo, hi
        return float(np.clip(pos_deg, lo_safe, hi_safe))

    def _raw_command_limits(self) -> Dict[int, tuple[float, float]]:
        out: Dict[int, tuple[float, float]] = {}
        for mid, lim in self.joint_limits_deg.items():
            lo, hi = float(lim[0]), float(lim[1])
            if self._limit_source == "raw":
                out[mid] = (min(lo, hi), max(lo, hi))
            else:
                lo_raw = self._cal_to_raw(mid, lo)
                hi_raw = self._cal_to_raw(mid, hi)
                out[mid] = (min(lo_raw, hi_raw), max(lo_raw, hi_raw))
        return out

    def _raw_limits_near_ref(
        self,
        motor_id: int,
        lim_raw: tuple[float, float],
        ref_raw_deg: Optional[float],
    ) -> tuple[float, float]:
        """
        Resolve a raw limit interval to the 360-branch nearest a reference raw value.
        """
        lo, hi = float(lim_raw[0]), float(lim_raw[1])
        if ref_raw_deg is None:
            return lo, hi
        ref = float(ref_raw_deg)
        best = (lo, hi)
        best_dist = float("inf")
        for k in range(-4, 5):
            cand_lo = lo + 360.0 * k
            cand_hi = hi + 360.0 * k
            center = 0.5 * (cand_lo + cand_hi)
            dist = abs(center - ref)
            if dist < best_dist:
                best = (cand_lo, cand_hi)
                best_dist = dist
        return best

    def _recompute_command_limits(self) -> None:
        """
        Command API is in calibrated/model-aligned degrees.
        - if state limits are raw: convert raw limits -> calibrated limits for clamping commands
        - if state limits are calibrated: use them directly
        """
        out: Dict[int, tuple[float, float]] = {}
        for mid, lim in self.joint_limits_deg.items():
            lo, hi = float(lim[0]), float(lim[1])
            if self._limit_source == "raw":
                lo_cal = self._raw_to_cal(mid, lo)
                hi_cal = self._raw_to_cal(mid, hi)
                out[mid] = (min(lo_cal, hi_cal), max(lo_cal, hi_cal))
            else:
                out[mid] = (min(lo, hi), max(lo, hi))
        self.command_limits_cal_deg = out

    def _is_near_stop(self) -> bool:
        with self._state_lock:
            snapshot = {mid: st for mid, st in self.state.items()}
        for mid, st in snapshot.items():
            lim = self.command_limits_cal_deg.get(mid)
            if lim is None:
                continue
            lo, hi = lim
            value = self._raw_to_cal_near_interval(mid, float(st.position_deg), lo, hi)
            if value <= lo + NEAR_STOP_MARGIN_DEG:
                return True
            if value >= hi - NEAR_STOP_MARGIN_DEG:
                return True
        return False

    # -------------------------
    # Logging
    # -------------------------
    def _open_log(self) -> None:
        if self._log_fp is not None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fp = self.log_path.open("w", newline="")
        self._log_writer = csv.writer(self._log_fp)

        header = ["time_s", "mode", "estop", "estop_reason", "damping_active", "missing_ids"]
        for mid in MOTOR_IDS:
            header += [
                f"m{mid}_pos_deg",
                f"m{mid}_target_pos_deg",
                f"m{mid}_vel_deg_s",
                f"m{mid}_tau_nm",
                f"m{mid}_temp_c",
            ]
        self._log_writer.writerow(header)
        self._log_fp.flush()

    def _log_state(self, missing_ids: Iterable[int]) -> None:
        if self._log_writer is None:
            return
        with self._state_lock:
            snapshot = {mid: st for mid, st in self.state.items()}
        with self._action_lock:
            action_snapshot = {mid: cmd for mid, cmd in self.action.items()}
        row = [time.time(), self.mode, int(self._estop), self._estop_reason, int(self._damping_active), ",".join(map(str, missing_ids))]
        for mid in MOTOR_IDS:
            st = snapshot[mid]
            row += [st.position_deg, action_snapshot[mid].position_deg, st.velocity_deg_s, st.torque_nm, st.temp_mos_c]
        self._log_writer.writerow(row)
        self._log_fp.flush()

    def _close_log(self) -> None:
        if self._log_fp is not None:
            self._log_fp.close()
        self._log_fp = None
        self._log_writer = None

    # -------------------------
    # Visualization
    # -------------------------
    def _start_viz_thread(self) -> None:
        if self._viz is None:
            return
        if self._viz_thread is not None and self._viz_thread.is_alive():
            return
        self._viz_running = True
        self._viz_thread = threading.Thread(target=self._viz_loop, daemon=True)
        self._viz_thread.start()

    def _stop_viz_thread(self) -> None:
        self._viz_running = False
        if self._viz_thread is not None:
            self._viz_thread.join(timeout=1.0)
        self._viz_thread = None

    def _publish_viz_state(self) -> None:
        if self._viz is None:
            return
        with self._state_lock:
            snap = {mid: float(self.state[mid].position_deg) for mid in MOTOR_IDS}
        with self._viz_data_lock:
            self._viz_raw_latest = snap

    def _viz_loop(self) -> None:
        period = 1.0 / max(1.0, float(self._viz_hz))
        while self._viz_running:
            if self._viz is None:
                time.sleep(period)
                continue
            with self._viz_data_lock:
                raw = self._viz_raw_latest
            if raw is not None:
                try:
                    q = self.motor_state_to_joint_state(raw, output_radians=True, nq=int(self._model_nq))
                    self._viz.display(q)
                except Exception:
                    pass
            time.sleep(period)

    def motor_state_to_joint_state(
        self,
        motor_raw_deg: Dict[int, float],
        *,
        output_radians: bool = True,
        nq: Optional[int] = None,
    ) -> np.ndarray:
        """
        Convert per-motor raw encoder positions (deg) to model joint state.
        Joint order:
        [L_hipz, L_hipx, L_hipy, L_knee, L_ankle_pitch, L_ankle_roll,
         R_hipz, R_hipx, R_hipy, R_knee, R_ankle_pitch, R_ankle_roll]
        """
        out_nq = int(self._model_nq if nq is None else nq)
        q_deg = np.zeros(max(12, out_nq))

        # Left leg
        l_hipz = self._raw_to_cal(1, motor_raw_deg[1])
        l_hipx = self._raw_to_cal(2, motor_raw_deg[2])
        l_hipy = self._raw_to_cal(3, motor_raw_deg[3])
        l_knee = self._raw_to_cal(4, motor_raw_deg[4])
        l_a1 = self._raw_to_cal(5, motor_raw_deg[5])
        l_a2 = self._raw_to_cal(6, motor_raw_deg[6])
        l_pitch, l_roll = self._ankle_pitch_roll_from_cal_values(l_a1, l_a2, side_name="left")
        q_deg[0:6] = [l_hipz, l_hipx, l_hipy, l_knee, l_pitch, l_roll]

        # Right leg
        r_hipz = self._raw_to_cal(7, motor_raw_deg[7])
        r_hipx = self._raw_to_cal(8, motor_raw_deg[8])
        r_hipy = self._raw_to_cal(9, motor_raw_deg[9])
        r_knee = self._raw_to_cal(10, motor_raw_deg[10])
        r_a1 = self._raw_to_cal(11, motor_raw_deg[11])
        r_a2 = self._raw_to_cal(12, motor_raw_deg[12])
        r_pitch, r_roll = self._ankle_pitch_roll_from_cal_values(r_a1, r_a2, side_name="right")
        q_deg[6:12] = [r_hipz, r_hipx, r_hipy, r_knee, r_pitch, r_roll]

        if output_radians:
            return np.deg2rad(q_deg[:out_nq])
        return q_deg[:out_nq]

    def joint_state_to_motor_state(
        self,
        q_joint,
        *,
        input_radians: bool = True,
        output_space: str = "calibrated",
    ) -> Dict[int, float]:
        """
        Inverse mapping from model joint state to per-motor state.
        output_space:
          - 'calibrated': model-aligned motor angles (deg)
          - 'raw': raw motor angles (deg)
        """
        if output_space not in ("calibrated", "raw"):
            raise ValueError("output_space must be 'calibrated' or 'raw'")

        q = np.asarray(q_joint, dtype=float).reshape(-1)
        if input_radians:
            q_deg = np.rad2deg(q)
        else:
            q_deg = q

        if q_deg.size < 12:
            raise ValueError("q_joint must contain at least 12 joints")

        out_cal: Dict[int, float] = {
            1: float(q_deg[0]),
            2: float(q_deg[1]),
            3: float(q_deg[2]),
            4: float(q_deg[3]),
            7: float(q_deg[6]),
            8: float(q_deg[7]),
            9: float(q_deg[8]),
            10: float(q_deg[9]),
        }
        l_a1, l_a2 = self._ankle_motors_from_pitch_roll(float(q_deg[4]), float(q_deg[5]), side_name="left")
        r_a1, r_a2 = self._ankle_motors_from_pitch_roll(float(q_deg[10]), float(q_deg[11]), side_name="right")
        out_cal[5] = float(l_a1)
        out_cal[6] = float(l_a2)
        out_cal[11] = float(r_a1)
        out_cal[12] = float(r_a2)

        if output_space == "calibrated":
            return out_cal
        return {mid: self._cal_to_raw(mid, cal_deg) for mid, cal_deg in out_cal.items()}

    def test_inverse_consistency(self, *, n_samples: int = 200, seed: int = 0, tol_deg: float = 1e-6) -> Dict[str, float]:
        """
        Check consistency of:
          q_joint(deg) -> motor_cal(deg) -> motor_raw(deg) -> q_joint(deg)
        Returns summary metrics and raises AssertionError if max error > tol_deg.
        """
        rng = np.random.default_rng(int(seed))
        max_err = 0.0
        mean_err_acc = 0.0
        n_vals = 0

        for _ in range(int(n_samples)):
            q = rng.uniform(low=-120.0, high=120.0, size=12)
            m_raw = self.joint_state_to_motor_state(q, input_radians=False, output_space="raw")
            q_back = self.motor_state_to_joint_state(m_raw, output_radians=False, nq=12)
            err = np.abs(q_back - q)
            max_err = max(max_err, float(np.max(err)))
            mean_err_acc += float(np.sum(err))
            n_vals += int(err.size)

        mean_err = mean_err_acc / max(1, n_vals)
        result = {
            "samples": float(n_samples),
            "max_abs_err_deg": float(max_err),
            "mean_abs_err_deg": float(mean_err),
            "tol_deg": float(tol_deg),
        }
        if max_err > float(tol_deg):
            raise AssertionError(
                f"Inverse consistency failed: max_abs_err_deg={max_err:.6g} > tol_deg={float(tol_deg):.6g}"
            )
        return result

    def _state_to_q_model(self) -> np.ndarray:
        with self._state_lock:
            s = {mid: st for mid, st in self.state.items()}
        raw = {mid: float(s[mid].position_deg) for mid in MOTOR_IDS}
        return self.motor_state_to_joint_state(raw, output_radians=True, nq=int(self._model_nq))

    # -------------------------
    # Low-level CAN send
    # -------------------------
    def _send8(self, motor_id: int, data8: Iterable[int]) -> None:
        msg = can.Message(arbitration_id=int(motor_id), data=list(data8), is_extended_id=False)
        self._bus_for_motor(motor_id).send(msg)

    def _send_cmd_ff(self, motor_id: int, cmd: int) -> None:
        data = [0xFF] * 8
        data[7] = int(cmd)
        self._send8(motor_id, data)

    def _bus_for_motor(self, motor_id: int):
        if int(motor_id) in CAN0_MOTOR_IDS:
            return self.bus_can0
        if int(motor_id) in CAN1_MOTOR_IDS:
            return self.bus_can1
        raise ValueError(f"Unsupported motor id {motor_id}, expected 1..12")

    def _recv_any(self, timeout_s: float):
        if float(timeout_s) <= 0.0:
            msg = self.bus_can0.recv(0.0)
            if msg is not None:
                return msg
            return self.bus_can1.recv(0.0)

        deadline = time.perf_counter() + max(0.0, float(timeout_s))
        while time.perf_counter() < deadline:
            msg = self.bus_can0.recv(0.0)
            if msg is not None:
                return msg
            msg = self.bus_can1.recv(0.0)
            if msg is not None:
                return msg
            time.sleep(0.0001)
        return None

    def _shutdown_buses(self) -> None:
        for bus in (self.bus_can0, self.bus_can1):
            try:
                bus.shutdown()
            except Exception:
                pass

    def _auto_correct_startup_wrap(self) -> bool:
        """
        One-shot startup correction for wrapped encoder values (e.g. 385 deg instead of -5 deg).
        It adjusts per-motor calibration offset by multiples of 360 deg to bring calibrated
        position close to configured limits, and optionally shifts limits by same delta.
        """
        with self._state_lock:
            snapshot = {mid: st for mid, st in self.state.items()}

        any_valid_state = any(st.stamp > 0.0 for st in snapshot.values())
        if not any_valid_state:
            return False

        updates = []
        for mid, st in snapshot.items():
            lim = self.joint_limits_deg.get(mid)
            if lim is None:
                continue

            lo, hi = lim
            cur_cal = self._raw_to_cal(mid, st.position_deg)
            center = 0.5 * (lo + hi)

            def dist_to_interval(x: float, a: float, b: float) -> float:
                if x < a:
                    return a - x
                if x > b:
                    return x - b
                return 0.0

            best_k = 0
            best_dist = dist_to_interval(cur_cal, lo, hi)
            best_center = abs(cur_cal - center)

            for k in range(-3, 4):
                cand = cur_cal + 360.0 * k
                cand_dist = dist_to_interval(cand, lo, hi)
                cand_center = abs(cand - center)
                if (cand_dist < best_dist) or (cand_dist == best_dist and cand_center < best_center):
                    best_k = k
                    best_dist = cand_dist
                    best_center = cand_center

            if best_k != 0:
                delta = 360.0 * best_k
                self.motor_offset_deg[mid] += delta
                if self._auto_shift_limits_with_wrap:
                    self.joint_limits_deg[mid] = (lo + delta, hi + delta)
                updates.append((mid, delta))

        if updates:
            # Offsets/limits changed -> refresh command-side limits in calibrated space.
            self._recompute_command_limits()
            upd_txt = ", ".join([f"m{mid}:{delta:+.1f}deg" for mid, delta in updates])
            print(f"[startup-wrap-correction] Applied offset/limit shifts -> {upd_txt}")

        return True

    def _normalize_to_limits(self, value_deg: float, lo: float, hi: float) -> float:
        """Return value + k*360 that best matches [lo, hi]."""
        x = float(value_deg)
        center = 0.5 * (lo + hi)
        best = x
        best_dist = float("inf")
        best_center = float("inf")
        for k in range(-3, 4):
            cand = x + 360.0 * k
            if cand < lo:
                dist = lo - cand
            elif cand > hi:
                dist = cand - hi
            else:
                dist = 0.0
            cdist = abs(cand - center)
            if (dist < best_dist) or (dist == best_dist and cdist < best_center):
                best = cand
                best_dist = dist
                best_center = cdist
        return best

    def _raw_to_cal(self, motor_id: int, raw_deg: float) -> float:
        return float(self.motor_sign[int(motor_id)] * float(raw_deg) + self.motor_offset_deg[int(motor_id)])

    def _cal_to_raw(self, motor_id: int, cal_deg: float) -> float:
        sign = float(self.motor_sign[int(motor_id)])
        if abs(sign) < 1e-9:
            raise ValueError(f"Invalid MOTOR_SIGN for motor {motor_id}: {sign}")
        return float((float(cal_deg) - self.motor_offset_deg[int(motor_id)]) / sign)

    def _raw_to_cal_near(self, motor_id: int, raw_deg: float, ref_cal_deg: float) -> float:
        """
        Convert raw->cal while resolving 360-wrap in calibrated space near a reference.
        """
        base = self._raw_to_cal(motor_id, raw_deg)
        step = 360.0 * float(self.motor_sign[int(motor_id)])
        best = base
        best_dist = abs(base - float(ref_cal_deg))
        for k in range(-4, 5):
            cand = base + step * k
            dist = abs(cand - float(ref_cal_deg))
            if dist < best_dist:
                best = cand
                best_dist = dist
        return float(best)

    def _resolve_raw_target_near_current(self, motor_id: int, raw_target_deg: float, raw_current_deg: float) -> float:
        """
        Pick equivalent raw target (raw_target + 360*k) closest to current raw.
        Prefer candidates inside MIT encoding range when possible.
        """
        spec = MOTORS[int(motor_id)]
        pmax_deg = float(np.degrees(spec.pmax_rad))
        cur = float(raw_current_deg)
        tgt = float(raw_target_deg)

        best = tgt
        best_dist = abs(tgt - cur)
        found_in_range = (-pmax_deg <= tgt <= pmax_deg)

        for k in range(-4, 5):
            cand = tgt + 360.0 * k
            dist = abs(cand - cur)
            in_range = (-pmax_deg <= cand <= pmax_deg)
            if found_in_range:
                if in_range and dist < best_dist:
                    best = cand
                    best_dist = dist
            else:
                if in_range:
                    best = cand
                    best_dist = dist
                    found_in_range = True
                elif dist < best_dist:
                    best = cand
                    best_dist = dist
        return float(best)

    def _raw_to_cal_near_interval(self, motor_id: int, raw_deg: float, lo: float, hi: float) -> float:
        """
        Convert raw->cal while resolving 360-wrap to best match a calibrated interval.
        """
        base = self._raw_to_cal(motor_id, raw_deg)
        step = 360.0 * float(self.motor_sign[int(motor_id)])
        center = 0.5 * (float(lo) + float(hi))
        best = base
        best_dist = float("inf")
        best_center = float("inf")
        for k in range(-4, 5):
            cand = base + step * k
            if cand < lo:
                dist = lo - cand
            elif cand > hi:
                dist = cand - hi
            else:
                dist = 0.0
            cdist = abs(cand - center)
            if (dist < best_dist) or (dist == best_dist and cdist < best_center):
                best = cand
                best_dist = dist
                best_center = cdist
        return float(best)


if __name__ == "__main__":
    robot = BipedalRobotController(control_hz=100.0, log_path="bipedal_state_log.csv")
    # mode 1: state streaming + meshcat
    robot.attach_default_meshcat()
    # robot.start(mode="state_only", auto_enable=False)

    # mode 2: command streaming
    # robot.start(mode="control", auto_enable=True)
    # robot.set_action(left={"hipz": 180.0}, right={"hipz": 180.0})
    pass
