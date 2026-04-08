from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import threading
import time

import numpy as np

from leg_test.mit import MotorState
from root_constant import (
    ANKLE_COUPLING_CALIBRATION_LEFT,
    ANKLE_COUPLING_CALIBRATION_RIGHT,
    JOINT_LIMITS_DEG,
    MOTOR_IDS,
    MOTOR_OFFSET_DEG,
    MOTOR_SIGN,
)

try:
    import mujoco
except Exception:  # pragma: no cover - optional runtime dependency
    mujoco = None

try:
    import mujoco.viewer as mj_viewer
except Exception:  # pragma: no cover - optional runtime dependency
    mj_viewer = None

DEFAULT_MJCF_PATH = Path("bipedal_plateform_no_arms/mjcf/sim_scene_safe.xml")

# PD gains tuned for MuJoCo sim (from lerobot_humanoid_no_arms_constants.py).
# These are higher than real-robot gains because the sim uses position actuators.
SIM_PD_GAINS_BY_MOTOR_ID: Dict[int, tuple[float, float]] = {
    1:  (40.0,  1.0),   # hipz_left
    2:  (110.0, 1.0),   # hipx_left
    3:  (110.0, 1.0),   # hipy_left
    4:  (110.0, 1.0),   # knee_left
    5:  (60.0,  2.0),   # ankle_a left
    6:  (60.0,  2.0),   # ankle_b left
    7:  (40.0,  1.0),   # hipz_right
    8:  (110.0, 1.0),   # hipx_right
    9:  (110.0, 1.0),   # hipy_right
    10: (110.0, 1.0),   # knee_right
    11: (60.0,  2.0),   # ankle_a right
    12: (60.0,  2.0),   # ankle_b right
}
LEROBOT_SIM_GAINS_BY_MOTOR_ID = SIM_PD_GAINS_BY_MOTOR_ID  # backward-compat alias

# Knees-bent standing reference pose (MuJoCo joint names).
# Used as a stable starting configuration for RL training resets.
SIM_KNEES_BENT_POSE_RAD: Dict[str, float] = {
    "hipz_right":   0.0,
    "hipx_right":   0.0,
    "hipy_right":   float(np.deg2rad( 20.0535)),
    "knee_right":   float(np.deg2rad( 40.1070)),
    "ankley_right": float(np.deg2rad( 20.0535)),
    "anklex_right": 0.0,
    "hipz_left":    0.0,
    "hipx_left":    0.0,
    "hipy_left":    float(np.deg2rad(-20.0535)),
    "knee_left":    float(np.deg2rad( 40.1070)),
    "ankley_left":  float(np.deg2rad(-20.0535)),
    "anklex_left":  0.0,
}
LEROBOT_KNEES_BENT_REF_POSE_RAD = SIM_KNEES_BENT_POSE_RAD  # backward-compat alias

# Drop height when spawning without a hardcoded pose (free-fall to ground).
SIM_DROP_HEIGHT_M = 2.5
LEROBOT_ENV_INIT_BASE_HEIGHT_M = SIM_DROP_HEIGHT_M  # backward-compat alias

# Hardcoded spawn pose recorded from a stable MJLab episode.
# qpos free-joint: [x, y, z, qw, qx, qy, qz]  (MuJoCo convention)
SIM_SPAWN_QPOS = np.array(
    [
        0.0032666486222296953,
        2.461623626004439e-05,
        0.77,                       # z ≈ 0.77 m (standing height)
        0.9996728897094727,         # qw ≈ 1 (nearly upright)
        6.715940253343433e-05,
        -0.02557525411248207,
        7.01636599842459e-05,
    ],
    dtype=float,
)
MJLAB_HARDCODED_SPAWN_QPOS_FREE = SIM_SPAWN_QPOS  # backward-compat alias

# Corresponding free-joint velocity at spawn
SIM_SPAWN_QVEL = np.array(
    [
        0.22695069015026093,
        0.0015118308365345001,
        0.1054210290312767,
        0.009430008940398693,
        -3.501528501510602,
        0.00918684620410204,
    ],
    dtype=float,
)
MJLAB_HARDCODED_SPAWN_QVEL_FREE = SIM_SPAWN_QVEL  # backward-compat alias


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


class SimBipedalRobotController:
    MODES = ("state_only", "control")

    def __init__(
        self,
        *,
        mjcf_path: Union[str, Path] = DEFAULT_MJCF_PATH,
        control_hz: float = 200.0,
        sim_dt: float = 0.005,
        initial_height_m: float = LEROBOT_ENV_INIT_BASE_HEIGHT_M,
        use_lerobot_reference: bool = True,
        fixed_base: bool = False,
        fixed_base_height_m: float = 0.77,
        hardcode_mjlab_spawn: bool = True,
        hardcode_mjlab_spawn_with_qvel: bool = False,
        auto_reset_on_divergence: bool = False,
        auto_reset_on_flip: bool = True,
        flip_reset_angle_rad: float = 1.2217304763960306,
        reset_hold_s: float = 0.,
    ):
        if mujoco is None:
            raise RuntimeError(
                "mujoco python package is required for SimBipedalRobotController. "
                "Install it in your environment (e.g. `pip install mujoco`)."
            )

        self.mjcf_path = Path(mjcf_path)
        if not self.mjcf_path.exists():
            raise FileNotFoundError(f"MJCF file not found: {self.mjcf_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self.mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = float(sim_dt)
        # Keep local MuJoCo options aligned with RL_policy/config.yaml.
        self.model.opt.ccd_iterations = 50

        self.control_hz = float(control_hz)
        self._initial_height_m = float(initial_height_m)
        self._use_lerobot_reference = bool(use_lerobot_reference)
        self._has_free_base = bool(self.model.nq >= 7 and self.model.nv >= 6)
        self._fixed_base = bool(fixed_base and self._has_free_base)
        self._fixed_base_height_m = float(fixed_base_height_m)
        self._hardcode_mjlab_spawn = bool(hardcode_mjlab_spawn and self._has_free_base)
        self._hardcode_mjlab_spawn_with_qvel = bool(hardcode_mjlab_spawn_with_qvel)
        self._auto_reset_on_divergence = bool(auto_reset_on_divergence)
        self._auto_reset_on_flip = bool(auto_reset_on_flip)
        self._flip_reset_angle_rad = float(flip_reset_angle_rad)
        self._reset_hold_s = float(max(0.0, reset_hold_s))
        self._hold_actuation_until_s = 0.0
        self.mode = "state_only"
        self._running = False
        self._loop_thread: Optional[threading.Thread] = None
        self._sim_reset_count = 0
        self._sim_step_count = 0

        self.state: Dict[int, MotorState] = {mid: MotorState() for mid in MOTOR_IDS}
        self.action: Dict[int, MotorCommand] = {mid: MotorCommand() for mid in MOTOR_IDS}
        if use_lerobot_reference:
            self.gains = {mid: JointGains(*LEROBOT_SIM_GAINS_BY_MOTOR_ID[mid]) for mid in MOTOR_IDS}
        else:
            self.gains = {mid: JointGains(50.0, 2.0) for mid in MOTOR_IDS}
        self.joint_velocity_deg_s = np.zeros(12, dtype=float)
        self.joint_velocity_rad_s = np.zeros(12, dtype=float)
        self.command_limits_cal_deg: Dict[int, tuple[float, float]] = {
            mid: tuple(JOINT_LIMITS_DEG[mid]) for mid in MOTOR_IDS if mid in JOINT_LIMITS_DEG
        }

        self.motor_sign: Dict[int, float] = {mid: float(MOTOR_SIGN[mid]) for mid in MOTOR_IDS}
        self.motor_offset_deg: Dict[int, float] = {mid: float(MOTOR_OFFSET_DEG[mid]) for mid in MOTOR_IDS}

        self._state_lock = threading.Lock()
        self._action_lock = threading.Lock()
        self._imu_lock = threading.Lock()
        self._sim_lock = threading.Lock()
        self._viewer_thread: Optional[threading.Thread] = None
        self._viewer_running = False
        self._last_warn_s: Dict[str, float] = {}
        self._last_debug_s: Dict[str, float] = {}
        self._debug_action_logs = False
        self._debug_action_log_interval_s = 0.5
        self._trace_file: Optional[Any] = None
        self._trace_writer: Optional[Any] = None
        self._trace_every_n: int = 1
        self._trace_step_idx: int = 0

        self._imu_state: Optional[Dict[str, Any]] = None
        self._orientation_quaternion_xyzw: Optional[tuple[float, float, float, float]] = None
        self._orientation_stamp_s: float = 0.0

        self._joint_name_order = (
            "hipz_left",
            "hipx_left",
            "hipy_left",
            "knee_left",
            "ankley_left",
            "anklex_left",
            "hipz_right",
            "hipx_right",
            "hipy_right",
            "knee_right",
            "ankley_right",
            "anklex_right",
        )
        self._joint_id_order = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self._joint_name_order]
        if any(jid < 0 for jid in self._joint_id_order):
            missing = [n for n, jid in zip(self._joint_name_order, self._joint_id_order) if jid < 0]
            raise RuntimeError(f"Missing joints in MJCF: {missing}")
        self._joint_qpos_adr = [int(self.model.jnt_qposadr[jid]) for jid in self._joint_id_order]
        self._joint_dof_adr = [int(self.model.jnt_dofadr[jid]) for jid in self._joint_id_order]
        self._joint_order_motor_ids = tuple(MOTOR_IDS)
        self._actuator_index_by_motor_id = self._build_position_actuator_map()
        self._has_position_actuators = len(self._actuator_index_by_motor_id) == len(MOTOR_IDS)
        if not self._has_position_actuators:
            raise RuntimeError(
                f"Position actuator mapping incomplete: found {len(self._actuator_index_by_motor_id)}/{len(MOTOR_IDS)}. "
                "This controller requires MuJoCo position actuators for all 12 joints."
            )
        self._reference_joint_pos_rad = np.asarray(
            [
                LEROBOT_KNEES_BENT_REF_POSE_RAD["hipz_left"],
                LEROBOT_KNEES_BENT_REF_POSE_RAD["hipx_left"],
                LEROBOT_KNEES_BENT_REF_POSE_RAD["hipy_left"],
                LEROBOT_KNEES_BENT_REF_POSE_RAD["knee_left"],
                LEROBOT_KNEES_BENT_REF_POSE_RAD["ankley_left"],
                LEROBOT_KNEES_BENT_REF_POSE_RAD["anklex_left"],
                LEROBOT_KNEES_BENT_REF_POSE_RAD["hipz_right"],
                LEROBOT_KNEES_BENT_REF_POSE_RAD["hipx_right"],
                LEROBOT_KNEES_BENT_REF_POSE_RAD["hipy_right"],
                LEROBOT_KNEES_BENT_REF_POSE_RAD["knee_right"],
                LEROBOT_KNEES_BENT_REF_POSE_RAD["ankley_right"],
                LEROBOT_KNEES_BENT_REF_POSE_RAD["anklex_right"],
            ],
            dtype=float,
        )

        self._imu_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self._sensor_slices = self._build_sensor_slices()

        with self._sim_lock:
            self._reset_sim_state_locked()
            self._sync_state_from_sim(time.time())
            self._latch_action_from_state()

    def set_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unsupported mode '{mode}', expected one of {self.MODES}")
        self.mode = mode

    def set_joint_gains(self, motor_id: int, *, kp: Optional[float] = None, kd: Optional[float] = None) -> None:
        g = self.gains[int(motor_id)]
        self.gains[int(motor_id)] = JointGains(g.kp if kp is None else float(kp), g.kd if kd is None else float(kd))

    def set_joint_limit(self, motor_id: int, lo_deg: float, hi_deg: float) -> None:
        self.command_limits_cal_deg[int(motor_id)] = (float(lo_deg), float(hi_deg))

    def set_max_command_delta(self, delta_deg: float) -> None:
        _ = float(delta_deg)

    def enable_all(self) -> None:
        return None

    def disable_all(self) -> None:
        return None

    def clear_estop(self) -> None:
        return None

    def get_estop_reason(self) -> str:
        return ""

    def reset(self) -> None:
        with self._sim_lock:
            self._sim_reset_count += 1
            self._reset_sim_state_locked()
            self._sync_state_from_sim(time.time())
            self._latch_action_to_reference_pose()
            self._enter_post_reset_hold_locked()

    def get_reset_counter(self) -> int:
        return int(self._sim_reset_count)

    def get_reference_joint_pos_rad(self) -> np.ndarray:
        return np.asarray(self._reference_joint_pos_rad, dtype=float).copy()

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
        left = left or {}
        right = right or {}
        left_velocity_deg_s = left_velocity_deg_s or {}
        right_velocity_deg_s = right_velocity_deg_s or {}
        left_torque_nm = left_torque_nm or {}
        right_torque_nm = right_torque_nm or {}

        with self._action_lock:
            if self._is_post_reset_hold_active():
                self._debug(
                    "set_action_hold",
                    f"ignored action during post-reset hold "
                    f"(remaining={self._post_reset_hold_remaining_s():.3f}s)",
                )
                return {mid: float(self.action[mid].position_deg) for mid in MOTOR_IDS}
            raw = {mid: float(self.action[mid].position_deg) for mid in MOTOR_IDS}
            q_cmd_deg = self.motor_state_to_joint_state(raw, output_radians=False, nq=12)
            qd_cmd_deg_s = np.full(12, float(velocity_deg_s), dtype=float)
            tau_cmd_nm = np.full(12, float(torque_nm), dtype=float)
            self._apply_joint_side_updates(q_cmd_deg, left=left, right=right)
            self._apply_joint_side_updates(qd_cmd_deg_s, left=left_velocity_deg_s, right=right_velocity_deg_s)
            self._apply_joint_side_updates(tau_cmd_nm, left=left_torque_nm, right=right_torque_nm)
            pos_raw = self.joint_state_to_motor_state(q_cmd_deg, input_radians=False, output_space="raw")
            vel_raw, tau_raw = self._joint_vel_tau_to_motor_raw(qd_cmd_deg_s, tau_cmd_nm)

            for mid in MOTOR_IDS:
                prev = self.action[mid]
                self.action[mid] = MotorCommand(
                    position_deg=float(pos_raw[mid]),
                    velocity_deg_s=float(vel_raw[mid]),
                    torque_nm=float(tau_raw[mid]),
                    kp=prev.kp if kp is None else float(kp),
                    kd=prev.kd if kd is None else float(kd),
                )
            self._debug(
                "set_action",
                "received set_action "
                f"L(hipy={float(q_cmd_deg[2]):.2f}, knee={float(q_cmd_deg[3]):.2f}) "
                f"R(hipy={float(q_cmd_deg[8]):.2f}, knee={float(q_cmd_deg[9]):.2f})",
            )
            return {mid: float(self.action[mid].position_deg) for mid in MOTOR_IDS}

    def get_state_snapshot(self) -> Dict[int, MotorState]:
        with self._state_lock:
            return {mid: st for mid, st in self.state.items()}

    def get_imu_snapshot(self) -> Optional[Dict[str, Any]]:
        with self._imu_lock:
            return None if self._imu_state is None else dict(self._imu_state)

    def get_orientation_quaternion(self) -> Optional[tuple[float, float, float, float]]:
        with self._imu_lock:
            return None if self._orientation_quaternion_xyzw is None else tuple(self._orientation_quaternion_xyzw)

    def get_orientation_timestamp(self) -> float:
        with self._imu_lock:
            return float(self._orientation_stamp_s)

    def get_combined_state_snapshot(self, *, include_joint_state: bool = True) -> Dict[str, Any]:
        with self._state_lock:
            motor_snapshot = {mid: asdict(st) for mid, st in self.state.items()}
            qd_deg_s = np.asarray(self.joint_velocity_deg_s, dtype=float).copy()
            qd_rad_s = np.asarray(self.joint_velocity_rad_s, dtype=float).copy()
        out: Dict[str, Any] = {
            "time_s": time.time(),
            "mode": self.mode,
            "estop": False,
            "estop_reason": "",
            "sim_reset_count": int(self._sim_reset_count),
            "sim_step_count": int(self._sim_step_count),
            "sim_timestep_s": float(self.model.opt.timestep),
            "post_reset_hold_active": bool(self._is_post_reset_hold_active()),
            "post_reset_hold_remaining_s": float(self._post_reset_hold_remaining_s()),
            "auto_reset_on_divergence": bool(self._auto_reset_on_divergence),
            "auto_reset_on_flip": bool(self._auto_reset_on_flip),
            "flip_reset_angle_rad": float(self._flip_reset_angle_rad),
            "fixed_base": bool(self._fixed_base),
            "fixed_base_height_m": float(self._fixed_base_height_m),
            "has_position_actuators": bool(self._has_position_actuators),
            "default_joint_pos_rad": self.get_reference_joint_pos_rad().tolist(),
            "default_joint_pos_deg": np.rad2deg(self.get_reference_joint_pos_rad()).tolist(),
            "motors": motor_snapshot,
            "imu": self.get_imu_snapshot(),
            "orientation_quaternion_xyzw": self.get_orientation_quaternion(),
            "orientation_timestamp_s": self.get_orientation_timestamp(),
        }
        if include_joint_state:
            raw = {mid: float(motor_snapshot[mid]["position_deg"]) for mid in MOTOR_IDS}
            tau_raw = {mid: float(motor_snapshot[mid]["torque_nm"]) for mid in MOTOR_IDS}
            out["joint_state_rad"] = self.motor_state_to_joint_state(raw, output_radians=True, nq=12).tolist()
            out["joint_state_deg"] = self.motor_state_to_joint_state(raw, output_radians=False, nq=12).tolist()
            out["joint_torque_nm"] = self.motor_torque_to_joint_torque(tau_raw, nq=12).tolist()
            out["joint_velocity_deg_s"] = qd_deg_s.tolist()
            out["joint_velocity_rad_s"] = qd_rad_s.tolist()
        return out

    def request_state_once(self, motor_ids: Iterable[int] = MOTOR_IDS) -> list[int]:
        _ = list(motor_ids)
        with self._sim_lock:
            self._sync_state_from_sim(time.time())
        return []

    def set_fixed_base(self, enabled: bool, *, height_m: Optional[float] = None) -> None:
        with self._sim_lock:
            if height_m is not None:
                self._fixed_base_height_m = float(height_m)
            if enabled and not self._has_free_base:
                self._warn("fixed_base_unavailable", "model has no free base; cannot enable fixed_base")
                self._fixed_base = False
                return
            self._fixed_base = bool(enabled)
            if self._fixed_base:
                self._apply_fixed_base_lock_locked()
                mujoco.mj_forward(self.model, self.data)
                self._sync_state_from_sim(time.time())

    def set_auto_reset_on_flip(self, enabled: bool, *, angle_rad: Optional[float] = None) -> None:
        with self._sim_lock:
            self._auto_reset_on_flip = bool(enabled)
            if angle_rad is not None:
                self._flip_reset_angle_rad = float(max(0.0, angle_rad))

    def start(self, *, mode: str = "state_only", auto_enable: bool = False) -> None:
        _ = bool(auto_enable)
        self.set_mode(mode)
        if self._loop_thread is not None and self._loop_thread.is_alive():
            return
        self._running = True
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

    def stop(self, *, disable_motors: bool = True) -> None:
        _ = bool(disable_motors)
        self._running = False
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=2.0)
        self._loop_thread = None
        self.stop_viewer()
        self.disable_debug_trace()

    def enable_debug_trace(self, path: Union[str, Path], *, every_n: int = 1) -> None:
        self.disable_debug_trace()
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        f = out_path.open("w", newline="")
        w = csv.writer(f)
        header = [
            "time_s",
            "step",
            "mode",
            "post_reset_hold_active",
            "sim_reset_count",
        ]
        header += [f"cmd_joint_deg_{i}" for i in range(12)]
        header += [f"state_joint_deg_{i}" for i in range(12)]
        header += [f"state_joint_vel_deg_s_{i}" for i in range(12)]
        header += [f"ctrl_rad_{i}" for i in range(12)]
        w.writerow(header)
        f.flush()
        self._trace_file = f
        self._trace_writer = w
        self._trace_every_n = max(1, int(every_n))
        self._trace_step_idx = 0
        print(f"[SimRobot] debug trace enabled: {out_path}")

    def disable_debug_trace(self) -> None:
        if self._trace_file is not None:
            try:
                self._trace_file.flush()
                self._trace_file.close()
            except Exception:
                pass
        self._trace_file = None
        self._trace_writer = None

    def start_viewer(self) -> None:
        if mj_viewer is None:
            raise RuntimeError("mujoco.viewer is not available. Install/render-enabled MuJoCo package.")
        if self._viewer_thread is not None and self._viewer_thread.is_alive():
            return
        self._viewer_running = True
        self._viewer_thread = threading.Thread(target=self._viewer_loop, daemon=True)
        self._viewer_thread.start()

    def stop_viewer(self) -> None:
        self._viewer_running = False
        if self._viewer_thread is not None:
            self._viewer_thread.join(timeout=1.0)
        self._viewer_thread = None

    def motor_state_to_joint_state(
        self,
        motor_raw_deg: Dict[int, float],
        *,
        output_radians: bool = True,
        nq: Optional[int] = None,
    ) -> np.ndarray:
        out_nq = 12 if nq is None else int(nq)
        q_deg = np.zeros(max(12, out_nq), dtype=float)

        l_hipz = self._raw_to_cal(1, motor_raw_deg[1])
        l_hipx = self._raw_to_cal(2, motor_raw_deg[2])
        l_hipy = self._raw_to_cal(3, motor_raw_deg[3])
        l_knee = self._raw_to_cal(4, motor_raw_deg[4])
        l_a1 = self._raw_to_cal(5, motor_raw_deg[5])
        l_a2 = self._raw_to_cal(6, motor_raw_deg[6])
        l_pitch, l_roll = self._ankle_pitch_roll_from_cal_values(l_a1, l_a2, side_name="left")
        q_deg[0:6] = [l_hipz, l_hipx, l_hipy, l_knee, l_pitch, l_roll]

        r_hipz = self._raw_to_cal(7, motor_raw_deg[7])
        r_hipx = self._raw_to_cal(8, motor_raw_deg[8])
        r_hipy = self._raw_to_cal(9, motor_raw_deg[9])
        r_knee = self._raw_to_cal(10, motor_raw_deg[10])
        r_a1 = self._raw_to_cal(11, motor_raw_deg[11])
        r_a2 = self._raw_to_cal(12, motor_raw_deg[12])
        r_pitch, r_roll = self._ankle_pitch_roll_from_cal_values(r_a1, r_a2, side_name="right")
        q_deg[6:12] = [r_hipz, r_hipx, r_hipy, r_knee, r_pitch, r_roll]

        return np.deg2rad(q_deg[:out_nq]) if output_radians else q_deg[:out_nq]

    def joint_state_to_motor_state(
        self,
        q_joint: Any,
        *,
        input_radians: bool = True,
        output_space: str = "calibrated",
    ) -> Dict[int, float]:
        if output_space not in ("calibrated", "raw"):
            raise ValueError("output_space must be 'calibrated' or 'raw'")
        q = np.asarray(q_joint, dtype=float).reshape(-1)
        q_deg = np.rad2deg(q) if input_radians else q
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
        out_cal[5], out_cal[6], out_cal[11], out_cal[12] = float(l_a1), float(l_a2), float(r_a1), float(r_a2)
        if output_space == "calibrated":
            return out_cal
        return {mid: self._cal_to_raw(mid, val) for mid, val in out_cal.items()}

    def _run_loop(self) -> None:
        period = 1.0 / max(1.0, float(self.control_hz))
        steps_per_tick = max(1, int(round(period / float(self.model.opt.timestep))))
        next_tick = time.perf_counter()
        while self._running:
            try:
                with self._sim_lock:
                    if self.mode == "control":
                        self._apply_position_actuator_ctrl()
                    else:
                        self._debug("mode_state_only", "mode=state_only, actuator control is disabled")
                        self.data.qfrc_applied[:] = 0.0
                        if self.data.ctrl.size:
                            self.data.ctrl[:] = 0.0
                    for _ in range(steps_per_tick):
                        mujoco.mj_step(self.model, self.data)
                        self._sim_step_count += 1
                    if self._fixed_base:
                        self._apply_fixed_base_lock_locked()
                        mujoco.mj_forward(self.model, self.data)
                    if self._sim_has_nonfinite_locked():
                        self._sim_reset_count += 1
                        self._warn("sim_nonfinite_reset", "non-finite state detected; resetting simulation")
                        self._reset_sim_state_locked()
                        self._sync_state_from_sim(time.time())
                        self._latch_action_to_reference_pose()
                        self._enter_post_reset_hold_locked()
                    elif self._sim_diverged_locked() and self._auto_reset_on_divergence:
                        self._sim_reset_count += 1
                        self._warn("sim_diverged_reset", "divergence threshold exceeded; resetting simulation")
                        self._reset_sim_state_locked()
                        self._sync_state_from_sim(time.time())
                        self._latch_action_to_reference_pose()
                        self._enter_post_reset_hold_locked()
                    elif self._sim_flipped_locked() and self._auto_reset_on_flip:
                        self._sim_reset_count += 1
                        self._warn("sim_flip_reset", "robot flipped/fell (mjlab-like termination); resetting simulation")
                        self._reset_sim_state_locked()
                        self._sync_state_from_sim(time.time())
                        self._latch_action_to_reference_pose()
                        self._enter_post_reset_hold_locked()
                    self._sync_state_from_sim(time.time())
                    self._maybe_log_trace_locked()
            except Exception as exc:
                self._warn("sim_loop_exception", f"{type(exc).__name__}: {exc}; resetting simulation")
                with self._sim_lock:
                    self._sim_reset_count += 1
                    self._reset_sim_state_locked()
                    self._sync_state_from_sim(time.time())
                    self._latch_action_to_reference_pose()
                    self._enter_post_reset_hold_locked()
            next_tick += period
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.perf_counter()

    def _viewer_loop(self) -> None:
        if mj_viewer is None:
            return
        try:
            # Keep viewer data isolated from control-loop data to avoid stack/visual copy races.
            view_model = mujoco.MjModel.from_xml_path(str(self.mjcf_path))
            view_data = mujoco.MjData(view_model)
            with mj_viewer.launch_passive(view_model, view_data) as viewer:
                while self._viewer_running and viewer.is_running():
                    with self._sim_lock:
                        qpos = np.asarray(self.data.qpos, dtype=float).copy()
                        qvel = np.asarray(self.data.qvel, dtype=float).copy()
                    view_data.qpos[:] = qpos
                    view_data.qvel[:] = qvel
                    mujoco.mj_forward(view_model, view_data)
                    viewer.sync()
                    time.sleep(1.0 / 60.0)
        except Exception as exc:
            self._warn("viewer_exception", f"{type(exc).__name__}: {exc}")
        finally:
            self._viewer_running = False

    def _apply_joint_pd_to_qfrc(self) -> None:
        with self._action_lock:
            cmd = {mid: self.action[mid] for mid in MOTOR_IDS}

        q_cmd_raw = {mid: float(cmd[mid].position_deg) for mid in MOTOR_IDS}
        qd_cmd_raw = {mid: float(cmd[mid].velocity_deg_s) for mid in MOTOR_IDS}
        tau_cmd_raw = {mid: float(cmd[mid].torque_nm) for mid in MOTOR_IDS}

        q_cmd_deg = self.motor_state_to_joint_state(q_cmd_raw, output_radians=False, nq=12)
        qd_cmd_deg_s = self.motor_velocity_to_joint_velocity(qd_cmd_raw, output_radians=False, nq=12)
        tau_cmd_joint = self._motor_tau_raw_to_joint_tau(tau_cmd_raw)

        q_cur = self._read_joint_q_deg()
        qd_cur = self._read_joint_qd_deg_s()
        self.data.qfrc_applied[:] = 0.0
        if self.data.ctrl.size:
            self.data.ctrl[:] = 0.0
        for j in range(12):
            dof_adr = self._joint_dof_adr[j]
            if j < 4:
                mid = j + 1
            elif j < 6:
                mid = 5 if j == 4 else 6
            elif j < 10:
                mid = j + 1
            else:
                mid = 11 if j == 10 else 12
            gains = self.gains[mid]
            kp = float(gains.kp if cmd[mid].kp is None else cmd[mid].kp)
            kd = float(gains.kd if cmd[mid].kd is None else cmd[mid].kd)
            tau = (
                kp * np.deg2rad(float(q_cmd_deg[j] - q_cur[j]))
                + kd * np.deg2rad(float(qd_cmd_deg_s[j] - qd_cur[j]))
                + float(tau_cmd_joint[j])
            )
            self.data.qfrc_applied[dof_adr] = float(tau)

    def _apply_position_actuator_ctrl(self) -> None:
        with self._action_lock:
            cmd = {mid: self.action[mid] for mid in MOTOR_IDS}
        q_cmd_raw = {mid: float(cmd[mid].position_deg) for mid in MOTOR_IDS}
        q_cmd_deg = self.motor_state_to_joint_state(q_cmd_raw, output_radians=False, nq=12)
        q_cmd_rad = np.deg2rad(np.asarray(q_cmd_deg, dtype=float))

        self.data.qfrc_applied[:] = 0.0
        if self.data.ctrl.size:
            self.data.ctrl[:] = 0.0
        for mid, aid in self._actuator_index_by_motor_id.items():
            idx = int(mid) - 1
            if idx < 0 or idx >= q_cmd_rad.size:
                continue
            u = float(q_cmd_rad[idx])
            if int(self.model.actuator_ctrllimited[aid]) != 0:
                lo = float(self.model.actuator_ctrlrange[aid, 0])
                hi = float(self.model.actuator_ctrlrange[aid, 1])
                u = float(np.clip(u, lo, hi))
            self.data.ctrl[aid] = u
        ctrl = np.asarray(self.data.ctrl, dtype=float).reshape(-1)
        q_cur_deg = self._read_joint_q_deg()
        err = q_cmd_deg - q_cur_deg
        self._debug(
            "actuator_ctrl",
            f"ctrl |mean|={float(np.mean(np.abs(ctrl))):.4f} "
            f"|err_deg|mean={float(np.mean(np.abs(err))):.2f} max={float(np.max(np.abs(err))):.2f}",
        )

    def _reset_sim_state_locked(self) -> None:
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        if self.data.act.size:
            self.data.act[:] = 0.0
        self.data.qacc_warmstart[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0

        if self._has_free_base:
            if self._hardcode_mjlab_spawn and not self._fixed_base:
                # Debug mode: match one known MJLab spawn exactly.
                self.data.qpos[0:7] = MJLAB_HARDCODED_SPAWN_QPOS_FREE
                if self._hardcode_mjlab_spawn_with_qvel:
                    self.data.qvel[0:6] = MJLAB_HARDCODED_SPAWN_QVEL_FREE
            else:
                self.data.qpos[0] = 0.0
                self.data.qpos[1] = 0.0
                self.data.qpos[2] = float(self._fixed_base_height_m if self._fixed_base else self._initial_height_m)
                # Free-joint quaternion in MuJoCo qpos is [w, x, y, z].
                self.data.qpos[3] = 1.0
                self.data.qpos[4] = 0.0
                self.data.qpos[5] = 0.0
                self.data.qpos[6] = 0.0
        if self._use_lerobot_reference:
            for jn, val in LEROBOT_KNEES_BENT_REF_POSE_RAD.items():
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid >= 0:
                    qadr = int(self.model.jnt_qposadr[jid])
                    self.data.qpos[qadr] = float(val)
        if self._fixed_base:
            self._apply_fixed_base_lock_locked()
        mujoco.mj_forward(self.model, self.data)

    def _apply_fixed_base_lock_locked(self) -> None:
        if not self._has_free_base:
            return
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = float(self._fixed_base_height_m)
        self.data.qpos[3] = 1.0
        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0
        self.data.qpos[6] = 0.0
        self.data.qvel[0:6] = 0.0

    def _sim_diverged_locked(self) -> bool:
        if float(np.linalg.norm(self.data.qvel)) > 200.0:
            return True
        if float(np.max(np.abs(self.data.qpos[7:]))) > 5.0:
            return True
        if self.data.qpos.size >= 3 and float(self.data.qpos[2]) < 0.05:
            return True
        return False

    def _sim_has_nonfinite_locked(self) -> bool:
        return (not np.all(np.isfinite(self.data.qpos))) or (not np.all(np.isfinite(self.data.qvel)))

    def _sim_flipped_locked(self) -> bool:
        if not self._has_free_base or self.data.qpos.size < 7:
            return False
        q = np.asarray(self.data.qpos[3:7], dtype=float).reshape(-1)
        n = float(np.linalg.norm(q))
        if n < 1e-9:
            return True
        q /= n
        w, x, y, z = [float(v) for v in q]
        # z-component of base local +Z axis in world frame.
        up_z = 1.0 - 2.0 * (x * x + y * y)
        up_z = float(np.clip(up_z, -1.0, 1.0))
        tilt = float(np.arccos(up_z))
        return tilt > float(self._flip_reset_angle_rad)

    def _is_post_reset_hold_active(self, now_s: Optional[float] = None) -> bool:
        t = time.time() if now_s is None else float(now_s)
        return t < float(self._hold_actuation_until_s)

    def _post_reset_hold_remaining_s(self, now_s: Optional[float] = None) -> float:
        t = time.time() if now_s is None else float(now_s)
        return float(max(0.0, float(self._hold_actuation_until_s) - t))

    def _enter_post_reset_hold_locked(self) -> None:
        self._hold_actuation_until_s = float(time.time() + self._reset_hold_s)

    def _warn(self, key: str, msg: str, interval_s: float = 1.0) -> None:
        now = time.time()
        last = float(self._last_warn_s.get(key, 0.0))
        if (now - last) < float(interval_s):
            return
        self._last_warn_s[key] = now
        print(f"[SimRobot][WARN][{key}] {msg}")

    def _debug(self, key: str, msg: str) -> None:
        if not self._debug_action_logs:
            return
        now = time.time()
        last = float(self._last_debug_s.get(key, 0.0))
        if (now - last) < float(self._debug_action_log_interval_s):
            return
        self._last_debug_s[key] = now
        print(f"[SimRobot][DEBUG][{key}] {msg}")

    def _maybe_log_trace_locked(self) -> None:
        if self._trace_writer is None or self._trace_file is None:
            return
        self._trace_step_idx += 1
        if (self._trace_step_idx % int(self._trace_every_n)) != 0:
            return
        with self._action_lock:
            q_cmd_raw = {mid: float(self.action[mid].position_deg) for mid in MOTOR_IDS}
        q_cmd_deg = self.motor_state_to_joint_state(q_cmd_raw, output_radians=False, nq=12)
        q_deg = self._read_joint_q_deg()
        qd_deg = self._read_joint_qd_deg_s()
        ctrl_rad = np.zeros(12, dtype=float)
        for i, mid in enumerate(MOTOR_IDS):
            aid = self._actuator_index_by_motor_id.get(int(mid))
            if aid is not None and aid < self.data.ctrl.size:
                ctrl_rad[i] = float(self.data.ctrl[aid])

        row = [
            f"{time.time():.6f}",
            str(self._trace_step_idx),
            str(self.mode),
            int(self._is_post_reset_hold_active()),
            int(self._sim_reset_count),
        ]
        row += [float(v) for v in np.asarray(q_cmd_deg, dtype=float).reshape(-1).tolist()]
        row += [float(v) for v in np.asarray(q_deg, dtype=float).reshape(-1).tolist()]
        row += [float(v) for v in np.asarray(qd_deg, dtype=float).reshape(-1).tolist()]
        row += [float(v) for v in np.asarray(ctrl_rad, dtype=float).reshape(-1).tolist()]
        self._trace_writer.writerow(row)
        self._trace_file.flush()

    def _sync_state_from_sim(self, stamp_s: float) -> None:
        q_deg = self._read_joint_q_deg()
        qd_deg_s = self._read_joint_qd_deg_s()
        motor_raw = self.joint_state_to_motor_state(q_deg, input_radians=False, output_space="raw")
        motor_qd_raw = self._joint_vel_deg_to_motor_raw(qd_deg_s)

        with self._state_lock:
            for mid in MOTOR_IDS:
                self.state[mid] = MotorState(
                    position_deg=float(motor_raw[mid]),
                    velocity_deg_s=float(motor_qd_raw[mid]),
                    torque_nm=0.0,
                    temp_mos_c=35.0,
                    stamp=float(stamp_s),
                )
            self.joint_velocity_deg_s = np.asarray(qd_deg_s, dtype=float).copy()
            self.joint_velocity_rad_s = np.deg2rad(self.joint_velocity_deg_s)

        self._warn_joint_state_out_of_bounds(q_deg)
        self._update_imu_state(stamp_s)

    def _warn_joint_state_out_of_bounds(self, q_deg: np.ndarray) -> None:
        # Limits in JOINT_LIMITS_DEG are defined in raw motor space.
        # Reconstruct raw motor positions from current joint-space state
        # (same conversion path as controller commands), then check limits.
        q = np.asarray(q_deg, dtype=float).reshape(-1)
        if q.size < 12:
            return
        motor_raw = self.joint_state_to_motor_state(q, input_radians=False, output_space="raw")
        for idx, mid in enumerate(MOTOR_IDS):
            lim = JOINT_LIMITS_DEG.get(int(mid))
            if lim is None:
                continue
            lo = float(min(lim))
            hi = float(max(lim))
            val = float(motor_raw.get(int(mid), 0.0))
            if lo <= val <= hi:
                continue
            if val < lo:
                dist = lo - val
                msg = (
                    f"motor_id={int(mid)} raw={val:.2f} deg below limit {lo:.2f} deg "
                    f"(delta={dist:.2f}); joint_state={float(q[idx]):.2f} deg"
                )
            else:
                dist = val - hi
                msg = (
                    f"motor_id={int(mid)} raw={val:.2f} deg above limit {hi:.2f} deg "
                    f"(delta={dist:.2f}); joint_state={float(q[idx]):.2f} deg"
                )
            self._warn(f"joint_state_oob_m{int(mid)}", msg, interval_s=0.5)

    def _build_sensor_slices(self) -> Dict[str, slice]:
        out: Dict[str, slice] = {}
        for sname in ("imu_ang_vel", "imu_lin_vel", "imu_lin_acc"):
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sname)
            if sid < 0:
                continue
            adr = int(self.model.sensor_adr[sid])
            dim = int(self.model.sensor_dim[sid])
            out[sname] = slice(adr, adr + dim)
        return out

    def _build_position_actuator_map(self) -> Dict[int, int]:
        out: Dict[int, int] = {}
        joint_to_motor = {jn: mid for jn, mid in zip(self._joint_name_order, self._joint_order_motor_ids)}
        for aid in range(int(self.model.nu)):
            jid = int(self.model.actuator_trnid[aid, 0])
            if jid < 0:
                continue
            jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if not jname:
                continue
            mid = joint_to_motor.get(jname)
            if mid is None:
                continue
            out[int(mid)] = int(aid)
        return out

    def _update_imu_state(self, stamp_s: float) -> None:
        quat_xyzw = self._imu_quaternion_xyzw()
        gyro = self._read_sensor3("imu_ang_vel")
        lin_vel = self._read_sensor3("imu_lin_vel")
        lin_acc = self._read_sensor3("imu_lin_acc")
        imu_state = {
            "timestamp_s": float(stamp_s),
            "quaternion_xyzw": list(quat_xyzw),
            "gyro_rads": list(gyro),
            "ang_vel_rad_s": list(gyro),
            "linear_velocity_mps": list(lin_vel),
            "lin_vel_m_s": list(lin_vel),
            "linear_acceleration_mps2": list(lin_acc),
            "sensor": "mujoco_sim",
            "available": True,
        }
        with self._imu_lock:
            self._imu_state = imu_state
            self._orientation_quaternion_xyzw = quat_xyzw
            self._orientation_stamp_s = float(stamp_s)

    def _read_sensor3(self, name: str) -> tuple[float, float, float]:
        sl = self._sensor_slices.get(name)
        if sl is None:
            return (0.0, 0.0, 0.0)
        vals = np.asarray(self.data.sensordata[sl], dtype=float).reshape(-1)
        if vals.size < 3:
            return (0.0, 0.0, 0.0)
        return (float(vals[0]), float(vals[1]), float(vals[2]))

    def _imu_quaternion_xyzw(self) -> tuple[float, float, float, float]:
        if self._imu_site_id < 0:
            return (0.0, 0.0, 0.0, 1.0)
        mat = np.asarray(self.data.site_xmat[self._imu_site_id], dtype=float).reshape(3, 3)
        qw = np.sqrt(max(0.0, 1.0 + mat[0, 0] + mat[1, 1] + mat[2, 2])) / 2.0
        qx = np.sqrt(max(0.0, 1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2])) / 2.0
        qy = np.sqrt(max(0.0, 1.0 - mat[0, 0] + mat[1, 1] - mat[2, 2])) / 2.0
        qz = np.sqrt(max(0.0, 1.0 - mat[0, 0] - mat[1, 1] + mat[2, 2])) / 2.0
        qx = np.copysign(qx, mat[2, 1] - mat[1, 2])
        qy = np.copysign(qy, mat[0, 2] - mat[2, 0])
        qz = np.copysign(qz, mat[1, 0] - mat[0, 1])
        q = np.asarray([qx, qy, qz, qw], dtype=float)
        n = float(np.linalg.norm(q))
        if n < 1e-9:
            return (0.0, 0.0, 0.0, 1.0)
        q /= n
        return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    def _read_joint_q_deg(self) -> np.ndarray:
        return np.rad2deg(np.asarray([self.data.qpos[adr] for adr in self._joint_qpos_adr], dtype=float))

    def _read_joint_qd_deg_s(self) -> np.ndarray:
        return np.rad2deg(np.asarray([self.data.qvel[adr] for adr in self._joint_dof_adr], dtype=float))

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
        left = left or {}
        right = right or {}
        idx_map = self._joint_index_map()
        for side_name, side in (("left", left), ("right", right)):
            for key, val in side.items():
                idx = idx_map[side_name].get(key)
                if idx is not None:
                    values[idx] = float(val)

    def _joint_vel_tau_to_motor_raw(
        self,
        qd_deg_s: np.ndarray,
        tau_nm: np.ndarray,
    ) -> tuple[Dict[int, float], Dict[int, float]]:
        qd_raw: Dict[int, float] = {}
        tau_raw: Dict[int, float] = {}

        direct = {0: 1, 1: 2, 2: 3, 3: 4, 6: 7, 7: 8, 8: 9, 9: 10}
        for qi, mid in direct.items():
            s = float(self.motor_sign[mid])
            qd_raw[mid] = float(qd_deg_s[qi] / s)
            tau_raw[mid] = float(tau_nm[qi] * s)

        sp_l = float(ANKLE_COUPLING_CALIBRATION_LEFT["pitch"]["sign"])
        sr_l = float(ANKLE_COUPLING_CALIBRATION_LEFT["roll"]["sign"])
        s5 = float(self.motor_sign[5])
        s6 = float(self.motor_sign[6])
        u_l = float(qd_deg_s[4]) / sp_l
        v_l = float(qd_deg_s[5]) / sr_l
        qd_raw[5] = float((u_l + v_l) / s5)
        qd_raw[6] = float((v_l - u_l) / s6)
        tau_raw[5] = float(s5 * (0.5 * sp_l * float(tau_nm[4]) + 0.5 * sr_l * float(tau_nm[5])))
        tau_raw[6] = float(s6 * (-0.5 * sp_l * float(tau_nm[4]) + 0.5 * sr_l * float(tau_nm[5])))

        sp_r = float(ANKLE_COUPLING_CALIBRATION_RIGHT["pitch"]["sign"])
        sr_r = float(ANKLE_COUPLING_CALIBRATION_RIGHT["roll"]["sign"])
        s11 = float(self.motor_sign[11])
        s12 = float(self.motor_sign[12])
        u_r = float(qd_deg_s[10]) / sp_r
        v_r = float(qd_deg_s[11]) / sr_r
        qd_raw[11] = float((u_r + v_r) / s11)
        qd_raw[12] = float((v_r - u_r) / s12)
        tau_raw[11] = float(s11 * (0.5 * sp_r * float(tau_nm[10]) + 0.5 * sr_r * float(tau_nm[11])))
        tau_raw[12] = float(s12 * (-0.5 * sp_r * float(tau_nm[10]) + 0.5 * sr_r * float(tau_nm[11])))
        return qd_raw, tau_raw

    def _joint_vel_deg_to_motor_raw(self, qd_deg_s: np.ndarray) -> Dict[int, float]:
        qd = np.asarray(qd_deg_s, dtype=float).reshape(-1)
        out: Dict[int, float] = {}
        out[1] = float(qd[0] / self.motor_sign[1])
        out[2] = float(qd[1] / self.motor_sign[2])
        out[3] = float(qd[2] / self.motor_sign[3])
        out[4] = float(qd[3] / self.motor_sign[4])
        out[7] = float(qd[6] / self.motor_sign[7])
        out[8] = float(qd[7] / self.motor_sign[8])
        out[9] = float(qd[8] / self.motor_sign[9])
        out[10] = float(qd[9] / self.motor_sign[10])

        sp_l = float(ANKLE_COUPLING_CALIBRATION_LEFT["pitch"]["sign"])
        sr_l = float(ANKLE_COUPLING_CALIBRATION_LEFT["roll"]["sign"])
        u_l = float(qd[4]) / sp_l
        v_l = float(qd[5]) / sr_l
        l_a1_cal = u_l + v_l
        l_a2_cal = v_l - u_l
        out[5] = float(l_a1_cal / self.motor_sign[5])
        out[6] = float(l_a2_cal / self.motor_sign[6])

        sp_r = float(ANKLE_COUPLING_CALIBRATION_RIGHT["pitch"]["sign"])
        sr_r = float(ANKLE_COUPLING_CALIBRATION_RIGHT["roll"]["sign"])
        u_r = float(qd[10]) / sp_r
        v_r = float(qd[11]) / sr_r
        r_a1_cal = u_r + v_r
        r_a2_cal = v_r - u_r
        out[11] = float(r_a1_cal / self.motor_sign[11])
        out[12] = float(r_a2_cal / self.motor_sign[12])
        return out

    def motor_velocity_to_joint_velocity(
        self,
        motor_raw_vel_deg_s: Dict[int, float],
        *,
        output_radians: bool = True,
        nq: Optional[int] = None,
    ) -> np.ndarray:
        out_nq = 12 if nq is None else int(nq)
        qd_deg = np.zeros(max(12, out_nq), dtype=float)
        qd_deg[0] = float(self.motor_sign[1] * motor_raw_vel_deg_s[1])
        qd_deg[1] = float(self.motor_sign[2] * motor_raw_vel_deg_s[2])
        qd_deg[2] = float(self.motor_sign[3] * motor_raw_vel_deg_s[3])
        qd_deg[3] = float(self.motor_sign[4] * motor_raw_vel_deg_s[4])
        qd_deg[6] = float(self.motor_sign[7] * motor_raw_vel_deg_s[7])
        qd_deg[7] = float(self.motor_sign[8] * motor_raw_vel_deg_s[8])
        qd_deg[8] = float(self.motor_sign[9] * motor_raw_vel_deg_s[9])
        qd_deg[9] = float(self.motor_sign[10] * motor_raw_vel_deg_s[10])
        l_a1 = float(self.motor_sign[5] * motor_raw_vel_deg_s[5])
        l_a2 = float(self.motor_sign[6] * motor_raw_vel_deg_s[6])
        qd_deg[4], qd_deg[5] = self._ankle_pitch_roll_vel_from_cal_values(l_a1, l_a2, side_name="left")
        r_a1 = float(self.motor_sign[11] * motor_raw_vel_deg_s[11])
        r_a2 = float(self.motor_sign[12] * motor_raw_vel_deg_s[12])
        qd_deg[10], qd_deg[11] = self._ankle_pitch_roll_vel_from_cal_values(r_a1, r_a2, side_name="right")
        return np.deg2rad(qd_deg[:out_nq]) if output_radians else qd_deg[:out_nq]

    def _motor_tau_raw_to_joint_tau(self, tau_raw: Dict[int, float]) -> np.ndarray:
        out = np.zeros(12, dtype=float)
        out[0] = float(tau_raw[1] / self.motor_sign[1])
        out[1] = float(tau_raw[2] / self.motor_sign[2])
        out[2] = float(tau_raw[3] / self.motor_sign[3])
        out[3] = float(tau_raw[4] / self.motor_sign[4])
        out[6] = float(tau_raw[7] / self.motor_sign[7])
        out[7] = float(tau_raw[8] / self.motor_sign[8])
        out[8] = float(tau_raw[9] / self.motor_sign[9])
        out[9] = float(tau_raw[10] / self.motor_sign[10])

        s5 = float(self.motor_sign[5])
        s6 = float(self.motor_sign[6])
        s11 = float(self.motor_sign[11])
        s12 = float(self.motor_sign[12])
        t5_cal = float(tau_raw[5] / s5)
        t6_cal = float(tau_raw[6] / s6)
        t11_cal = float(tau_raw[11] / s11)
        t12_cal = float(tau_raw[12] / s12)

        sp_l = float(ANKLE_COUPLING_CALIBRATION_LEFT["pitch"]["sign"])
        sr_l = float(ANKLE_COUPLING_CALIBRATION_LEFT["roll"]["sign"])
        out[4] = float((t5_cal - t6_cal) / sp_l)
        out[5] = float((t5_cal + t6_cal) / sr_l)

        sp_r = float(ANKLE_COUPLING_CALIBRATION_RIGHT["pitch"]["sign"])
        sr_r = float(ANKLE_COUPLING_CALIBRATION_RIGHT["roll"]["sign"])
        out[10] = float((t11_cal - t12_cal) / sp_r)
        out[11] = float((t11_cal + t12_cal) / sr_r)
        return out

    def motor_torque_to_joint_torque(
        self,
        motor_raw_tau_nm: Dict[int, float],
        *,
        nq: Optional[int] = None,
    ) -> np.ndarray:
        out_nq = int(self._model_nq if nq is None else nq)
        return self._motor_tau_raw_to_joint_tau(motor_raw_tau_nm)[:out_nq]

    def _ankle_pitch_roll_from_cal_values(self, a1: float, a2: float, *, side_name: str) -> tuple[float, float]:
        cfg = ANKLE_COUPLING_CALIBRATION_LEFT if side_name == "left" else ANKLE_COUPLING_CALIBRATION_RIGHT
        sign_p = float(cfg["pitch"]["sign"])
        off_p = float(cfg["pitch"]["offset_deg"])
        sign_r = float(cfg["roll"]["sign"])
        off_r = float(cfg["roll"]["offset_deg"])
        pitch = sign_p * ((float(a1) - float(a2)) / 2.0) + off_p
        roll = sign_r * ((float(a1) + float(a2)) / 2.0) + off_r
        return pitch, roll

    def _ankle_pitch_roll_vel_from_cal_values(self, a1_vel: float, a2_vel: float, *, side_name: str) -> tuple[float, float]:
        cfg = ANKLE_COUPLING_CALIBRATION_LEFT if side_name == "left" else ANKLE_COUPLING_CALIBRATION_RIGHT
        sign_p = float(cfg["pitch"]["sign"])
        sign_r = float(cfg["roll"]["sign"])
        pitch_vel = sign_p * ((float(a1_vel) - float(a2_vel)) / 2.0)
        roll_vel = sign_r * ((float(a1_vel) + float(a2_vel)) / 2.0)
        return pitch_vel, roll_vel

    def _ankle_motors_from_pitch_roll(self, pitch: float, roll: float, *, side_name: str) -> tuple[float, float]:
        cfg = ANKLE_COUPLING_CALIBRATION_LEFT if side_name == "left" else ANKLE_COUPLING_CALIBRATION_RIGHT
        sign_p = float(cfg["pitch"]["sign"])
        off_p = float(cfg["pitch"]["offset_deg"])
        sign_r = float(cfg["roll"]["sign"])
        off_r = float(cfg["roll"]["offset_deg"])
        p = (float(pitch) - off_p) / sign_p
        r = (float(roll) - off_r) / sign_r
        return (p + r), (r - p)

    def _raw_to_cal(self, motor_id: int, raw_deg: float) -> float:
        return float(self.motor_sign[int(motor_id)] * float(raw_deg) + self.motor_offset_deg[int(motor_id)])

    def _cal_to_raw(self, motor_id: int, cal_deg: float) -> float:
        sign = float(self.motor_sign[int(motor_id)])
        if abs(sign) < 1e-9:
            raise ValueError(f"Invalid motor sign for motor {motor_id}: {sign}")
        return float((float(cal_deg) - self.motor_offset_deg[int(motor_id)]) / sign)

    def _latch_action_from_state(self) -> None:
        with self._state_lock, self._action_lock:
            for mid in MOTOR_IDS:
                st = self.state[mid]
                prev = self.action[mid]
                self.action[mid] = MotorCommand(
                    position_deg=float(st.position_deg),
                    velocity_deg_s=0.0,
                    torque_nm=0.0,
                    kp=prev.kp,
                    kd=prev.kd,
                )

    def _latch_action_to_reference_pose(self) -> None:
        q_ref_rad = np.asarray(self._reference_joint_pos_rad, dtype=float).reshape(-1)
        pos_raw = self.joint_state_to_motor_state(q_ref_rad, input_radians=True, output_space="raw")
        with self._action_lock:
            for mid in MOTOR_IDS:
                prev = self.action[mid]
                self.action[mid] = MotorCommand(
                    position_deg=float(pos_raw[mid]),
                    velocity_deg_s=0.0,
                    torque_nm=0.0,
                    kp=prev.kp,
                    kd=prev.kd,
                )
