from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import argparse
import importlib
import json
import pickle
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque

import numpy as np

from bipedal_robot import BipedalRobotController

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional
    yaml = None


JOINT_ORDER_12 = (
    ("left", "hipz"),
    ("left", "hipx"),
    ("left", "hipy"),
    ("left", "knee"),
    ("left", "ankle_pitch"),
    ("left", "ankle_roll"),
    ("right", "hipz"),
    ("right", "hipx"),
    ("right", "hipy"),
    ("right", "knee"),
    ("right", "ankle_pitch"),
    ("right", "ankle_roll"),
)

JOINT_KEY_TO_MOTOR_ID = {
    "left.hipz": 1,
    "left.hipx": 2,
    "left.hipy": 3,
    "left.knee": 4,
    "left.ankle_pitch": 5,
    "left.ankle_roll": 6,
    "right.hipz": 7,
    "right.hipx": 8,
    "right.hipy": 9,
    "right.knee": 10,
    "right.ankle_pitch": 11,
    "right.ankle_roll": 12,
}


@dataclass
class AgentSpec:
    obs_keys: List[str]
    action_keys: List[str]
    history_len: int = 1
    inference_hz: float = 50.0
    action_mode: str = "absolute"  # absolute or delta
    action_scale: float = 1.0
    policy_terms: List[str] = field(default_factory=list)
    action_scales_rad: List[float] = field(default_factory=list)


def _load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text()
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML is required to read YAML config files.")
        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    if suffix == ".json":
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    raise ValueError(f"Unsupported config format: {path}")


def _flatten_strings(value: Any) -> List[str]:
    out: List[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, (list, tuple)):
        for v in value:
            out.extend(_flatten_strings(v))
    elif isinstance(value, dict):
        for v in value.values():
            out.extend(_flatten_strings(v))
    return out


def _get_first(cfg: Dict[str, Any], keys: Sequence[str], default: Any) -> Any:
    for key in keys:
        cur: Any = cfg
        ok = True
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                ok = False
                break
            cur = cur[part]
        if ok:
            return cur
    return default


def _build_imu_from_config(cfg: Dict[str, Any]) -> Optional[Any]:
    imu_cfg = _get_first(
        cfg,
        keys=(
            "real_robot.imu",
            "robot.imu",
            "imu",
            "hardware.imu",
        ),
        default=None,
    )
    if not isinstance(imu_cfg, dict):
        return None
    enabled = bool(imu_cfg.get("enabled", True))
    if not enabled:
        return None

    from IMU_integration import IMU

    sensor = str(imu_cfg.get("sensor", imu_cfg.get("type", "bno085"))).strip().lower()
    mock = bool(imu_cfg.get("mock", False))
    kwargs: Dict[str, Any] = {}
    if sensor == "bno085":
        kwargs["address"] = int(imu_cfg.get("address", 0x4A))
        reports = imu_cfg.get("reports")
        if isinstance(reports, (list, tuple)):
            kwargs["reports"] = tuple(str(r) for r in reports)
    elif sensor == "jy901":
        kwargs["port"] = str(imu_cfg.get("port", "/dev/ttyAMA0"))
        kwargs["baudrate"] = int(imu_cfg.get("baudrate", 9600))
        kwargs["timeout_s"] = float(imu_cfg.get("timeout_s", 0.2))
        kwargs["autostart"] = bool(imu_cfg.get("autostart", True))
    else:
        raise ValueError(f"Unsupported imu.sensor '{sensor}'. Expected 'bno085' or 'jy901'.")

    imu = IMU(sensor=sensor, mock=mock, **kwargs)
    mock_state = imu_cfg.get("mock_state")
    if isinstance(mock_state, dict):
        imu.set_mock(mock, **mock_state)
    return imu


def _normalize_joint_name(name: str) -> Optional[str]:
    s = name.lower().replace("-", "_").replace(" ", "_")
    side = None
    if "left" in s or s.endswith("_l") or s.startswith("l_"):
        side = "left"
    if "right" in s or s.endswith("_r") or s.startswith("r_"):
        side = "right"
    if side is None:
        return None

    if "hipz" in s or "hip_z" in s:
        joint = "hipz"
    elif "hipx" in s or "hip_x" in s:
        joint = "hipx"
    elif "hipy" in s or "hip_y" in s:
        joint = "hipy"
    elif "knee" in s:
        joint = "knee"
    elif "ankley" in s or "ankle_y" in s or "anklepitch" in s or "ankle_pitch" in s:
        joint = "ankle_pitch"
    elif "anklex" in s or "ankle_x" in s or "ankleroll" in s or "ankle_roll" in s:
        joint = "ankle_roll"
    else:
        return None
    return f"{side}.{joint}"


def _default_obs_keys() -> List[str]:
    return [
        "joint_state_deg[0]",
        "joint_state_deg[1]",
        "joint_state_deg[2]",
        "joint_state_deg[3]",
        "joint_state_deg[4]",
        "joint_state_deg[5]",
        "joint_state_deg[6]",
        "joint_state_deg[7]",
        "joint_state_deg[8]",
        "joint_state_deg[9]",
        "joint_state_deg[10]",
        "joint_state_deg[11]",
        "orientation_quaternion_xyzw[0]",
        "orientation_quaternion_xyzw[1]",
        "orientation_quaternion_xyzw[2]",
        "orientation_quaternion_xyzw[3]",
    ]


def _default_action_keys() -> List[str]:
    return [f"{side}.{joint}" for side, joint in JOINT_ORDER_12]


def _extract_policy_terms(cfg: Dict[str, Any]) -> List[str]:
    terms_dict = _get_first(
        cfg,
        keys=(
            "env_cfg.value.observations.policy.terms",
            "env_cfg.observations.policy.terms",
            "observations.policy.terms",
        ),
        default={},
    )
    if isinstance(terms_dict, dict) and terms_dict:
        return list(terms_dict.keys())
    return []


def _extract_action_scales(cfg: Dict[str, Any], action_keys: List[str]) -> List[float]:
    scales_cfg = _get_first(
        cfg,
        keys=(
            "env_cfg.value.actions.joint_pos.scale",
            "env_cfg.actions.joint_pos.scale",
            "actions.joint_pos.scale",
            "action.scale",
        ),
        default={},
    )
    if not isinstance(scales_cfg, dict):
        return [1.0] * len(action_keys)

    def _joint_scale(joint: str) -> float:
        for k, v in scales_cfg.items():
            kl = str(k).lower()
            if joint == "hipz" and "hipz" in kl:
                return float(v)
            if joint == "hipx" and "hipx" in kl:
                return float(v)
            if joint == "hipy" and "hipy" in kl:
                return float(v)
            if joint == "knee" and "knee" in kl:
                return float(v)
            if joint == "ankle_pitch" and ("ankley" in kl or "ankle_y" in kl or "anklepitch" in kl):
                return float(v)
            if joint == "ankle_roll" and ("anklex" in kl or "ankle_x" in kl or "ankleroll" in kl):
                return float(v)
        return 1.0

    out: List[float] = []
    for key in action_keys:
        _, joint = key.split(".", 1)
        out.append(_joint_scale(joint))
    return out


def infer_agent_spec(cfg: Dict[str, Any]) -> AgentSpec:
    obs_raw = _get_first(
        cfg,
        keys=(
            "observation.keys",
            "observation.terms",
            "observations",
            "obs.keys",
            "obs",
            "policy.observation.keys",
            "policy.observations",
        ),
        default=None,
    )
    action_raw = _get_first(
        cfg,
        keys=(
            "action.keys",
            "actions",
            "policy.action.keys",
            "policy.actions",
            "name_mot",
        ),
        default=None,
    )

    obs_keys = _flatten_strings(obs_raw) if obs_raw is not None else _default_obs_keys()
    if not obs_keys:
        obs_keys = _default_obs_keys()

    action_keys: List[str] = []
    for key in (_flatten_strings(action_raw) if action_raw is not None else []):
        norm = _normalize_joint_name(key)
        if norm is not None:
            action_keys.append(norm)
    if not action_keys:
        action_keys = _default_action_keys()

    policy_terms = _extract_policy_terms(cfg)

    history_len = int(
        _get_first(
            cfg,
            keys=(
                "history_len",
                "history.length",
                "observation.history_len",
                "obs_history",
                "env_cfg.value.observations.policy.history_length",
                "env_cfg.observations.policy.history_length",
            ),
            default=1,
        )
        or 1
    )

    inference_hz = float(
        _get_first(
            cfg,
            keys=("inference_hz", "policy.inference_hz", "control_hz"),
            default=50.0,
        )
    )

    action_mode = str(_get_first(cfg, keys=("action_mode", "policy.action_mode"), default="absolute")).lower().strip()
    if action_mode not in ("delta", "absolute"):
        action_mode = "absolute"
    action_scale = float(_get_first(cfg, keys=("action_scale", "policy.action_scale"), default=1.0))

    action_scales = _extract_action_scales(cfg, action_keys)

    return AgentSpec(
        obs_keys=list(obs_keys),
        action_keys=list(action_keys),
        history_len=max(1, history_len),
        inference_hz=max(1.0, inference_hz),
        action_mode=action_mode,
        action_scale=action_scale,
        policy_terms=policy_terms,
        action_scales_rad=action_scales,
    )


class PolicyWrapper:
    def __init__(self, policy: Any):
        self.policy = policy
        self._torch = None
        self._onnx_session = None
        self._onnx_input_name: Optional[str] = None
        self._onnx_output_name: Optional[str] = None
        self._onnx_input_dim: Optional[int] = None
        try:
            import torch  # type: ignore

            self._torch = torch
        except Exception:
            self._torch = None

    @staticmethod
    def load(policy_path: Path, cfg: Dict[str, Any]) -> "PolicyWrapper":
        path = Path(policy_path)
        suffix = path.suffix.lower()

        if suffix == ".onnx":
            try:
                import onnxruntime as ort  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    "onnxruntime is required for ONNX policy inference. Install with `pip install onnxruntime`."
                ) from exc
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            w = PolicyWrapper(policy=None)
            w._onnx_session = sess
            w._onnx_input_name = sess.get_inputs()[0].name
            w._onnx_output_name = sess.get_outputs()[0].name
            ishape = sess.get_inputs()[0].shape
            if isinstance(ishape, list) and len(ishape) >= 2 and isinstance(ishape[-1], int):
                w._onnx_input_dim = int(ishape[-1])
            return w

        if suffix in (".pt", ".pth"):
            try:
                import torch  # type: ignore

                try:
                    model = torch.jit.load(str(path), map_location="cpu")
                    model.eval()
                    return PolicyWrapper(model)
                except Exception:
                    model = torch.load(str(path), map_location="cpu")
                    if hasattr(model, "eval"):
                        model.eval()
                    return PolicyWrapper(model)
            except Exception as exc:
                raise RuntimeError(f"Failed to load torch policy: {exc}") from exc

        if suffix in (".pkl", ".pickle"):
            with path.open("rb") as f:
                obj = pickle.load(f)
            return PolicyWrapper(obj)

        module_name = _get_first(cfg, keys=("policy.module",), default=None)
        fn_name = _get_first(cfg, keys=("policy.loader_fn",), default="load_policy")
        if isinstance(module_name, str) and module_name:
            mod = importlib.import_module(module_name)
            fn = getattr(mod, str(fn_name))
            policy = fn(str(path))
            return PolicyWrapper(policy)

        raise ValueError(f"Unsupported policy format: {path}")

    @property
    def expected_input_dim(self) -> Optional[int]:
        return self._onnx_input_dim

    def infer(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(1, -1)

        if self._onnx_session is not None:
            out = self._onnx_session.run([self._onnx_output_name], {self._onnx_input_name: x})[0]
            return np.asarray(out, dtype=np.float32).reshape(-1)

        if self._torch is not None and hasattr(self.policy, "__call__"):
            with self._torch.no_grad():
                tx = self._torch.from_numpy(x)
                out = self.policy(tx)
                if isinstance(out, tuple):
                    out = out[0]
                if hasattr(out, "detach"):
                    out = out.detach().cpu().numpy()
                return np.asarray(out, dtype=np.float32).reshape(-1)

        if hasattr(self.policy, "predict"):
            out = self.policy.predict(x)
            return np.asarray(out, dtype=np.float32).reshape(-1)
        if hasattr(self.policy, "act"):
            out = self.policy.act(x)
            return np.asarray(out, dtype=np.float32).reshape(-1)
        if callable(self.policy):
            out = self.policy(x)
            return np.asarray(out, dtype=np.float32).reshape(-1)

        raise RuntimeError("Policy object has no supported inference method.")


def _extract_from_snapshot(snapshot: Dict[str, Any], key: str) -> float:
    if key.endswith("]") and "[" in key:
        base, idx_txt = key[:-1].split("[", 1)
        idx = int(idx_txt)
        vec = snapshot.get(base)
        if isinstance(vec, (list, tuple)) and 0 <= idx < len(vec):
            return float(vec[idx])
        return 0.0

    if key.startswith("motors."):
        _, motor_id_txt, field = key.split(".", 2)
        motors = snapshot.get("motors", {})
        item = motors.get(int(motor_id_txt), {}) if isinstance(motors, dict) else {}
        if isinstance(item, dict):
            return float(item.get(field, 0.0))
        return 0.0

    value: Any = snapshot
    for part in key.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return 0.0
    if value is None:
        return 0.0
    return float(value)


def _joint_array_to_command(q_deg: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
    left = {
        "hipz": float(q_deg[0]),
        "hipx": float(q_deg[1]),
        "hipy": float(q_deg[2]),
        "knee": float(q_deg[3]),
        "ankle_pitch": float(q_deg[4]),
        "ankle_roll": float(q_deg[5]),
    }
    right = {
        "hipz": float(q_deg[6]),
        "hipx": float(q_deg[7]),
        "hipy": float(q_deg[8]),
        "knee": float(q_deg[9]),
        "ankle_pitch": float(q_deg[10]),
        "ankle_roll": float(q_deg[11]),
    }
    return left, right


def _quat_xyzw_to_rotmat(q_xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in q_xyzw]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )


@dataclass
class RLAgent:
    robot: BipedalRobotController
    spec: AgentSpec
    policy: PolicyWrapper
    obs_history: deque = field(init=False)
    _running: bool = field(default=False, init=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False)
    _last_obs: Optional[np.ndarray] = field(default=None, init=False)
    _last_action: Optional[np.ndarray] = field(default=None, init=False)
    _last_policy_action: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float32), init=False)
    _default_joint_pos_rad: Optional[np.ndarray] = field(default=None, init=False)
    _prev_q_rad: Optional[np.ndarray] = field(default=None, init=False)
    _prev_q_t_s: Optional[float] = field(default=None, init=False)
    _command_twist: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32), init=False)

    def __post_init__(self) -> None:
        self.obs_history = deque(maxlen=int(self.spec.history_len))

    @classmethod
    def from_files(cls, robot: BipedalRobotController, config_path: str, policy_path: str) -> "RLAgent":
        cfg = _load_config(Path(config_path))
        spec = infer_agent_spec(cfg)
        policy = PolicyWrapper.load(Path(policy_path), cfg)
        return cls(robot=robot, spec=spec, policy=policy)

    def set_command_twist(self, lin_x: float, lin_y: float, yaw_rate: float) -> None:
        self._command_twist = np.array([float(lin_x), float(lin_y), float(yaw_rate)], dtype=np.float32)

    def apply_model_gains_from_mjcf(self, mjcf_path: str) -> Dict[str, float]:
        """
        Load position actuator gains from MJCF and apply them to controller gains.
        Supports:
          - global default gains from <default><position .../>
          - per-joint gains from <actuator><position joint=\"...\" .../>
        Mapping is done on normalized joint keys.
        """
        path = Path(mjcf_path)
        if not path.exists():
            raise FileNotFoundError(f"MJCF file not found: {path}")

        tree = ET.parse(path)
        root = tree.getroot()

        default_kp: Optional[float] = None
        default_kd: Optional[float] = None

        # First <position> encountered under <default> is used as fallback.
        for default_elem in root.findall(".//default"):
            pos = default_elem.find("position")
            if pos is None:
                continue
            kp_txt = pos.attrib.get("kp")
            kv_txt = pos.attrib.get("kv")
            dr_txt = pos.attrib.get("dampratio")
            if kp_txt is None:
                continue
            kp = float(kp_txt)
            if kv_txt is not None:
                kd = float(kv_txt)
            elif dr_txt is not None:
                # Approximation used when MJCF gives damping ratio but no explicit kv.
                kd = 2.0 * float(dr_txt) * float(np.sqrt(max(kp, 0.0)))
            else:
                kd = None
            default_kp, default_kd = kp, kd
            break

        # Collect any per-joint actuator overrides.
        per_joint: Dict[str, tuple[float, Optional[float]]] = {}
        for pos in root.findall(".//actuator/position"):
            jname = pos.attrib.get("joint", "")
            if not jname:
                continue
            jkey = _normalize_joint_name(jname)
            if jkey is None:
                continue
            kp_txt = pos.attrib.get("kp")
            if kp_txt is None:
                continue
            kp = float(kp_txt)
            kv_txt = pos.attrib.get("kv")
            dr_txt = pos.attrib.get("dampratio")
            if kv_txt is not None:
                kd = float(kv_txt)
            elif dr_txt is not None:
                kd = 2.0 * float(dr_txt) * float(np.sqrt(max(kp, 0.0)))
            else:
                kd = None
            per_joint[jkey] = (kp, kd)

        if default_kp is None and not per_joint:
            raise ValueError(f"No position gains found in MJCF: {path}")

        n_applied = 0
        kps: List[float] = []
        kds: List[float] = []
        for joint_key in _default_action_keys():
            motor_id = JOINT_KEY_TO_MOTOR_ID[joint_key]
            kp_kd = per_joint.get(joint_key)
            if kp_kd is None:
                if default_kp is None:
                    continue
                kp, kd = default_kp, default_kd
            else:
                kp, kd = kp_kd

            if kd is None:
                self.robot.set_joint_gains(motor_id, kp=kp)
            else:
                self.robot.set_joint_gains(motor_id, kp=kp, kd=kd)
                kds.append(float(kd))
            kps.append(float(kp))
            n_applied += 1

        return {
            "applied_motors": float(n_applied),
            "kp_mean": float(np.mean(kps)) if kps else 0.0,
            "kp_min": float(np.min(kps)) if kps else 0.0,
            "kp_max": float(np.max(kps)) if kps else 0.0,
            "kd_mean": float(np.mean(kds)) if kds else 0.0,
            "kd_min": float(np.min(kds)) if kds else 0.0,
            "kd_max": float(np.max(kds)) if kds else 0.0,
        }

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    def get_debug_state(self) -> Dict[str, Any]:
        return {
            "running": bool(self._running),
            "history_size": int(len(self.obs_history)),
            "history_len": int(self.spec.history_len),
            "inference_hz": float(self.spec.inference_hz),
            "action_mode": str(self.spec.action_mode),
            "obs_dim": int(self._last_obs.size) if self._last_obs is not None else 0,
            "action_dim": int(self._last_action.size) if self._last_action is not None else 0,
            "policy_terms": list(self.spec.policy_terms),
        }

    def _term_observation_vector(self, snapshot: Dict[str, Any], term_name: str) -> np.ndarray:
        q_deg = np.asarray(snapshot.get("joint_state_deg", [0.0] * 12), dtype=np.float32)
        q_rad = np.deg2rad(q_deg)

        now_s = float(snapshot.get("time_s", time.time()))
        if self._prev_q_rad is None or self._prev_q_t_s is None or now_s <= self._prev_q_t_s:
            qd_rad_s = np.zeros_like(q_rad)
        else:
            dt = max(1e-4, now_s - self._prev_q_t_s)
            qd_rad_s = (q_rad - self._prev_q_rad) / dt

        if term_name == "actions":
            return self._last_policy_action.copy()
        if term_name == "base_ang_vel":
            imu = snapshot.get("imu") or {}
            gyro = imu.get("gyro_rads") if isinstance(imu, dict) else None
            if isinstance(gyro, (list, tuple)) and len(gyro) >= 3:
                return np.array([float(gyro[0]), float(gyro[1]), float(gyro[2])], dtype=np.float32)
            return np.zeros(3, dtype=np.float32)
        if term_name == "base_lin_vel":
            return np.zeros(3, dtype=np.float32)
        if term_name == "command":
            return self._command_twist.copy()
        if term_name == "joint_pos":
            return q_rad.astype(np.float32, copy=False)
        if term_name == "joint_vel":
            return qd_rad_s.astype(np.float32, copy=False)
        if term_name == "projected_gravity":
            q = snapshot.get("orientation_quaternion_xyzw")
            if isinstance(q, (list, tuple)) and len(q) == 4:
                r = _quat_xyzw_to_rotmat(q)
                g_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
                return (r.T @ g_world).astype(np.float32)
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)

        # Unimplemented term -> zero-length contribution.
        return np.zeros(0, dtype=np.float32)

    def _build_obs_now(self) -> np.ndarray:
        snapshot = self.robot.get_combined_state_snapshot(include_joint_state=True)

        if self.spec.policy_terms:
            parts = [self._term_observation_vector(snapshot, t) for t in self.spec.policy_terms]
            obs = np.concatenate(parts, axis=0) if parts else np.zeros(0, dtype=np.float32)
        else:
            values = [_extract_from_snapshot(snapshot, key) for key in self.spec.obs_keys]
            obs = np.asarray(values, dtype=np.float32)

        q_deg = np.asarray(snapshot.get("joint_state_deg", [0.0] * 12), dtype=np.float32)
        q_rad = np.deg2rad(q_deg)
        self._prev_q_rad = q_rad
        self._prev_q_t_s = float(snapshot.get("time_s", time.time()))
        return obs

    def _build_history_obs(self, obs_now: np.ndarray) -> np.ndarray:
        self.obs_history.append(obs_now)
        if len(self.obs_history) < self.spec.history_len:
            first = self.obs_history[0]
            while len(self.obs_history) < self.spec.history_len:
                self.obs_history.appendleft(first.copy())
        return np.concatenate(list(self.obs_history), axis=0).astype(np.float32, copy=False)

    def _adapt_obs_dim_for_policy(self, obs: np.ndarray) -> np.ndarray:
        exp = self.policy.expected_input_dim
        if exp is None:
            return obs
        if obs.size == exp:
            return obs
        if obs.size > exp:
            return obs[:exp]
        out = np.zeros(exp, dtype=np.float32)
        out[: obs.size] = obs
        return out

    def _apply_action(self, action_vec: np.ndarray) -> None:
        act = np.asarray(action_vec, dtype=np.float32).reshape(-1)
        n = min(len(self.spec.action_keys), int(act.size))
        if n <= 0:
            return

        if self._last_policy_action.size != n:
            self._last_policy_action = np.zeros(n, dtype=np.float32)
        self._last_policy_action[:] = act[:n]

        snapshot = self.robot.get_combined_state_snapshot(include_joint_state=True)
        q_cur_deg = np.asarray(snapshot.get("joint_state_deg", [0.0] * 12), dtype=np.float32).reshape(-1)
        if q_cur_deg.size < 12:
            q_cur_deg = np.pad(q_cur_deg, (0, 12 - q_cur_deg.size))
        q_cur_rad = np.deg2rad(q_cur_deg)

        if self._default_joint_pos_rad is None:
            self._default_joint_pos_rad = q_cur_rad.copy()

        q_cmd_rad = q_cur_rad.copy() if self.spec.action_mode == "delta" else self._default_joint_pos_rad.copy()

        default_keys = _default_action_keys()
        for i in range(n):
            key = self.spec.action_keys[i]
            if key not in default_keys:
                continue
            idx = default_keys.index(key)
            scale_i = float(self.spec.action_scales_rad[i]) if i < len(self.spec.action_scales_rad) else 1.0
            value_rad = float(act[i]) * scale_i * float(self.spec.action_scale)
            if self.spec.action_mode == "delta":
                q_cmd_rad[idx] = q_cur_rad[idx] + value_rad
            else:
                q_cmd_rad[idx] = self._default_joint_pos_rad[idx] + value_rad

        q_cmd_deg = np.rad2deg(q_cmd_rad)
        left, right = _joint_array_to_command(q_cmd_deg)
        self.robot.set_action(left=left, right=right)

    def _run_loop(self) -> None:
        period = 1.0 / float(self.spec.inference_hz)
        next_tick = time.perf_counter()
        while self._running:
            obs_now = self._build_obs_now()
            obs_hist = self._build_history_obs(obs_now)
            obs_in = self._adapt_obs_dim_for_policy(obs_hist)
            action = self.policy.infer(obs_in)

            self._last_obs = obs_in
            self._last_action = action
            self._apply_action(action)

            next_tick += period
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.perf_counter()


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run an RL policy agent on BipedalRobotController.")
    parser.add_argument("--config", required=True, help="MJLab-like policy config file (yaml/json).")
    parser.add_argument("--policy", required=True, help="Policy checkpoint path (.onnx/.pt/.pth/.pkl).")
    parser.add_argument("--mjcf", type=str, default=None, help="Optional MJCF robot.xml path to import actuator gains.")
    parser.add_argument("--inference-hz", type=float, default=None, help="Override inference frequency.")
    parser.add_argument("--mode", type=str, default="control", choices=("control", "state_only"))
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    imu = _build_imu_from_config(cfg)
    robot = BipedalRobotController(control_hz=100.0, imu=imu)
    agent = RLAgent.from_files(robot, args.config, args.policy)
    mjcf_path = args.mjcf
    if mjcf_path is None:
        # Convenient default for your current repo layout.
        candidate = Path(args.policy).resolve().parent / "robot.xml"
        if candidate.exists():
            mjcf_path = str(candidate)
    if mjcf_path is not None:
        stats = agent.apply_model_gains_from_mjcf(mjcf_path)
        print(
            f"[RLAgent] applied MJCF gains from {mjcf_path}: "
            f"motors={int(stats['applied_motors'])}, "
            f"kp(mean/min/max)=({stats['kp_mean']:.3f}/{stats['kp_min']:.3f}/{stats['kp_max']:.3f}), "
            f"kd(mean/min/max)=({stats['kd_mean']:.3f}/{stats['kd_min']:.3f}/{stats['kd_max']:.3f})"
        )
    if args.inference_hz is not None:
        agent.spec.inference_hz = float(max(1.0, args.inference_hz))

    robot.start(mode=args.mode, auto_enable=(args.mode == "control"))
    agent.start()
    print(
        f"[RLAgent] running with hz={agent.spec.inference_hz:.1f}, "
        f"history={agent.spec.history_len}, terms={agent.spec.policy_terms}"
    )
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        agent.stop()
        robot.stop(disable_motors=True)


if __name__ == "__main__":
    _main()
