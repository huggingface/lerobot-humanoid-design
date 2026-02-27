from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import argparse
import csv
import importlib
import json
import pickle
import re
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque

import numpy as np

from bipedal_robot import BipedalRobotController
from root_constant import ANKLE_COUPLING_CALIBRATION_LEFT, ANKLE_COUPLING_CALIBRATION_RIGHT, COMMAND_MARGIN_DEG

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

ANKLE_TRUE_LIMIT_CLAMP_EPS_DEG = 0.01


@dataclass
class AgentSpec:
    obs_keys: List[str]
    action_keys: List[str]
    history_len: int = 1
    inference_hz: float = 50.0
    action_mode: str = "absolute"  # absolute or delta
    action_scale: float = 1.0
    ankle_action_abs_limit: Optional[float] = None  # clamp ankle policy outputs before action scaling
    clamp_ankle_to_true_limits: bool = False
    joint_vel_source: str = "snapshot"  # snapshot | finite_diff | auto
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


def _extract_name_list_from_onnx_bytes(
    onnx_path: Path,
    *,
    normalizer: Optional[Any] = None,
    allowed_terms: Optional[set[str]] = None,
) -> List[str]:
    """
    Best-effort extraction of comma-separated name lists embedded in ONNX exports.
    """
    if not onnx_path.exists() or onnx_path.suffix.lower() != ".onnx":
        return []
    try:
        text = onnx_path.read_bytes().decode("latin1", errors="ignore")
    except Exception:
        return []

    best: List[str] = []
    best_score = 0
    for m in re.finditer(r"([A-Za-z0-9_]+(?:,[A-Za-z0-9_]+){3,})", text):
        raw_tokens = [tok.strip().lower() for tok in m.group(1).split(",") if tok.strip()]
        if not raw_tokens:
            continue
        if allowed_terms is not None:
            tokens = []
            for t in raw_tokens:
                mapped = None
                for term in allowed_terms:
                    if t == term or term in t:
                        mapped = term
                        break
                if mapped is not None:
                    tokens.append(mapped)
            score = len(tokens)
        elif normalizer is not None:
            norm_tokens = [normalizer(t) for t in raw_tokens]
            tokens = [t for t in norm_tokens if isinstance(t, str)]
            score = len(tokens)
        else:
            tokens = raw_tokens
            score = len(tokens)
        if score > best_score:
            best = tokens
            best_score = score
    return best


def _extract_onnx_joint_action_keys(policy_path: Path) -> List[str]:
    raw = _extract_name_list_from_onnx_bytes(policy_path, normalizer=_normalize_joint_name)
    if not raw:
        return []
    out: List[str] = []
    for key in raw:
        if key not in out:
            out.append(key)
    return out


def _extract_onnx_policy_terms(policy_path: Path) -> List[str]:
    allowed = {"actions", "base_ang_vel", "base_lin_vel", "command", "joint_pos", "joint_vel", "projected_gravity"}
    terms = _extract_name_list_from_onnx_bytes(policy_path, allowed_terms=allowed)
    out: List[str] = []
    for t in terms:
        if t in allowed and t not in out:
            out.append(t)
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
    clamp_ankle_to_true_limits_raw = _get_first(
        cfg,
        keys=(
            "clamp_ankle_to_true_limits",
            "policy.clamp_ankle_to_true_limits",
            "clamp_ankle_to_limits",
            "policy.clamp_ankle_to_limits",
        ),
        default=False,
    )
    if isinstance(clamp_ankle_to_true_limits_raw, str):
        clamp_ankle_to_true_limits = clamp_ankle_to_true_limits_raw.strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    else:
        clamp_ankle_to_true_limits = bool(clamp_ankle_to_true_limits_raw)
    ankle_action_abs_limit_raw = _get_first(
        cfg,
        keys=(
            "ankle_action_abs_limit",
            "policy.ankle_action_abs_limit",
            "action.ankle_action_abs_limit",
            "policy.action.ankle_action_abs_limit",
        ),
        default=None,
    )
    ankle_action_abs_limit = None
    if ankle_action_abs_limit_raw is not None:
        try:
            ankle_action_abs_limit = abs(float(ankle_action_abs_limit_raw))
        except Exception:
            ankle_action_abs_limit = None

    joint_vel_source = str(
        _get_first(
            cfg,
            keys=(
                "joint_vel_source",
                "policy.joint_vel_source",
                "observation.joint_vel_source",
                "observations.joint_vel_source",
            ),
            default="snapshot",
        )
    ).strip().lower()
    joint_vel_source_alias = {
        "fd": "finite_diff",
        "finite_difference": "finite_diff",
        "finite_differences": "finite_diff",
        "joint_state": "snapshot",
        "joint_state_fd_fallback": "auto",
    }
    joint_vel_source = joint_vel_source_alias.get(joint_vel_source, joint_vel_source)
    if joint_vel_source not in ("snapshot", "finite_diff", "auto"):
        joint_vel_source = "snapshot"

    action_scales = _extract_action_scales(cfg, action_keys)

    return AgentSpec(
        obs_keys=list(obs_keys),
        action_keys=list(action_keys),
        history_len=max(1, history_len),
        inference_hz=max(1.0, inference_hz),
        action_mode=action_mode,
        action_scale=action_scale,
        ankle_action_abs_limit=ankle_action_abs_limit,
        clamp_ankle_to_true_limits=clamp_ankle_to_true_limits,
        joint_vel_source=joint_vel_source,
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
        x = _sanitize_f32_vector(obs).reshape(1, -1)

        if self._onnx_session is not None:
            out = self._onnx_session.run([self._onnx_output_name], {self._onnx_input_name: x})[0]
            return _sanitize_f32_vector(out)

        if self._torch is not None and hasattr(self.policy, "__call__"):
            with self._torch.no_grad():
                tx = self._torch.from_numpy(x)
                out = self.policy(tx)
                if isinstance(out, tuple):
                    out = out[0]
                if hasattr(out, "detach"):
                    out = out.detach().cpu().numpy()
                return _sanitize_f32_vector(out)

        if hasattr(self.policy, "predict"):
            out = self.policy.predict(x)
            return _sanitize_f32_vector(out)
        if hasattr(self.policy, "act"):
            out = self.policy.act(x)
            return _sanitize_f32_vector(out)
        if callable(self.policy):
            out = self.policy(x)
            return _sanitize_f32_vector(out)

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


def _sanitize_f32_vector(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _ankle_cfg_for_side(side: str) -> Dict[str, Any]:
    return ANKLE_COUPLING_CALIBRATION_LEFT if side == "left" else ANKLE_COUPLING_CALIBRATION_RIGHT


def _clamp_within_interval(value: float, lo: float, hi: float) -> float:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return float(value)
    if lo > hi:
        lo, hi = hi, lo
    return float(np.clip(float(value), float(lo), float(hi)))


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
    _command_source: Optional[Any] = field(default=None, init=False)
    _warn_last_t_s: Dict[str, float] = field(default_factory=dict, init=False)
    _warn_counts: Dict[str, int] = field(default_factory=dict, init=False)
    _warn_interval_s: float = field(default=1.0, init=False)
    _log_obs: bool = field(default=False, init=False)
    _log_action: bool = field(default=False, init=False)
    _log_every_n: int = field(default=1, init=False)
    _log_path: Optional[Path] = field(default=None, init=False)
    _log_file: Optional[Any] = field(default=None, init=False)
    _log_writer: Optional[Any] = field(default=None, init=False)
    _log_step_idx: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.obs_history = deque(maxlen=int(self.spec.history_len))

    def _snapshot_has_valid_joint_state(self, snapshot: Dict[str, Any]) -> bool:
        motors = snapshot.get("motors")
        if not isinstance(motors, dict) or not motors:
            return False
        for mid in JOINT_KEY_TO_MOTOR_ID.values():
            st = motors.get(mid, motors.get(str(mid)))
            if not isinstance(st, dict):
                return False
            try:
                if float(st.get("stamp", 0.0)) <= 0.0:
                    return False
            except Exception:
                return False
        return True

    def _clamp_ankles_to_true_limits(self, q_cmd_deg: np.ndarray) -> np.ndarray:
        if not bool(self.spec.clamp_ankle_to_true_limits):
            return q_cmd_deg
        limits = getattr(self.robot, "command_limits_cal_deg", None)
        if not isinstance(limits, dict):
            return q_cmd_deg
        q = np.asarray(q_cmd_deg, dtype=np.float32).copy()
        for side_name, pitch_idx, roll_idx, a1_mid, a2_mid in (
            ("left", 4, 5, 5, 6),
            ("right", 10, 11, 11, 12),
        ):
            lim1 = limits.get(a1_mid)
            lim2 = limits.get(a2_mid)
            if lim1 is None or lim2 is None:
                continue
            a1_lo, a1_hi = float(min(lim1)), float(max(lim1))
            a2_lo, a2_hi = float(min(lim2)), float(max(lim2))
            # Match the controller's safe command margin so agent-side clamp is consistent.
            # Stay slightly inside the controller's accepted interval to avoid
            # edge-case rejects from float rounding at the exact boundary.
            extra_eps = float(ANKLE_TRUE_LIMIT_CLAMP_EPS_DEG)
            a1_lo += float(COMMAND_MARGIN_DEG) + extra_eps
            a1_hi -= float(COMMAND_MARGIN_DEG) + extra_eps
            a2_lo += float(COMMAND_MARGIN_DEG) + extra_eps
            a2_hi -= float(COMMAND_MARGIN_DEG) + extra_eps
            if a1_lo > a1_hi:
                a1_lo, a1_hi = float(min(lim1)), float(max(lim1))
            if a2_lo > a2_hi:
                a2_lo, a2_hi = float(min(lim2)), float(max(lim2))

            cfg = _ankle_cfg_for_side(side_name)
            sign_p = float(cfg["pitch"]["sign"])
            off_p = float(cfg["pitch"]["offset_deg"])
            sign_r = float(cfg["roll"]["sign"])
            off_r = float(cfg["roll"]["offset_deg"])

            p = float(q[pitch_idx])
            r = float(q[roll_idx])
            p_lin = (p - off_p) / sign_p
            r_lin = (r - off_r) / sign_r

            # Clamp pitch while holding roll fixed.
            p_lin_lo = max(a1_lo - r_lin, r_lin - a2_hi)
            p_lin_hi = min(a1_hi - r_lin, r_lin - a2_lo)
            if p_lin_lo <= p_lin_hi:
                p_lin = _clamp_within_interval(p_lin, p_lin_lo, p_lin_hi)
                p = sign_p * p_lin + off_p

            # Clamp roll while holding (possibly updated) pitch fixed.
            p_lin = (p - off_p) / sign_p
            r_lin_lo = max(a1_lo - p_lin, a2_lo + p_lin)
            r_lin_hi = min(a1_hi - p_lin, a2_hi + p_lin)
            if r_lin_lo <= r_lin_hi:
                r_lin = _clamp_within_interval(r_lin, r_lin_lo, r_lin_hi)
                r = sign_r * r_lin + off_r

            # Final projection pass on pitch to account for updated roll.
            r_lin = (r - off_r) / sign_r
            p_lin_lo = max(a1_lo - r_lin, r_lin - a2_hi)
            p_lin_hi = min(a1_hi - r_lin, r_lin - a2_lo)
            if p_lin_lo <= p_lin_hi:
                p_lin = _clamp_within_interval((p - off_p) / sign_p, p_lin_lo, p_lin_hi)
                p = sign_p * p_lin + off_p

            if abs(p - float(q[pitch_idx])) > 1e-6 or abs(r - float(q[roll_idx])) > 1e-6:
                self._warn(
                    "ankle_joint_limit_clamped",
                    (
                        f"{side_name} ankle cmd clamped "
                        f"pitch {float(q[pitch_idx]):.2f}->{p:.2f} deg, "
                        f"roll {float(q[roll_idx]):.2f}->{r:.2f} deg"
                    ),
                )
            q[pitch_idx] = p
            q[roll_idx] = r
        return q

    def _warn(self, key: str, message: str) -> None:
        now = time.time()
        count = int(self._warn_counts.get(key, 0)) + 1
        self._warn_counts[key] = count
        last_t = float(self._warn_last_t_s.get(key, 0.0))
        if (now - last_t) < float(self._warn_interval_s):
            return
        self._warn_last_t_s[key] = now
        print(f"[RLAgent][WARN][{key}] {message} (count={count})")

    @classmethod
    def from_files(
        cls,
        robot: BipedalRobotController,
        config_path: str,
        policy_path: str,
        *,
        log_path: Optional[str] = None,
        log_observation: bool = False,
        log_action: bool = False,
        log_every_n: int = 1,
        ankle_action_abs_limit: Optional[float] = None,
        clamp_ankle_to_true_limits: Optional[bool] = None,
    ) -> "RLAgent":
        cfg = _load_config(Path(config_path))
        spec = infer_agent_spec(cfg)
        ppath = Path(policy_path)

        # Prefer model-embedded ordering when available. This avoids action/obs
        # permutation bugs between exported policy and YAML dict key order.
        onnx_action_keys = _extract_onnx_joint_action_keys(ppath)
        if onnx_action_keys:
            spec.action_keys = onnx_action_keys
            spec.action_scales_rad = _extract_action_scales(cfg, spec.action_keys)
        onnx_terms = _extract_onnx_policy_terms(ppath)
        if len(onnx_terms) >= len(spec.policy_terms):
            spec.policy_terms = onnx_terms
        if ankle_action_abs_limit is not None:
            spec.ankle_action_abs_limit = abs(float(ankle_action_abs_limit))
        if clamp_ankle_to_true_limits is not None:
            spec.clamp_ankle_to_true_limits = bool(clamp_ankle_to_true_limits)

        policy = PolicyWrapper.load(ppath, cfg)
        agent = cls(robot=robot, spec=spec, policy=policy)
        if log_observation or log_action:
            resolved_log_path = Path(log_path) if log_path else Path("rl_agent_debug_log.csv")
            agent.configure_logging(
                log_path=resolved_log_path,
                log_observation=log_observation,
                log_action=log_action,
                log_every_n=log_every_n,
            )
        return agent

    def configure_logging(
        self,
        *,
        log_path: Path,
        log_observation: bool = True,
        log_action: bool = True,
        log_every_n: int = 1,
    ) -> None:
        self._close_log_file()
        self._log_obs = bool(log_observation)
        self._log_action = bool(log_action)
        self._log_every_n = max(1, int(log_every_n))
        self._log_step_idx = 0
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        f = self._log_path.open("w", newline="")
        writer = csv.writer(f)
        # `action_pre_scale` is the direct policy output before action_scale/joint scaling.
        writer.writerow(["time_s", "step", "observation", "action_pre_scale"])
        f.flush()
        self._log_file = f
        self._log_writer = writer
        print(
            f"[RLAgent] logging enabled path={self._log_path} "
            f"(obs={self._log_obs}, action={self._log_action}, every_n={self._log_every_n})"
        )

    def disable_logging(self) -> None:
        self._log_obs = False
        self._log_action = False
        self._close_log_file()

    def _close_log_file(self) -> None:
        if self._log_file is not None:
            try:
                self._log_file.flush()
                self._log_file.close()
            except Exception:
                pass
        self._log_file = None
        self._log_writer = None
        self._log_path = None

    def _maybe_log_step(self, obs: np.ndarray, action_pre_scale: np.ndarray) -> None:
        if self._log_writer is None or self._log_file is None:
            return
        self._log_step_idx += 1
        if (self._log_step_idx % self._log_every_n) != 0:
            return
        obs_payload = ""
        action_payload = ""
        if self._log_obs:
            obs_payload = json.dumps(np.asarray(obs, dtype=np.float32).reshape(-1).tolist(), separators=(",", ":"))
        if self._log_action:
            action_payload = json.dumps(
                np.asarray(action_pre_scale, dtype=np.float32).reshape(-1).tolist(), separators=(",", ":")
            )
        self._log_writer.writerow([f"{time.time():.6f}", str(self._log_step_idx), obs_payload, action_payload])
        self._log_file.flush()

    def set_command_twist(self, lin_x: float, lin_y: float, yaw_rate: float) -> None:
        self._command_twist = np.array([float(lin_x), float(lin_y), float(yaw_rate)], dtype=np.float32)

    def set_command_source(self, source: Optional[Any]) -> None:
        """
        Attach external command source with method:
          - get_command_twist() -> (lin_x, lin_y, yaw_rate)
        """
        self._command_source = source

    def _refresh_command_from_source(self) -> None:
        src = self._command_source
        if src is None:
            return
        getter = getattr(src, "get_command_twist", None)
        if getter is None or not callable(getter):
            self._warn("cmd_source_invalid", "command source has no callable get_command_twist()")
            return
        cmd = getter()
        if not isinstance(cmd, (list, tuple, np.ndarray)) or len(cmd) < 3:
            self._warn("cmd_source_bad_value", f"command source returned invalid value type={type(cmd).__name__}")
            return
        self.set_command_twist(float(cmd[0]), float(cmd[1]), float(cmd[2]))

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
        self._close_log_file()

    def get_debug_state(self) -> Dict[str, Any]:
        return {
            "running": bool(self._running),
            "history_size": int(len(self.obs_history)),
            "history_len": int(self.spec.history_len),
            "inference_hz": float(self.spec.inference_hz),
            "command_twist": [float(v) for v in self._command_twist.tolist()],
            "command_source_attached": bool(self._command_source is not None),
            "action_mode": str(self.spec.action_mode),
            "joint_vel_source": str(self.spec.joint_vel_source),
            "obs_dim": int(self._last_obs.size) if self._last_obs is not None else 0,
            "action_dim": int(self._last_action.size) if self._last_action is not None else 0,
            "policy_terms": list(self.spec.policy_terms),
            "warn_counts": dict(self._warn_counts),
        }

    def _term_observation_vector(self, snapshot: Dict[str, Any], term_name: str) -> np.ndarray:
        q_raw = np.asarray(snapshot.get("joint_state_deg", [0.0] * 12), dtype=np.float32).reshape(-1)
        bad_q = np.where(~np.isfinite(q_raw))[0]
        if bad_q.size:
            self._warn("joint_state_nonfinite", f"joint_state_deg has non-finite at idx={bad_q.tolist()[:6]}")
        q_deg = _sanitize_f32_vector(q_raw)
        if q_deg.size < 12:
            q_deg = np.pad(q_deg, (0, 12 - q_deg.size))
        q_rad = np.deg2rad(q_deg)
        q_rad = _sanitize_f32_vector(q_rad)

        if self._default_joint_pos_rad is None and self._snapshot_has_valid_joint_state(snapshot):
            self._default_joint_pos_rad = q_rad.copy()

        now_s = float(snapshot.get("time_s", time.time()))
        qd_snap = np.asarray(snapshot.get("joint_velocity_rad_s", []), dtype=np.float32).reshape(-1)
        qd_fd_available = self._prev_q_rad is not None and self._prev_q_t_s is not None and now_s > self._prev_q_t_s
        if qd_fd_available:
            dt = max(1e-4, now_s - self._prev_q_t_s)
            qd_fd = _sanitize_f32_vector((q_rad - self._prev_q_rad) / dt)
        else:
            qd_fd = np.zeros_like(q_rad)
        qd_snap_ok = qd_snap.size >= 12
        qd_snap_vec = _sanitize_f32_vector(qd_snap[:12]) if qd_snap_ok else np.zeros_like(q_rad)

        joint_vel_source = str(getattr(self.spec, "joint_vel_source", "snapshot") or "snapshot").strip().lower()
        if joint_vel_source == "finite_diff":
            qd_rad_s = qd_fd
        elif joint_vel_source == "auto":
            qd_rad_s = qd_snap_vec if qd_snap_ok else qd_fd
        else:
            qd_rad_s = qd_snap_vec if qd_snap_ok else qd_fd

        if term_name == "actions":
            return self._last_policy_action.copy()
        if term_name == "base_ang_vel":
            imu = snapshot.get("imu") or {}
            gyro = None
            if isinstance(imu, dict):
                gyro = imu.get("gyro_rads")
                if gyro is None:
                    gyro = imu.get("ang_vel_rad_s")
            if isinstance(gyro, (list, tuple)) and len(gyro) >= 3:
                g_raw = np.asarray([float(gyro[0]), float(gyro[1]), float(gyro[2])], dtype=np.float32)
                if not np.all(np.isfinite(g_raw)):
                    self._warn("imu_gyro_nonfinite", f"imu gyro contains non-finite values={g_raw.tolist()}")
                return _sanitize_f32_vector([float(gyro[0]), float(gyro[1]), float(gyro[2])])
            return np.zeros(3, dtype=np.float32)
        if term_name == "base_lin_vel":
            imu = snapshot.get("imu") or {}
            lin_vel = None
            if isinstance(imu, dict):
                lin_vel = imu.get("linear_velocity_mps")
                if lin_vel is None:
                    lin_vel = imu.get("lin_vel_m_s")
            if isinstance(lin_vel, (list, tuple)) and len(lin_vel) >= 3:
                v_raw = np.asarray([float(lin_vel[0]), float(lin_vel[1]), float(lin_vel[2])], dtype=np.float32)
                if not np.all(np.isfinite(v_raw)):
                    self._warn("imu_linvel_nonfinite", f"imu linear velocity contains non-finite values={v_raw.tolist()}")
                return _sanitize_f32_vector([float(lin_vel[0]), float(lin_vel[1]), float(lin_vel[2])])
            return np.zeros(3, dtype=np.float32)
        if term_name == "command":
            return _sanitize_f32_vector(self._command_twist)
        if term_name == "joint_pos":
            # Policy was trained on joint_pos_rel observations.
            if self._default_joint_pos_rad is None:
                return np.zeros_like(q_rad)
            return _sanitize_f32_vector(q_rad - self._default_joint_pos_rad)
        if term_name == "joint_vel":
            return _sanitize_f32_vector(qd_rad_s)
        if term_name == "projected_gravity":
            q = snapshot.get("orientation_quaternion_xyzw")
            if isinstance(q, (list, tuple)) and len(q) == 4:
                q_arr = np.asarray([float(v) for v in q], dtype=np.float32)
                if not np.all(np.isfinite(q_arr)):
                    self._warn("imu_quat_nonfinite", f"orientation quaternion non-finite values={q_arr.tolist()}")
                qn = float(np.linalg.norm(q_arr))
                if qn < 1e-8:
                    self._warn("imu_quat_zero_norm", f"orientation quaternion near zero norm={qn:.3e}")
                r = _quat_xyzw_to_rotmat(q)
                g_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
                return _sanitize_f32_vector(r.T @ g_world)
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
        bad_obs = np.where(~np.isfinite(np.asarray(obs, dtype=np.float32).reshape(-1)))[0]
        if bad_obs.size:
            self._warn("obs_nonfinite", f"observation vector contains non-finite at idx={bad_obs.tolist()[:8]}")

        q_deg = _sanitize_f32_vector(snapshot.get("joint_state_deg", [0.0] * 12))
        if q_deg.size < 12:
            q_deg = np.pad(q_deg, (0, 12 - q_deg.size))
        q_rad = np.deg2rad(q_deg)
        q_rad = _sanitize_f32_vector(q_rad)
        if self._default_joint_pos_rad is None and self._snapshot_has_valid_joint_state(snapshot):
            self._default_joint_pos_rad = q_rad.copy()
        self._prev_q_rad = q_rad
        self._prev_q_t_s = float(snapshot.get("time_s", time.time()))
        return _sanitize_f32_vector(obs)

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
        act_raw = np.asarray(action_vec, dtype=np.float32).reshape(-1)
        bad_act = np.where(~np.isfinite(act_raw))[0]
        if bad_act.size:
            self._warn("action_nonfinite", f"policy action has non-finite at idx={bad_act.tolist()[:8]}")
        act = _sanitize_f32_vector(act_raw)
        if act.size > 0:
            max_abs = float(np.max(np.abs(act)))
            if max_abs > 5.0:
                self._warn("action_large", f"policy action abs max={max_abs:.3f} (>5.0)")
        n = min(len(self.spec.action_keys), int(act.size))
        if n <= 0:
            return

        if self._last_policy_action.size != n:
            self._last_policy_action = np.zeros(n, dtype=np.float32)
        self._last_policy_action[:] = act[:n]

        snapshot = self.robot.get_combined_state_snapshot(include_joint_state=True)
        if not self._snapshot_has_valid_joint_state(snapshot):
            self._warn("action_wait_state", "skipping action apply until valid motor state snapshot is available")
            return
        q_cur_deg = _sanitize_f32_vector(snapshot.get("joint_state_deg", [0.0] * 12))
        if q_cur_deg.size < 12:
            q_cur_deg = np.pad(q_cur_deg, (0, 12 - q_cur_deg.size))
        q_cur_rad = _sanitize_f32_vector(np.deg2rad(q_cur_deg))

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
            act_i = float(act[i])
            ankle_lim = self.spec.ankle_action_abs_limit
            if (
                ankle_lim is not None
                and ankle_lim > 0.0
                and (key.endswith("ankle_pitch") or key.endswith("ankle_roll"))
            ):
                clipped = float(np.clip(act_i, -ankle_lim, ankle_lim))
                if clipped != act_i:
                    self._warn(
                        "ankle_action_clamped",
                        f"{key} policy action {act_i:.3f} clamped to {clipped:.3f} (limit={ankle_lim:.3f})",
                    )
                act_i = clipped
            value_rad = act_i * scale_i * float(self.spec.action_scale)
            if not np.isfinite(value_rad):
                continue
            if self.spec.action_mode == "delta":
                q_cmd_rad[idx] = q_cur_rad[idx] + value_rad
            else:
                q_cmd_rad[idx] = self._default_joint_pos_rad[idx] + value_rad

        q_cmd_deg = _sanitize_f32_vector(np.rad2deg(q_cmd_rad))
        q_cmd_deg = self._clamp_ankles_to_true_limits(q_cmd_deg)
        dq_cmd_deg = q_cmd_deg - q_cur_deg[: q_cmd_deg.size]
        max_step = float(np.max(np.abs(dq_cmd_deg))) if dq_cmd_deg.size else 0.0
        if max_step > 60.0:
            self._warn("command_step_large", f"joint command step abs max={max_step:.2f} deg")
        left, right = _joint_array_to_command(q_cmd_deg)
        self.robot.set_action(left=left, right=right)

    def _run_loop(self) -> None:
        period = 1.0 / float(self.spec.inference_hz)
        next_tick = time.perf_counter()
        while self._running:
            try:
                self._refresh_command_from_source()
                obs_now = self._build_obs_now()
                obs_hist = self._build_history_obs(obs_now)
                obs_in = self._adapt_obs_dim_for_policy(obs_hist)
                action = self.policy.infer(obs_in)

                self._last_obs = obs_in
                self._last_action = action
                self._maybe_log_step(obs_in, action)
                self._apply_action(action)
            except Exception as exc:
                self._warn("loop_exception", f"run loop exception: {type(exc).__name__}: {exc}")

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
    parser.add_argument("--log-path", type=str, default=None, help="Optional CSV path for debug logging.")
    parser.add_argument("--log-observation", action="store_true", help="Log policy observation vectors.")
    parser.add_argument("--log-action", action="store_true", help="Log policy action vectors.")
    parser.add_argument("--log-every-n", type=int, default=1, help="Log every N inference steps.")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    imu = _build_imu_from_config(cfg)
    robot = BipedalRobotController(control_hz=100.0, imu=imu)
    agent = RLAgent.from_files(
        robot,
        args.config,
        args.policy,
        log_path=args.log_path,
        log_observation=bool(args.log_observation),
        log_action=bool(args.log_action),
        log_every_n=max(1, int(args.log_every_n)),
    )
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
