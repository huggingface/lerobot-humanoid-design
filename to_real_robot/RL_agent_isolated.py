from __future__ import annotations

import csv
import importlib
import json
import pickle
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from bipedal_robot import BipedalRobotController
except Exception:  # pragma: no cover
    BipedalRobotController = Any  # type: ignore

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


POLICY_ACTION_KEYS = [
    "right.hipz",
    "right.hipx",
    "right.hipy",
    "right.knee",
    "right.ankle_pitch",
    "right.ankle_roll",
    "left.hipz",
    "left.hipx",
    "left.hipy",
    "left.knee",
    "left.ankle_pitch",
    "left.ankle_roll",
]

# Snapshot joint_state_* order from robot API:
# [left(6), right(6)] -> policy order [right(6), left(6)].
SNAPSHOT_TO_POLICY_JOINT_IDX = np.array([6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5], dtype=np.int64)
JOINT_TORQUE_TERM_NAMES = ("joint_torque", "joint_torques", "joint_effort", "joint_efforts")


@dataclass
class AgentSpec:
    action_keys: List[str]
    history_len: int = 1
    inference_hz: float = 50.0
    action_scale: float = 1.0
    policy_terms: List[str] = field(default_factory=list)
    action_scales_rad: List[float] = field(default_factory=list)
    encoder_bias_rad: List[float] = field(default_factory=list)
    obs_term_scales: Dict[str, float] = field(default_factory=dict)
    joint_vel_source: str = "auto"  # auto | finite_diff
    debug_zero_actions_obs: bool = False
    # "canonical" (mjlab): terms reordered to (base_ang_vel, projected_gravity,
    # joint_pos, joint_vel, actions, command); history is time-major
    # ([obs_{t-H+1}, ..., obs_t]).
    # "isaaclab" (AGILE / Isaac Lab): terms stay in env.yaml insertion order;
    # history is term-major ([term0_{t-H+1..t}, term1_{t-H+1..t}, ...]) to match
    # Isaac Lab's flatten_history_dim=True + concatenate_terms=True export.
    obs_layout: str = "canonical"
    # Per-term dims in policy_terms order, populated lazily on first obs build.
    term_dims: List[int] = field(default_factory=list)


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
        return [str(k) for k in terms_dict.keys()]
    return []


def _canonicalize_policy_terms(policy_terms: List[str]) -> List[str]:
    terms = [str(t) for t in policy_terms]
    terms_set = set(terms)
    core_terms = ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command")
    supported_terms = set(core_terms) | {"base_lin_vel"} | set(JOINT_TORQUE_TERM_NAMES)
    torque_term = next((term for term in terms if term in JOINT_TORQUE_TERM_NAMES), None)
    if (
        set(core_terms).issubset(terms_set)
        and len([term for term in terms_set if term in JOINT_TORQUE_TERM_NAMES]) <= 1
        and not (terms_set - supported_terms)
    ):
        canonical: List[str] = []
        if "base_lin_vel" in terms_set:
            canonical.append("base_lin_vel")
        canonical.extend(core_terms)
        if torque_term is not None:
            canonical.append(torque_term)
        return canonical
    return terms


def _extract_action_keys(cfg: Dict[str, Any]) -> List[str]:
    action_raw = _get_first(
        cfg,
        keys=(
            "action.keys",
            "actions",
            "policy.action.keys",
            "policy.actions",
            "name_mot",
            "env_cfg.value.actions.joint_pos.joint_names",
            "env_cfg.actions.joint_pos.joint_names",
            "actions.joint_pos.joint_names",
        ),
        default=None,
    )
    parsed: List[str] = []
    for key in (_flatten_strings(action_raw) if action_raw is not None else []):
        key_s = str(key).strip()
        if not key_s or ".*" in key_s:
            continue
        norm = _normalize_joint_name(key_s)
        if norm is not None:
            parsed.append(norm)

    if len(parsed) == 12:
        return parsed
    return list(POLICY_ACTION_KEYS)


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


def _extract_observation_term_scales(cfg: Dict[str, Any], policy_terms: List[str]) -> Dict[str, float]:
    terms_cfg = _get_first(
        cfg,
        keys=(
            "env_cfg.value.observations.policy.terms",
            "env_cfg.observations.policy.terms",
            "observations.policy.terms",
        ),
        default={},
    )
    if not isinstance(terms_cfg, dict):
        return {}

    out: Dict[str, float] = {}
    for term_name in policy_terms:
        term_cfg = terms_cfg.get(term_name)
        if not isinstance(term_cfg, dict):
            continue
        scale = term_cfg.get("scale", None)
        if scale is None:
            continue
        try:
            out[str(term_name)] = float(scale)
        except Exception:
            continue
    return out


def _extract_encoder_bias_rad(cfg: Dict[str, Any], action_keys: List[str]) -> List[float]:
    raw = _get_first(
        cfg,
        keys=(
            "encoder_bias_rad",
            "policy.encoder_bias_rad",
            "observation.encoder_bias_rad",
            "env_cfg.value.events.encoder_bias.value",
        ),
        default=None,
    )
    if raw is None:
        return [0.0] * len(action_keys)
    if isinstance(raw, (list, tuple)):
        vals = [float(v) for v in raw]
        if len(vals) < len(action_keys):
            vals = vals + [0.0] * (len(action_keys) - len(vals))
        return vals[: len(action_keys)]
    if isinstance(raw, dict):
        out: List[float] = []
        for key in action_keys:
            candidates = (
                key,
                key.replace(".", "_"),
                key.replace(".", ""),
                key.split(".", 1)[1].replace("ankle_pitch", "ankley").replace("ankle_roll", "anklex")
                + "_"
                + key.split(".", 1)[0],
            )
            v = 0.0
            for c in candidates:
                if c in raw:
                    v = float(raw[c])
                    break
            out.append(v)
        return out
    return [0.0] * len(action_keys)


def _extract_default_joint_pos_rad_from_cfg(cfg: Dict[str, Any]) -> Optional[np.ndarray]:
    joint_pos = _get_first(
        cfg,
        keys=(
            "env_cfg.value.scene.entities.robot.init_state.joint_pos",
            "env_cfg.scene.entities.robot.init_state.joint_pos",
            "scene.entities.robot.init_state.joint_pos",
        ),
        default=None,
    )
    if not isinstance(joint_pos, dict):
        return None

    names = [
        "hipz_right",
        "hipx_right",
        "hipy_right",
        "knee_right",
        "ankley_right",
        "anklex_right",
        "hipz_left",
        "hipx_left",
        "hipy_left",
        "knee_left",
        "ankley_left",
        "anklex_left",
    ]
    q = np.zeros(12, dtype=np.float32)
    for i, name in enumerate(names):
        if name not in joint_pos:
            return None
        try:
            q[i] = float(joint_pos[name])
        except Exception:
            return None
    return q


def infer_agent_spec(cfg: Dict[str, Any]) -> AgentSpec:
    action_keys = _extract_action_keys(cfg)
    raw_terms = _extract_policy_terms(cfg)
    obs_layout = str(cfg.get("_obs_layout", "canonical")).strip().lower() or "canonical"
    if obs_layout == "isaaclab":
        # Preserve training-time insertion order; the exported ONNX was built
        # with flatten_history_dim=True + concatenate_terms=True, so the input
        # vector order is determined by env.yaml term order, not by mjlab's
        # canonical preference.
        policy_terms = list(raw_terms)
    else:
        policy_terms = _canonicalize_policy_terms(raw_terms)
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
    inference_hz = float(_get_first(cfg, keys=("inference_hz", "policy.inference_hz", "control_hz"), default=50.0))
    action_scale = float(_get_first(cfg, keys=("action_scale", "policy.action_scale"), default=1.0))
    joint_vel_source = str(
        _get_first(
            cfg,
            keys=(
                "joint_vel_source",
                "policy.joint_vel_source",
                "observation.joint_vel_source",
                "observations.joint_vel_source",
            ),
            default="auto",
        )
    ).strip().lower()
    if joint_vel_source in ("fd", "finite_difference", "finite_differences"):
        joint_vel_source = "finite_diff"
    if joint_vel_source in ("snapshot", "joint_state", "joint_state_fd_fallback"):
        joint_vel_source = "auto"
    if joint_vel_source not in ("auto", "finite_diff"):
        joint_vel_source = "auto"

    return AgentSpec(
        action_keys=list(action_keys),
        history_len=max(1, history_len),
        inference_hz=max(1.0, inference_hz),
        action_scale=action_scale,
        policy_terms=policy_terms,
        action_scales_rad=_extract_action_scales(cfg, action_keys),
        encoder_bias_rad=_extract_encoder_bias_rad(cfg, action_keys),
        obs_term_scales=_extract_observation_term_scales(cfg, policy_terms),
        joint_vel_source=joint_vel_source,
        obs_layout=obs_layout,
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
                raise RuntimeError("onnxruntime is required for ONNX policy inference.") from exc
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
            return PolicyWrapper(fn(str(path)))

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
            return np.asarray(self.policy.predict(x), dtype=np.float32).reshape(-1)
        if hasattr(self.policy, "act"):
            return np.asarray(self.policy.act(x), dtype=np.float32).reshape(-1)
        if callable(self.policy):
            return np.asarray(self.policy(x), dtype=np.float32).reshape(-1)
        raise RuntimeError("Policy object has no supported inference method.")


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


def _action_array_policy_order_to_command(q_deg_policy_order: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
    q = np.asarray(q_deg_policy_order, dtype=np.float32).reshape(-1)
    if q.size < 12:
        q = np.pad(q, (0, 12 - q.size))
    right = {
        "hipz": float(q[0]),
        "hipx": float(q[1]),
        "hipy": float(q[2]),
        "knee": float(q[3]),
        "ankle_pitch": float(q[4]),
        "ankle_roll": float(q[5]),
    }
    left = {
        "hipz": float(q[6]),
        "hipx": float(q[7]),
        "hipy": float(q[8]),
        "knee": float(q[9]),
        "ankle_pitch": float(q[10]),
        "ankle_roll": float(q[11]),
    }
    return left, right


@dataclass
class RLAgent:
    robot: BipedalRobotController
    spec: AgentSpec
    policy: PolicyWrapper
    obs_history: deque = field(init=False)
    _running: bool = field(default=False, init=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False)
    _default_joint_pos_rad: Optional[np.ndarray] = field(default=None, init=False)
    _command_twist: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32), init=False)
    _command_source: Optional[Any] = field(default=None, init=False)
    _last_obs: Optional[np.ndarray] = field(default=None, init=False)
    _last_action: Optional[np.ndarray] = field(default=None, init=False)
    _last_policy_action: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float32), init=False)
    _prev_q_rad: Optional[np.ndarray] = field(default=None, init=False)
    _prev_q_t_s: Optional[float] = field(default=None, init=False)
    _prev_obs_joint_pos: Optional[np.ndarray] = field(default=None, init=False)
    _curr_obs_joint_pos: Optional[np.ndarray] = field(default=None, init=False)
    _prev_obs_joint_vel: Optional[np.ndarray] = field(default=None, init=False)
    _curr_obs_joint_vel: Optional[np.ndarray] = field(default=None, init=False)
    _prev_obs_joint_torque: Optional[np.ndarray] = field(default=None, init=False)
    _curr_obs_joint_torque: Optional[np.ndarray] = field(default=None, init=False)
    _log_obs: bool = field(default=False, init=False)
    _log_action: bool = field(default=False, init=False)
    _log_every_n: int = field(default=1, init=False)
    _log_path: Optional[Path] = field(default=None, init=False)
    _log_file: Optional[Any] = field(default=None, init=False)
    _log_writer: Optional[Any] = field(default=None, init=False)
    _log_step_idx: int = field(default=0, init=False)
    _hz_lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        self.obs_history = deque(maxlen=int(self.spec.history_len))
        self._last_policy_action = np.zeros(len(self.spec.action_keys), dtype=np.float32)

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
        _ = (ankle_action_abs_limit, clamp_ankle_to_true_limits)
        cfg = _load_config(Path(config_path))
        spec = infer_agent_spec(cfg)
        if not spec.policy_terms:
            raise ValueError("Missing policy observation terms in config.")
        q_ref = _extract_default_joint_pos_rad_from_cfg(cfg)
        if q_ref is None or q_ref.size < 12:
            raise ValueError("Missing init_state.joint_pos with 12 joints in config.")
        policy = PolicyWrapper.load(Path(policy_path), cfg)
        agent = cls(robot=robot, spec=spec, policy=policy)
        agent._default_joint_pos_rad = q_ref[:12].astype(np.float32, copy=True)
        if log_observation or log_action:
            resolved_log_path = Path(log_path) if log_path else Path("rl_agent_isolated_debug_log.csv")
            agent.configure_logging(
                log_path=resolved_log_path,
                log_observation=log_observation,
                log_action=log_action,
                log_every_n=max(1, int(log_every_n)),
            )
        return agent

    def set_command_twist(self, lin_x: float, lin_y: float, yaw_rate: float) -> None:
        self._command_twist = np.array([float(lin_x), float(lin_y), float(yaw_rate)], dtype=np.float32)

    def set_command_source(self, source: Optional[Any]) -> None:
        self._command_source = source

    def set_inference_hz(self, hz: float) -> None:
        val = max(1.0, float(hz))
        with self._hz_lock:
            self.spec.inference_hz = val

    def get_inference_hz(self) -> float:
        with self._hz_lock:
            return float(self.spec.inference_hz)

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
        self._log_file = self._log_path.open("w", newline="")
        self._log_writer = csv.writer(self._log_file)
        self._log_writer.writerow(["time_s", "step", "observation", "action_pre_scale"])
        self._log_file.flush()

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

    def _refresh_command_from_source(self) -> None:
        src = self._command_source
        if src is None:
            return
        getter = getattr(src, "get_command_twist", None)
        if getter is None or not callable(getter):
            return
        cmd = getter()
        if not isinstance(cmd, (list, tuple, np.ndarray)) or len(cmd) < 3:
            return
        self.set_command_twist(float(cmd[0]), float(cmd[1]), float(cmd[2]))

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.obs_history.clear()
        self._prev_q_rad = None
        self._prev_q_t_s = None
        self._prev_obs_joint_pos = None
        self._curr_obs_joint_pos = None
        self._prev_obs_joint_vel = np.zeros(12, dtype=np.float32)
        self._curr_obs_joint_vel = None
        self._prev_obs_joint_torque = np.zeros(12, dtype=np.float32)
        self._curr_obs_joint_torque = None
        self._last_policy_action = np.zeros(len(self.spec.action_keys), dtype=np.float32)
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
            "joint_vel_source": str(self.spec.joint_vel_source),
            "debug_zero_actions_obs": bool(self.spec.debug_zero_actions_obs),
            "obs_dim": int(self._last_obs.size) if self._last_obs is not None else 0,
            "action_dim": int(self._last_action.size) if self._last_action is not None else 0,
            "policy_terms": list(self.spec.policy_terms),
            "action_scale": float(self.spec.action_scale),
            "obs_term_scales": dict(self.spec.obs_term_scales),
        }

    def _policy_order_joint_state(self, snapshot: Dict[str, Any], key: str) -> np.ndarray:
        vals = np.asarray(snapshot.get(key, [0.0] * 12), dtype=np.float32).reshape(-1)
        if vals.size < 12:
            vals = np.pad(vals, (0, 12 - vals.size))
        vals = vals[:12]
        return vals[SNAPSHOT_TO_POLICY_JOINT_IDX]

    def _apply_obs_term_scale(self, term_name: str, values: np.ndarray) -> np.ndarray:
        out = np.asarray(values, dtype=np.float32).reshape(-1)
        scale = float(self.spec.obs_term_scales.get(term_name, 1.0))
        if np.isfinite(scale) and scale != 1.0:
            out = (out * scale).astype(np.float32, copy=False)
        return out

    def _term_observation_vector(self, snapshot: Dict[str, Any], term_name: str) -> np.ndarray:
        if term_name == "actions":
            if self.spec.debug_zero_actions_obs:
                return self._apply_obs_term_scale(term_name, np.zeros_like(self._last_policy_action, dtype=np.float32))
            return self._apply_obs_term_scale(term_name, self._last_policy_action.copy())
        if term_name == "command":
            return self._apply_obs_term_scale(term_name, self._command_twist.copy())
        if term_name == "base_ang_vel":
            imu = snapshot.get("imu") or {}
            gyro = imu.get("gyro_rads") if isinstance(imu, dict) else None
            if gyro is None and isinstance(imu, dict):
                gyro = imu.get("ang_vel_rad_s")
            if isinstance(gyro, (list, tuple)) and len(gyro) >= 3:
                return self._apply_obs_term_scale(
                    term_name,
                    np.asarray([float(gyro[0]), float(gyro[1]), float(gyro[2])], dtype=np.float32),
                )
            return self._apply_obs_term_scale(term_name, np.zeros(3, dtype=np.float32))
        if term_name == "base_lin_vel":
            imu = snapshot.get("imu") or {}
            lin_vel = imu.get("linear_velocity_mps") if isinstance(imu, dict) else None
            if lin_vel is None and isinstance(imu, dict):
                lin_vel = imu.get("lin_vel_m_s")
            if isinstance(lin_vel, (list, tuple)) and len(lin_vel) >= 3:
                return self._apply_obs_term_scale(
                    term_name,
                    np.asarray([float(lin_vel[0]), float(lin_vel[1]), float(lin_vel[2])], dtype=np.float32),
                )
            return self._apply_obs_term_scale(term_name, np.zeros(3, dtype=np.float32))
        if term_name == "projected_gravity":
            g_direct = snapshot.get("projected_gravity")
            if isinstance(g_direct, (list, tuple, np.ndarray)) and len(g_direct) >= 3:
                return self._apply_obs_term_scale(
                    term_name,
                    np.asarray([float(g_direct[0]), float(g_direct[1]), float(g_direct[2])], dtype=np.float32),
                )
            q = snapshot.get("orientation_quaternion_xyzw")
            if isinstance(q, (list, tuple)) and len(q) == 4:
                r = _quat_xyzw_to_rotmat(q)
                return self._apply_obs_term_scale(
                    term_name,
                    (r.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32),
                )
            return self._apply_obs_term_scale(term_name, np.array([0.0, 0.0, -1.0], dtype=np.float32))
        if term_name in ("joint_pos", "joint_vel"):
            q_deg = self._policy_order_joint_state(snapshot, "joint_state_deg")
            q_rad = np.deg2rad(q_deg)
            now_s = float(snapshot.get("time_s", time.time()))
            qd_snap = self._policy_order_joint_state(snapshot, "joint_velocity_rad_s")
            if self._prev_q_rad is not None and self._prev_q_t_s is not None and now_s > self._prev_q_t_s:
                dt = max(1e-4, now_s - self._prev_q_t_s)
                qd_fd = (q_rad - self._prev_q_rad) / dt
            else:
                qd_fd = np.zeros_like(q_rad, dtype=np.float32)
            if term_name == "joint_pos":
                if self._default_joint_pos_rad is None:
                    return np.zeros(12, dtype=np.float32)
                qpos_now = (q_rad - self._default_joint_pos_rad).astype(np.float32, copy=False)
                qpos_now = self._apply_obs_term_scale(term_name, qpos_now)
                self._curr_obs_joint_pos = qpos_now.copy()
                return qpos_now
            if self.spec.joint_vel_source == "finite_diff":
                qd_now = qd_fd.astype(np.float32, copy=False)
            else:
                qd_now = (0.5 * qd_snap + 0.5 * qd_fd).astype(np.float32, copy=False)
            qd_now = self._apply_obs_term_scale(term_name, qd_now)
            self._curr_obs_joint_vel = qd_now.copy()
            return qd_now
        if term_name in JOINT_TORQUE_TERM_NAMES:
            tau_now = self._policy_order_joint_state(snapshot, "joint_torque_nm").astype(np.float32, copy=False)
            tau_now = self._apply_obs_term_scale(term_name, tau_now)
            self._curr_obs_joint_torque = tau_now.copy()
            return tau_now
        return np.zeros(0, dtype=np.float32)

    def _build_obs_now(self, snapshot: Dict[str, Any]) -> np.ndarray:
        self._curr_obs_joint_pos = None
        self._curr_obs_joint_vel = None
        self._curr_obs_joint_torque = None
        parts = [self._term_observation_vector(snapshot, name) for name in self.spec.policy_terms]
        if parts and not self.spec.term_dims:
            self.spec.term_dims = [int(p.size) for p in parts]
        obs = np.concatenate(parts, axis=0).astype(np.float32, copy=False) if parts else np.zeros(0, dtype=np.float32)

        q_deg = self._policy_order_joint_state(snapshot, "joint_state_deg")
        self._prev_q_rad = np.deg2rad(q_deg).astype(np.float32, copy=False)
        self._prev_q_t_s = float(snapshot.get("time_s", time.time()))
        if self._curr_obs_joint_pos is not None:
            self._prev_obs_joint_pos = self._curr_obs_joint_pos.copy()
        if self._curr_obs_joint_vel is not None:
            self._prev_obs_joint_vel = self._curr_obs_joint_vel.copy()
        if self._curr_obs_joint_torque is not None:
            self._prev_obs_joint_torque = self._curr_obs_joint_torque.copy()
        return obs

    def _build_history_obs(self, obs_now: np.ndarray) -> np.ndarray:
        self.obs_history.append(obs_now)
        if len(self.obs_history) < self.spec.history_len:
            first = self.obs_history[0]
            while len(self.obs_history) < self.spec.history_len:
                self.obs_history.appendleft(first.copy())

        if self.spec.obs_layout == "isaaclab" and self.spec.term_dims:
            # Term-major: [term0_{t-H+1..t}, term1_{t-H+1..t}, ...]. This matches
            # Isaac Lab's ObservationManager output when flatten_history_dim=True
            # is applied at each term and concatenate_terms=True at the group
            # (per observation_manager.py:reshape(..., -1) then torch.cat along
            # the feature axis).
            arr = np.stack(list(self.obs_history), axis=0)  # (H, D)
            out_parts: List[np.ndarray] = []
            off = 0
            for d in self.spec.term_dims:
                slab = arr[:, off : off + d]  # (H, d)
                # Row-major reshape gives [t0_j0..t0_{d-1}, t1_j0..t1_{d-1}, ...]
                # — time slowest, component fastest within term.
                out_parts.append(slab.reshape(-1))
                off += d
            return np.concatenate(out_parts, axis=0).astype(np.float32, copy=False)

        # mjlab / canonical: time-major [obs_{t-H+1}, ..., obs_t].
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
        act = np.nan_to_num(act, nan=0.0, posinf=0.0, neginf=0.0)
        n = min(len(self.spec.action_keys), int(act.size))
        if n <= 0:
            return

        if self._last_policy_action.size != len(self.spec.action_keys):
            self._last_policy_action = np.zeros(len(self.spec.action_keys), dtype=np.float32)
        self._last_policy_action[:] = 0.0
        self._last_policy_action[:n] = act[:n]

        if self._default_joint_pos_rad is None:
            return
        q_cmd_rad = self._default_joint_pos_rad.copy()
        key_to_idx = {k: i for i, k in enumerate(POLICY_ACTION_KEYS)}
        for i, key in enumerate(self.spec.action_keys[:n]):
            idx = key_to_idx.get(key)
            if idx is None:
                continue
            scale_i = float(self.spec.action_scales_rad[i]) if i < len(self.spec.action_scales_rad) else 1.0
            bias_i = float(self.spec.encoder_bias_rad[i]) if i < len(self.spec.encoder_bias_rad) else 0.0
            q_cmd_rad[idx] = (
                self._default_joint_pos_rad[idx]
                + float(act[i]) * scale_i * float(self.spec.action_scale)
                - bias_i
            )

        q_cmd_deg = np.rad2deg(q_cmd_rad)
        left, right = _action_array_policy_order_to_command(q_cmd_deg)
        self.robot.set_action(left=left, right=right)

    def _run_loop(self) -> None:
        next_tick = time.perf_counter()
        while self._running:
            try:
                self._refresh_command_from_source()
                snapshot = self.robot.get_combined_state_snapshot(include_joint_state=True)
                obs_now = self._build_obs_now(snapshot)
                obs_hist = self._build_history_obs(obs_now)
                obs_in = self._adapt_obs_dim_for_policy(obs_hist)
                action = self.policy.infer(obs_in)
                self._last_obs = obs_in
                self._last_action = action
                self._maybe_log_step(obs_in, action)
                self._apply_action(action)
            except Exception as exc:
                print(f"[RLAgentIsolated][WARN] loop exception: {type(exc).__name__}: {exc}")

            period = 1.0 / self.get_inference_hz()
            next_tick += period
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.perf_counter()
