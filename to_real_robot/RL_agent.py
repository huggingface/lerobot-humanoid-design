from __future__ import annotations

import csv
import importlib
import json
import pickle
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from bipedal_robot import BipedalRobotController
except Exception:  # pragma: no cover - allow sim-only usage
    BipedalRobotController = Any  # type: ignore

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

PREFERRED_POLICY_ACTION_ORDER_12 = [
    "right.hipz",
    "left.hipz",
    "right.hipx",
    "left.hipx",
    "right.hipy",
    "left.hipy",
    "right.knee",
    "left.knee",
    "right.ankle_pitch",
    "left.ankle_pitch",
    "right.ankle_roll",
    "left.ankle_roll",
]

# Newer MJLab exports embed joint observations/actions in block order:
# [right(6), left(6)].
POLICY_BLOCK_ACTION_KEYS_12 = [
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
SNAPSHOT_TO_POLICY_JOINT_IDX = np.array([6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5], dtype=np.int64)

# Debug sign overrides applied on top of action scales.
ACTION_SCALE_SIGN_OVERRIDES_BY_KEY = {
    "left.hipy": -1.0,
    "right.hipy": -1.0,
    "left.ankle_pitch": -1.0,
    "right.ankle_pitch": -1.0,
    "left.ankle_roll": -1.0,
    "right.ankle_roll": -1.0,
    "left.hipx": -1.0,
    "right.hipx": -1.0,
}

JOINT_TORQUE_TERM_NAMES = ("joint_torque", "joint_torques", "joint_effort", "joint_efforts")

@dataclass
class AgentSpec:
    action_keys: List[str]
    history_len: int = 1
    inference_hz: float = 50.0
    action_scale: float = 1.0
    policy_terms: List[str] = field(default_factory=list)
    action_scales_rad: List[float] = field(default_factory=list)
    obs_term_scales: Dict[str, float] = field(default_factory=dict)
    joint_vel_source: str = "auto"  # auto | finite_diff
    use_policy_joint_order: bool = False


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


def _canonicalize_policy_terms(policy_terms: List[str]) -> List[str]:
    """Normalize known MJLab velocity observation term layouts.

    Some exported YAML configs reorder dict keys compared to the runtime
    insertion order used when the policy was trained. For known velocity-task
    term families, force the canonical runtime order to match training,
    including the optional torque observation block.
    """
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
        sign = float(ACTION_SCALE_SIGN_OVERRIDES_BY_KEY.get(key, 1.0))
        out.append(_joint_scale(joint) * sign)
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


def _extract_name_list_from_onnx_bytes(
    onnx_path: Path,
    *,
    normalizer: Optional[Any] = None,
) -> List[str]:
    """Best-effort extraction of embedded comma-separated metadata from ONNX exports."""
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
        if normalizer is not None:
            norm_tokens = [normalizer(tok) for tok in raw_tokens]
            tokens = [tok for tok in norm_tokens if isinstance(tok, str)]
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


def _uses_block_policy_joint_order(action_keys: Sequence[str]) -> bool:
    return list(action_keys) == POLICY_BLOCK_ACTION_KEYS_12


def _reshape_joint_vector_12(values: Any, *, dtype: Any = np.float32) -> np.ndarray:
    arr = np.asarray(values if values is not None else [0.0] * 12, dtype=dtype).reshape(-1)
    if arr.size < 12:
        arr = np.pad(arr, (0, 12 - arr.size))
    return arr[:12]


def _policy_order_joint_vector(values: Any, *, use_policy_joint_order: bool, dtype: Any = np.float32) -> np.ndarray:
    arr = _reshape_joint_vector_12(values, dtype=dtype)
    if use_policy_joint_order:
        arr = arr[SNAPSHOT_TO_POLICY_JOINT_IDX]
    return arr


def _dict_joint_pos_to_q_rad(joint_pos: Dict[str, Any]) -> Optional[np.ndarray]:
    key_to_idx = {
        "hipz_left": 0,
        "hipx_left": 1,
        "hipy_left": 2,
        "knee_left": 3,
        "ankley_left": 4,
        "anklex_left": 5,
        "hipz_right": 6,
        "hipx_right": 7,
        "hipy_right": 8,
        "knee_right": 9,
        "ankley_right": 10,
        "anklex_right": 11,
    }
    q = np.zeros(12, dtype=np.float32)
    for name, idx in key_to_idx.items():
        if name not in joint_pos:
            return None
        try:
            q[idx] = float(joint_pos[name])
        except Exception:
            return None
    return q


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
    return _dict_joint_pos_to_q_rad(joint_pos)


def infer_agent_spec(cfg: Dict[str, Any]) -> AgentSpec:
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

    action_keys: List[str] = []
    for key in (_flatten_strings(action_raw) if action_raw is not None else []):
        key_s = str(key).strip()
        if not key_s or ".*" in key_s:
            continue
        norm = _normalize_joint_name(key_s)
        if norm is not None:
            action_keys.append(norm)
    if not action_keys:
        # This policy family exports actions in right/left-interleaved order.
        action_keys = list(PREFERRED_POLICY_ACTION_ORDER_12)

    policy_terms = _canonicalize_policy_terms(_extract_policy_terms(cfg))

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
    # Backward compatibility aliases.
    if joint_vel_source in ("snapshot", "joint_state", "joint_state_fd_fallback"):
        joint_vel_source = "auto"
    if joint_vel_source in ("fd", "finite_difference", "finite_differences"):
        joint_vel_source = "finite_diff"
    if joint_vel_source not in ("auto", "finite_diff"):
        joint_vel_source = "auto"

    action_scales = _extract_action_scales(cfg, action_keys)
    obs_term_scales = _extract_observation_term_scales(cfg, policy_terms)

    return AgentSpec(
        action_keys=list(action_keys),
        history_len=max(1, history_len),
        inference_hz=max(1.0, inference_hz),
        action_scale=action_scale,
        policy_terms=policy_terms,
        action_scales_rad=action_scales,
        obs_term_scales=obs_term_scales,
        joint_vel_source=joint_vel_source,
        use_policy_joint_order=_uses_block_policy_joint_order(action_keys),
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

    def select_onnx_action_output(self, action_dim: int) -> None:
        if self._onnx_session is None:
            return
        outputs = list(self._onnx_session.get_outputs())
        if not outputs:
            return

        def _last_dim(shape: Any) -> Optional[int]:
            if not isinstance(shape, list) or not shape:
                return None
            last = shape[-1]
            return int(last) if isinstance(last, int) else None

        exact = [o for o in outputs if _last_dim(o.shape) == int(action_dim)]
        if exact:
            chosen = exact[0]
            self._onnx_output_name = chosen.name
            print(f"[RLAgent] ONNX output selected by dim=={action_dim}: {chosen.name}")
            return

        fixed = [(o, _last_dim(o.shape)) for o in outputs]
        fixed = [(o, d) for (o, d) in fixed if d is not None and d > 0]
        if fixed:
            # Fallback: choose the widest output; action heads are typically wider than scalar value heads.
            chosen = max(fixed, key=lambda x: int(x[1]))[0]
            self._onnx_output_name = chosen.name
            print(
                f"[RLAgent][WARN] no ONNX output with dim=={action_dim}; "
                f"fallback to widest output '{chosen.name}'"
            )

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
    _prev_obs_joint_pos: Optional[np.ndarray] = field(default=None, init=False)
    _curr_obs_joint_pos: Optional[np.ndarray] = field(default=None, init=False)
    _prev_obs_joint_vel: Optional[np.ndarray] = field(default=None, init=False)
    _curr_obs_joint_vel: Optional[np.ndarray] = field(default=None, init=False)
    _prev_obs_joint_torque: Optional[np.ndarray] = field(default=None, init=False)
    _curr_obs_joint_torque: Optional[np.ndarray] = field(default=None, init=False)
    _command_twist: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32), init=False)
    _command_source: Optional[Any] = field(default=None, init=False)
    _log_obs: bool = field(default=False, init=False)
    _log_action: bool = field(default=False, init=False)
    _log_every_n: int = field(default=1, init=False)
    _log_path: Optional[Path] = field(default=None, init=False)
    _log_file: Optional[Any] = field(default=None, init=False)
    _log_writer: Optional[Any] = field(default=None, init=False)
    _log_step_idx: int = field(default=0, init=False)
    _last_hold_warn_t_s: float = field(default=0.0, init=False)
    _last_action_debug_t_s: float = field(default=0.0, init=False)
    _last_safety_warn_t_s: float = field(default=0.0, init=False)
    _action_abs_limit_for_debug: float = field(default=1000000.0, init=False)
    _sim_decimation_steps: Optional[int] = field(default=None, init=False)
    _next_inference_sim_step: Optional[int] = field(default=None, init=False)
    _last_seen_sim_step: Optional[int] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.obs_history = deque(maxlen=int(self.spec.history_len))

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
            raise ValueError("Expected policy observation terms in config at env_cfg.value.observations.policy.terms")
        q_ref = _extract_default_joint_pos_rad_from_cfg(cfg)
        if q_ref is None or q_ref.size < 12:
            raise ValueError("Expected env_cfg.value.scene.entities.robot.init_state.joint_pos with 12 joints")
        ppath = Path(policy_path)
        onnx_action_keys = _extract_onnx_joint_action_keys(ppath)
        if len(onnx_action_keys) == 12:
            spec.action_keys = list(onnx_action_keys)
            spec.action_scales_rad = _extract_action_scales(cfg, spec.action_keys)
            spec.use_policy_joint_order = _uses_block_policy_joint_order(spec.action_keys)
        policy = PolicyWrapper.load(ppath, cfg)
        policy.select_onnx_action_output(len(spec.action_keys))
        agent = cls(robot=robot, spec=spec, policy=policy)
        agent._default_joint_pos_rad = q_ref[:12].astype(np.float32, copy=True)
        print(
            f"[RLAgent] loaded spec: terms={spec.policy_terms}, "
            f"action_keys={spec.action_keys}, action_scales_rad={spec.action_scales_rad}, "
            f"base_action_scale={spec.action_scale}, joint_vel_source={spec.joint_vel_source}, "
            f"use_policy_joint_order={spec.use_policy_joint_order}, obs_term_scales={spec.obs_term_scales}"
        )
        if log_observation or log_action:
            resolved_log_path = Path(log_path) if log_path else Path("rl_agent_debug_log.csv")
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
        self._prev_obs_joint_vel = None
        self._curr_obs_joint_vel = None
        self._prev_obs_joint_torque = None
        self._curr_obs_joint_torque = None
        self._sim_decimation_steps = None
        self._next_inference_sim_step = None
        self._last_seen_sim_step = None
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
            "obs_dim": int(self._last_obs.size) if self._last_obs is not None else 0,
            "action_dim": int(self._last_action.size) if self._last_action is not None else 0,
            "policy_terms": list(self.spec.policy_terms),
            "use_policy_joint_order": bool(self.spec.use_policy_joint_order),
            "obs_term_scales": dict(self.spec.obs_term_scales),
        }

    def _policy_order_joint_state(self, snapshot: Dict[str, Any], key: str) -> np.ndarray:
        return _policy_order_joint_vector(
            snapshot.get(key, [0.0] * 12),
            use_policy_joint_order=bool(self.spec.use_policy_joint_order),
            dtype=np.float32,
        )

    def _policy_order_default_joint_pos_rad(self) -> np.ndarray:
        if self._default_joint_pos_rad is None:
            raise RuntimeError("default joint reference pose not initialized")
        return _policy_order_joint_vector(
            self._default_joint_pos_rad,
            use_policy_joint_order=bool(self.spec.use_policy_joint_order),
            dtype=np.float32,
        )

    def _apply_obs_term_scale(self, term_name: str, values: np.ndarray) -> np.ndarray:
        out = np.asarray(values, dtype=np.float32).reshape(-1)
        scale = float(self.spec.obs_term_scales.get(term_name, 1.0))
        if np.isfinite(scale) and scale != 1.0:
            out = (out * scale).astype(np.float32, copy=False)
        return out

    def _term_observation_vector(self, snapshot: Dict[str, Any], term_name: str) -> np.ndarray:
        q_deg = self._policy_order_joint_state(snapshot, "joint_state_deg")
        q_rad = np.deg2rad(q_deg)

        qd_snap = self._policy_order_joint_state(snapshot, "joint_velocity_rad_s")
        now_s = float(snapshot.get("time_s", time.time()))
        if self._prev_q_rad is None or self._prev_q_t_s is None or now_s <= self._prev_q_t_s:
            qd_fd = np.zeros_like(q_rad)
        else:
            dt = max(1e-4, now_s - self._prev_q_t_s)
            qd_fd = (q_rad - self._prev_q_rad) / dt

        if self.spec.joint_vel_source in ("finite_diff", "finite_difference", "finite_differences", "fd"):
            qd_rad_s = qd_fd
        else:
            qd_rad_s = qd_snap

        if term_name == "actions":
            return self._apply_obs_term_scale(term_name, self._last_policy_action.copy())
        if term_name == "base_ang_vel":
            imu = snapshot.get("imu") or {}
            gyro = imu.get("gyro_rads") if isinstance(imu, dict) else None
            if gyro is None and isinstance(imu, dict):
                gyro = imu.get("ang_vel_rad_s")
            if isinstance(gyro, (list, tuple)) and len(gyro) >= 3:
                return self._apply_obs_term_scale(
                    term_name,
                    np.array([float(gyro[0]), float(gyro[1]), float(gyro[2])], dtype=np.float32),
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
                    np.array([float(lin_vel[0]), float(lin_vel[1]), float(lin_vel[2])], dtype=np.float32),
                )
            return self._apply_obs_term_scale(term_name, np.zeros(3, dtype=np.float32))
        if term_name == "command":
            return self._apply_obs_term_scale(term_name, self._command_twist.copy())
        if term_name == "joint_pos":
            qpos_now = (q_rad - self._policy_order_default_joint_pos_rad()).astype(np.float32, copy=False)
            # Observation-only convention fix (do not affect action reference).
            qpos_now = qpos_now.copy()
            qpos_now[[2, 4, 8, 10]] *= -1.0
            qpos_now = self._apply_obs_term_scale(term_name, qpos_now)
            self._curr_obs_joint_pos = qpos_now.copy()
            if self._prev_obs_joint_pos is None:
                return qpos_now
            return self._prev_obs_joint_pos.astype(np.float32, copy=False)
        if term_name == "joint_vel":
            qd_now = self._apply_obs_term_scale(term_name, qd_rad_s.astype(np.float32, copy=False))
            self._curr_obs_joint_vel = qd_now.copy()
            if self._prev_obs_joint_vel is None:
                return np.zeros_like(qd_now, dtype=np.float32)
            return self._prev_obs_joint_vel.astype(np.float32, copy=False)
        if term_name in JOINT_TORQUE_TERM_NAMES:
            tau_now = self._policy_order_joint_state(snapshot, "joint_torque_nm")
            tau_now = self._apply_obs_term_scale(term_name, tau_now)
            self._curr_obs_joint_torque = tau_now.copy()
            if self._prev_obs_joint_torque is None:
                return np.zeros_like(tau_now, dtype=np.float32)
            return self._prev_obs_joint_torque.astype(np.float32, copy=False)
        if term_name == "projected_gravity":
            q = snapshot.get("orientation_quaternion_xyzw")
            if isinstance(q, (list, tuple)) and len(q) == 4:
                r = _quat_xyzw_to_rotmat(q)
                g_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
                return self._apply_obs_term_scale(term_name, (r.T @ g_world).astype(np.float32))
            return self._apply_obs_term_scale(term_name, np.array([0.0, 0.0, -1.0], dtype=np.float32))

        return np.zeros(0, dtype=np.float32)

    def _build_obs_now(self, snapshot: Optional[Dict[str, Any]] = None) -> np.ndarray:
        if snapshot is None:
            snapshot = self.robot.get_combined_state_snapshot(include_joint_state=True)

        self._curr_obs_joint_pos = None
        self._curr_obs_joint_vel = None
        self._curr_obs_joint_torque = None
        obs_map: Dict[str, np.ndarray] = {}
        for term_name in self.spec.policy_terms:
            obs_map[term_name] = self._term_observation_vector(snapshot, term_name)
        parts = [obs_map[t] for t in self.spec.policy_terms]
        obs = np.concatenate(parts, axis=0) if parts else np.zeros(0, dtype=np.float32)

        q_deg = self._policy_order_joint_state(snapshot, "joint_state_deg")
        q_rad = np.deg2rad(q_deg)
        self._prev_q_rad = q_rad
        self._prev_q_t_s = float(snapshot.get("time_s", time.time()))
        if self._curr_obs_joint_pos is not None:
            self._prev_obs_joint_pos = self._curr_obs_joint_pos.copy()
        if self._curr_obs_joint_vel is not None:
            self._prev_obs_joint_vel = self._curr_obs_joint_vel.copy()
        if self._curr_obs_joint_torque is not None:
            self._prev_obs_joint_torque = self._curr_obs_joint_torque.copy()
        return obs.astype(np.float32, copy=False)

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

        # Safety guard for debug runs: avoid NaN/Inf commands and unbounded action magnitudes.
        if not np.all(np.isfinite(act[:n])):
            act = np.nan_to_num(act, nan=0.0, posinf=0.0, neginf=0.0)
            now = time.time()
            if now - float(self._last_safety_warn_t_s) > 1.0:
                print("[RLAgent][WARN] non-finite policy action detected; replaced with zeros.")
                self._last_safety_warn_t_s = now
        lim = float(self._action_abs_limit_for_debug)
        act[:n] = np.clip(act[:n], -lim, lim)

        if self._last_policy_action.size != n:
            self._last_policy_action = np.zeros(n, dtype=np.float32)
        self._last_policy_action[:] = act[:n]

        snapshot = self.robot.get_combined_state_snapshot(include_joint_state=True)
        if bool(snapshot.get("post_reset_hold_active", False)):
            now = time.time()
            if now - float(self._last_hold_warn_t_s) > 1.0:
                rem = float(snapshot.get("post_reset_hold_remaining_s", 0.0))
                print(f"[RLAgent] sim post-reset hold active ({rem:.3f}s remaining): commands temporarily ignored by sim.")
                self._last_hold_warn_t_s = now
        q_cur_deg = np.asarray(snapshot.get("joint_state_deg", [0.0] * 12), dtype=np.float32).reshape(-1)
        if q_cur_deg.size < 12:
            q_cur_deg = np.pad(q_cur_deg, (0, 12 - q_cur_deg.size))

        q_cmd_rad = self._default_joint_pos_rad.copy()

        default_keys = _default_action_keys()
        key_to_idx = {k: i for i, k in enumerate(default_keys)}
        action_by_key: Dict[str, float] = {}
        for i in range(n):
            action_by_key[self.spec.action_keys[i]] = float(act[i])
        for i, key in enumerate(self.spec.action_keys[:n]):
            if key not in key_to_idx:
                continue
            idx = key_to_idx[key]
            scale_i = float(self.spec.action_scales_rad[i]) if i < len(self.spec.action_scales_rad) else 1.0
            value_rad = float(action_by_key[key]) * scale_i * float(self.spec.action_scale)
            q_cmd_rad[idx] = self._default_joint_pos_rad[idx] + value_rad

        q_cmd_deg = np.rad2deg(q_cmd_rad)
        left, right = _joint_array_to_command(q_cmd_deg)
        applied = self.robot.set_action(left=left, right=right)
        now = time.time()
        if now - float(self._last_action_debug_t_s) > 0.5:
            l_hipy = float(left["hipy"])
            l_knee = float(left["knee"])
            r_hipy = float(right["hipy"])
            r_knee = float(right["knee"])
            act_abs_max = float(np.max(np.abs(act[:n]))) if n > 0 else 0.0
            print(
                "[RLAgent][DEBUG] apply_action "
                f"n={n} act_abs_max={act_abs_max:.4f} "
                f"L(hipy={l_hipy:.2f},knee={l_knee:.2f}) "
                f"R(hipy={r_hipy:.2f},knee={r_knee:.2f}) "
                f"sim_cmd_mid1={float(applied.get(1, 0.0)):.2f}"
            )
            self._last_action_debug_t_s = now

    def _run_loop(self) -> None:
        period = 1.0 / float(self.spec.inference_hz)
        next_tick = time.perf_counter()
        while self._running:
            try:
                snapshot = self.robot.get_combined_state_snapshot(include_joint_state=True)

                using_sim_sync = False
                sim_step = snapshot.get("sim_step_count", None)
                sim_dt = snapshot.get("sim_timestep_s", None)
                if sim_step is not None and sim_dt is not None:
                    try:
                        sim_step_i = int(sim_step)
                        sim_dt_f = float(sim_dt)
                        if sim_dt_f > 0.0:
                            using_sim_sync = True
                            decim = max(1, int(round((1.0 / float(self.spec.inference_hz)) / sim_dt_f)))
                            self._sim_decimation_steps = decim
                            # Re-sync only on initialization or when sim step counter moved backward (reset).
                            if self._next_inference_sim_step is None:
                                self._next_inference_sim_step = sim_step_i
                            elif (
                                self._last_seen_sim_step is not None
                                and sim_step_i < int(self._last_seen_sim_step)
                            ):
                                self._next_inference_sim_step = sim_step_i
                            self._last_seen_sim_step = sim_step_i
                            if sim_step_i < int(self._next_inference_sim_step):
                                time.sleep(0.0005)
                                continue
                            self._next_inference_sim_step = int(self._next_inference_sim_step) + decim
                            # Catch up if we were delayed.
                            while sim_step_i >= int(self._next_inference_sim_step):
                                    self._next_inference_sim_step += decim
                    except Exception:
                        using_sim_sync = False
                        self._sim_decimation_steps = None
                else:
                    self._sim_decimation_steps = None

                self._refresh_command_from_source()
                obs_now = self._build_obs_now(snapshot=snapshot)
                obs_hist = self._build_history_obs(obs_now)
                obs_in = self._adapt_obs_dim_for_policy(obs_hist)
                action = self.policy.infer(obs_in)

                self._last_obs = obs_in
                self._last_action = action
                self._maybe_log_step(obs_in, action)
                self._apply_action(action)
            except Exception as exc:
                print(f"[RLAgent][WARN] loop exception: {type(exc).__name__}: {exc}")

            if using_sim_sync:
                time.sleep(0.0005)
                continue
            next_tick += period
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.perf_counter()
