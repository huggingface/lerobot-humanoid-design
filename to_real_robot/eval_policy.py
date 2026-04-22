from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sim_robot import SimBipedalRobotController
from RL_agent_isolated import RLAgent


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_POLICY_ROOT = REPO_ROOT / "RL_policy"


# AGILE (Isaac Lab) observation term names → RL_agent_isolated canonical names.
_AGILE_TERM_RENAME = {
    "velocity_commands": "command",
    "controlled_joint_pos": "joint_pos",
    "controlled_joint_vel": "joint_vel",
}
# Keys on observations.policy that sit alongside the terms (i.e. not themselves terms).
_POLICY_META_KEYS = {
    "concatenate_terms",
    "concatenate_dim",
    "enable_corruption",
    "history_length",
    "flatten_history_dim",
}


def _sanitize_for_safe_yaml(value):
    """Recursively convert an object tree (from yaml.unsafe_load) into primitives
    that yaml.safe_dump accepts: tuples → lists, drops slice/type objects."""
    if isinstance(value, dict):
        return {str(k): _sanitize_for_safe_yaml(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_safe_yaml(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)) or value is None:
        return value
    # slice, type, function — unrepresentable; drop.
    return None


def _adapt_agile_env_to_mjlab_config(env_yaml_path: Path, config_yaml_path: Path) -> None:
    """Translate an AGILE-exported env.yaml into the mjlab-convention config.yaml
    that to_real_robot/RL_agent_isolated.py expects.

    AGILE puts obs terms directly under observations.policy.<term_name> with names
    like velocity_commands / controlled_joint_{pos,vel}, and keeps robot config at
    scene.robot. RL_agent_isolated looks for observations.policy.terms.<term_name>
    with canonical names (command, joint_pos, joint_vel) and scene.entities.robot.
    """
    import yaml  # type: ignore

    with open(env_yaml_path) as f:
        env = yaml.unsafe_load(f)
    if not isinstance(env, dict):
        raise ValueError(f"env.yaml did not parse into a dict: {env_yaml_path}")

    obs = env.get("observations")
    if isinstance(obs, dict):
        for group_name in ("policy", "critic"):
            group = obs.get(group_name)
            if not isinstance(group, dict):
                continue
            terms: dict = {}
            meta: dict = {}
            for k, v in group.items():
                if k in _POLICY_META_KEYS:
                    meta[k] = v
                else:
                    terms[_AGILE_TERM_RENAME.get(k, k)] = v
            rebuilt = dict(meta)
            rebuilt["terms"] = terms
            obs[group_name] = rebuilt

    scene = env.get("scene")
    if isinstance(scene, dict) and "robot" in scene and "entities" not in scene:
        scene["entities"] = {"robot": scene.pop("robot")}

    # RL_agent_isolated's history_len lookup only matches env_cfg.* paths or a
    # top-level `history_len`; mirror the policy history_length to the top level.
    policy_group = (
        env.get("observations", {}).get("policy")
        if isinstance(env.get("observations"), dict)
        else None
    )
    if isinstance(policy_group, dict) and "history_length" in policy_group:
        env.setdefault("history_len", policy_group["history_length"])

    clean = _sanitize_for_safe_yaml(env)
    with open(config_yaml_path, "w") as f:
        yaml.safe_dump(clean, f, sort_keys=False)


@dataclass
class EvalConfig:
    policy_name: str
    command_vx: float = 0.3
    command_vy: float = 0.0
    command_yaw: float = 0.0
    duration_s: float = 10.0
    fall_height_m: float = 0.35
    flip_angle_rad: float = 1.2217
    warmup_s: float = 1.0
    control_hz: float = 200.0
    sample_hz: float = 50.0
    seed: int = 0
    init_joint_noise_deg: float = 2.0
    init_base_yaw_noise_deg: float = 5.0
    obs_joint_pos_noise_deg: float = 0.0
    obs_joint_vel_noise_rad_s: float = 0.0
    obs_gyro_noise_rad_s: float = 0.0
    obs_lin_vel_noise_mps: float = 0.0
    obs_quat_tilt_noise_deg: float = 0.0
    action_latency_ms: float = 0.0
    action_noise_deg: float = 0.0
    push_interval_s: float = 0.0
    push_force_n: float = 0.0
    lin_vel_mode: str = "sim"  # sim | zero | leaky
    leaky_lin_vel_tau_s: float = 1.5
    gyro_bias_drift_std: float = 0.0  # rad/s/sqrt(s), Ornstein-Uhlenbeck diffusion
    gyro_bias_tau_s: float = 100.0  # mean-reversion time constant
    tilt_bias_drift_deg_per_sqrt_s: float = 0.0  # orientation drift injected into projected_gravity
    tilt_bias_tau_s: float = 200.0
    obs_dropout_prob: float = 0.0  # probability of freezing obs for one step


@dataclass
class EpisodeMetrics:
    policy_name: str
    command_vx: float
    seed: int
    duration_s: float
    survived: bool
    survival_time_s: float
    fall_reason: str
    distance_x: float
    distance_y: float
    mean_base_height: float
    std_base_height: float
    mean_abs_pitch_deg: float
    mean_abs_roll_deg: float
    mean_measured_vx: float
    tracking_err_vx: float
    num_samples: int


class ConstantCommandSource:
    def __init__(self, lin_x: float = 0.0, lin_y: float = 0.0, yaw_rate: float = 0.0):
        self._cmd = (float(lin_x), float(lin_y), float(yaw_rate))

    def get_command_twist(self):
        return self._cmd


class PerturbedRobot:
    """Duck-typed proxy: injects observation noise and action latency/noise before forwarding.

    RLAgent only calls get_combined_state_snapshot() and set_action(); everything else
    falls through to the wrapped robot via __getattr__.
    """

    def __init__(
        self,
        robot,
        *,
        obs_joint_pos_noise_deg: float = 0.0,
        obs_joint_vel_noise_rad_s: float = 0.0,
        obs_gyro_noise_rad_s: float = 0.0,
        obs_lin_vel_noise_mps: float = 0.0,
        obs_quat_tilt_noise_deg: float = 0.0,
        action_latency_ms: float = 0.0,
        action_noise_deg: float = 0.0,
        lin_vel_mode: str = "sim",
        leaky_lin_vel_tau_s: float = 1.5,
        gyro_bias_drift_std: float = 0.0,
        gyro_bias_tau_s: float = 100.0,
        tilt_bias_drift_deg_per_sqrt_s: float = 0.0,
        tilt_bias_tau_s: float = 200.0,
        obs_dropout_prob: float = 0.0,
        seed: int = 0,
    ):
        self._robot = robot
        self._rng = np.random.default_rng(int(seed))
        self._jp_sig = float(obs_joint_pos_noise_deg)
        self._jv_sig = float(obs_joint_vel_noise_rad_s)
        self._gyro_sig = float(obs_gyro_noise_rad_s)
        self._lv_sig = float(obs_lin_vel_noise_mps)
        self._tilt_sig_deg = float(obs_quat_tilt_noise_deg)
        self._act_latency_s = max(0.0, float(action_latency_ms)) / 1000.0
        self._act_noise_deg = float(action_noise_deg)
        mode = str(lin_vel_mode).lower()
        if mode not in ("sim", "zero", "leaky"):
            raise ValueError(f"lin_vel_mode must be sim|zero|leaky, got {lin_vel_mode!r}")
        self._lin_vel_mode = mode
        self._leaky_tau_s = max(0.05, float(leaky_lin_vel_tau_s))
        self._leaky_v_body = np.zeros(3, dtype=float)
        self._leaky_prev_t_s: Optional[float] = None
        # Ornstein-Uhlenbeck gyro bias drift (3-axis)
        self._gyro_bias_drift_std = float(gyro_bias_drift_std)
        self._gyro_bias_tau_s = max(1.0, float(gyro_bias_tau_s))
        self._gyro_bias = np.zeros(3, dtype=float)
        # Orientation (tilt) bias drift — applied as a slowly-wandering rotation on quaternion
        self._tilt_bias_drift_rad = float(np.deg2rad(max(0.0, tilt_bias_drift_deg_per_sqrt_s)))
        self._tilt_bias_tau_s = max(1.0, float(tilt_bias_tau_s))
        self._tilt_bias_axis_angle = np.zeros(3, dtype=float)  # axis-angle representation
        # Obs dropout
        self._obs_dropout_prob = float(np.clip(obs_dropout_prob, 0.0, 1.0))
        self._last_snap: Optional[dict] = None
        self._drift_prev_t_s: Optional[float] = None

    def __getattr__(self, name):
        return getattr(self._robot, name)

    def _step_ou(self, state: np.ndarray, dt: float, sigma: float, tau: float) -> np.ndarray:
        """Advance an Ornstein-Uhlenbeck process one step."""
        decay = float(np.exp(-dt / tau))
        diffusion = sigma * float(np.sqrt(dt))
        return state * decay + self._rng.normal(0.0, diffusion, size=state.shape)

    def _step_drift(self, now_s: float) -> None:
        prev = self._drift_prev_t_s
        self._drift_prev_t_s = now_s
        if prev is None or now_s <= prev:
            return
        dt = min(0.1, now_s - prev)
        if self._gyro_bias_drift_std > 0.0:
            self._gyro_bias = self._step_ou(
                self._gyro_bias, dt, self._gyro_bias_drift_std, self._gyro_bias_tau_s
            )
        if self._tilt_bias_drift_rad > 0.0:
            self._tilt_bias_axis_angle = self._step_ou(
                self._tilt_bias_axis_angle, dt, self._tilt_bias_drift_rad, self._tilt_bias_tau_s
            )

    def _apply_tilt_bias_to_quat(self, q):
        aa = self._tilt_bias_axis_angle
        angle = float(np.linalg.norm(aa))
        if angle < 1e-9:
            return q
        axis = aa / angle
        h = 0.5 * angle
        s, c = float(np.sin(h)), float(np.cos(h))
        dqx, dqy, dqz, dqw = axis[0] * s, axis[1] * s, axis[2] * s, c
        px, py, pz, pw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        rx = pw * dqx + px * dqw + py * dqz - pz * dqy
        ry = pw * dqy - px * dqz + py * dqw + pz * dqx
        rz = pw * dqz + px * dqy - py * dqx + pz * dqw
        rw = pw * dqw - px * dqx - py * dqy - pz * dqz
        norm = float(np.sqrt(rx * rx + ry * ry + rz * rz + rw * rw)) or 1.0
        return [rx / norm, ry / norm, rz / norm, rw / norm]

    def _noise_quat_xyzw(self, q):
        axis = self._rng.normal(0.0, 1.0, size=3)
        n = float(np.linalg.norm(axis))
        if n < 1e-9 or self._tilt_sig_deg <= 0.0:
            return q
        axis = axis / n
        angle = float(np.deg2rad(self._rng.normal(0.0, self._tilt_sig_deg)))
        h = 0.5 * angle
        s, c = float(np.sin(h)), float(np.cos(h))
        dqx, dqy, dqz, dqw = axis[0] * s, axis[1] * s, axis[2] * s, c
        px, py, pz, pw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        rx = pw * dqx + px * dqw + py * dqz - pz * dqy
        ry = pw * dqy - px * dqz + py * dqw + pz * dqx
        rz = pw * dqz + px * dqy - py * dqx + pz * dqw
        rw = pw * dqw - px * dqx - py * dqy - pz * dqz
        norm = float(np.sqrt(rx * rx + ry * ry + rz * rz + rw * rw)) or 1.0
        return [rx / norm, ry / norm, rz / norm, rw / norm]

    def _replace_lin_vel(self, snap: dict) -> None:
        imu = snap.get("imu")
        if not isinstance(imu, dict):
            return
        if self._lin_vel_mode == "zero":
            imu["linear_velocity_mps"] = [0.0, 0.0, 0.0]
            imu["lin_vel_m_s"] = [0.0, 0.0, 0.0]
            return
        if self._lin_vel_mode == "leaky":
            now_s = float(snap.get("time_s", time.time()))
            prev_t = self._leaky_prev_t_s
            self._leaky_prev_t_s = now_s
            if prev_t is None or now_s <= prev_t:
                dt = 0.0
            else:
                dt = min(0.05, now_s - prev_t)
            a_imu = imu.get("linear_acceleration_mps2") or [0.0, 0.0, 0.0]
            a_imu = np.asarray(a_imu, dtype=float).reshape(-1)[:3]
            q = snap.get("orientation_quaternion_xyzw") or imu.get("quaternion_xyzw")
            if isinstance(q, (list, tuple)) and len(q) == 4:
                R = _quat_xyzw_to_rotmat_local(q)
                g_body = R.T @ np.array([0.0, 0.0, -9.81], dtype=float)
            else:
                g_body = np.array([0.0, 0.0, -9.81], dtype=float)
            a_motion = a_imu + g_body
            if dt > 0.0:
                self._leaky_v_body = self._leaky_v_body + a_motion * dt
                decay = float(np.exp(-dt / self._leaky_tau_s))
                self._leaky_v_body = self._leaky_v_body * decay
            v = self._leaky_v_body.tolist()
            imu["linear_velocity_mps"] = v
            imu["lin_vel_m_s"] = v

    def get_combined_state_snapshot(self, *args, **kwargs):
        # Obs dropout: return last snapshot unchanged with some probability
        if self._obs_dropout_prob > 0.0 and self._last_snap is not None:
            if float(self._rng.random()) < self._obs_dropout_prob:
                return self._last_snap

        snap = self._robot.get_combined_state_snapshot(*args, **kwargs)

        now_s = float(snap.get("time_s", time.time()))
        self._step_drift(now_s)

        if self._lin_vel_mode != "sim":
            self._replace_lin_vel(snap)

        if self._jp_sig > 0.0 and "joint_state_deg" in snap:
            jp = np.asarray(snap["joint_state_deg"], dtype=float)
            jp = jp + self._rng.normal(0.0, self._jp_sig, size=jp.shape)
            snap["joint_state_deg"] = jp.tolist()
            snap["joint_state_rad"] = np.deg2rad(jp).tolist()

        if self._jv_sig > 0.0 and "joint_velocity_rad_s" in snap:
            jv = np.asarray(snap["joint_velocity_rad_s"], dtype=float)
            jv = jv + self._rng.normal(0.0, self._jv_sig, size=jv.shape)
            snap["joint_velocity_rad_s"] = jv.tolist()
            snap["joint_velocity_deg_s"] = np.rad2deg(jv).tolist()

        imu = snap.get("imu")
        if isinstance(imu, dict):
            # White noise + bias drift on gyro
            if "gyro_rads" in imu:
                g = np.asarray(imu["gyro_rads"], dtype=float)
                if self._gyro_sig > 0.0:
                    g = g + self._rng.normal(0.0, self._gyro_sig, size=g.shape)
                if self._gyro_bias_drift_std > 0.0:
                    g = g + self._gyro_bias
                if self._gyro_sig > 0.0 or self._gyro_bias_drift_std > 0.0:
                    imu["gyro_rads"] = g.tolist()
                    imu["ang_vel_rad_s"] = g.tolist()
            if self._lv_sig > 0.0 and "linear_velocity_mps" in imu:
                lv = np.asarray(imu["linear_velocity_mps"], dtype=float)
                lv = lv + self._rng.normal(0.0, self._lv_sig, size=lv.shape)
                imu["linear_velocity_mps"] = lv.tolist()
                imu["lin_vel_m_s"] = lv.tolist()
            if "quaternion_xyzw" in imu:
                q = imu["quaternion_xyzw"]
                modified = False
                # Tilt bias drift (slow orientation wander)
                if self._tilt_bias_drift_rad > 0.0:
                    q = self._apply_tilt_bias_to_quat(q)
                    modified = True
                # White tilt noise on top
                if self._tilt_sig_deg > 0.0:
                    q = self._noise_quat_xyzw(q)
                    modified = True
                if modified:
                    imu["quaternion_xyzw"] = q
                    snap["orientation_quaternion_xyzw"] = tuple(q)

        self._last_snap = snap
        return snap

    def set_action(self, **kwargs):
        if self._act_noise_deg > 0.0:
            for side in ("left", "right"):
                d = kwargs.get(side)
                if isinstance(d, dict) and d:
                    kwargs[side] = {
                        k: float(v) + float(self._rng.normal(0.0, self._act_noise_deg))
                        for k, v in d.items()
                    }
        if self._act_latency_s > 0.0:
            time.sleep(self._act_latency_s)
        return self._robot.set_action(**kwargs)


class RandomPusher:
    """Applies horizontal force impulses to the torso on a poisson-like schedule.

    Start by calling start(); call stop() to join the thread. Runs independently of
    the sim loop — the sim loop sees nonzero xfrc_applied for ~50 ms per push.
    """

    TORSO_BODY_NAME = "torso_subassembly"
    IMPULSE_HOLD_S = 0.05

    def __init__(self, robot, *, mean_interval_s: float, force_n: float, seed: int = 0):
        import mujoco as _mj

        self._robot = robot
        self._rng = np.random.default_rng(int(seed) + 10_000)
        self._mean_interval_s = float(mean_interval_s)
        self._force_n = float(force_n)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._torso_id = int(
            _mj.mj_name2id(robot.model, _mj.mjtObj.mjOBJ_BODY, self.TORSO_BODY_NAME)
        )

    def start(self) -> None:
        if self._mean_interval_s <= 0.0 or self._force_n <= 0.0:
            return
        if self._torso_id < 0:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        try:
            with self._robot._sim_lock:
                if 0 <= self._torso_id < self._robot.model.nbody:
                    self._robot.data.xfrc_applied[self._torso_id, :] = 0.0
        except Exception:
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            wait_s = float(self._rng.exponential(self._mean_interval_s))
            if self._stop.wait(timeout=wait_s):
                break
            direction = self._rng.normal(size=3)
            direction[2] = 0.0
            n = float(np.linalg.norm(direction))
            if n < 1e-9:
                continue
            force = (direction / n) * self._force_n
            try:
                with self._robot._sim_lock:
                    self._robot.data.xfrc_applied[self._torso_id, 0:3] = force
                    self._robot.data.xfrc_applied[self._torso_id, 3:6] = 0.0
            except Exception:
                continue
            if self._stop.wait(timeout=self.IMPULSE_HOLD_S):
                break
            try:
                with self._robot._sim_lock:
                    self._robot.data.xfrc_applied[self._torso_id, :] = 0.0
            except Exception:
                pass


def _quat_xyzw_to_rotmat_local(q) -> np.ndarray:
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=float)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=float,
    )


def _quat_wxyz_to_rpy(qw: float, qx: float, qy: float, qz: float):
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = float(np.arctan2(sinr_cosp, cosr_cosp))
    sinp = float(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    pitch = float(np.arcsin(sinp))
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = float(np.arctan2(siny_cosp, cosy_cosp))
    return roll, pitch, yaw


def _base_tilt_rad(qw: float, qx: float, qy: float, qz: float) -> float:
    up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    return float(np.arccos(np.clip(up_z, -1.0, 1.0)))


def _apply_initial_perturbation(robot: SimBipedalRobotController, cfg: EvalConfig) -> None:
    import mujoco as _mj

    rng = np.random.default_rng(int(cfg.seed))
    joint_noise_rad = float(np.deg2rad(max(0.0, cfg.init_joint_noise_deg)))
    yaw_noise_rad = float(np.deg2rad(max(0.0, cfg.init_base_yaw_noise_deg)))
    if joint_noise_rad <= 0.0 and yaw_noise_rad <= 0.0:
        return
    with robot._sim_lock:
        if joint_noise_rad > 0.0:
            delta = rng.uniform(-joint_noise_rad, joint_noise_rad, size=len(robot._joint_qpos_adr))
            for adr, d in zip(robot._joint_qpos_adr, delta):
                robot.data.qpos[int(adr)] += float(d)
        if yaw_noise_rad > 0.0 and robot._has_free_base and robot.data.qpos.size >= 7:
            dyaw = float(rng.uniform(-yaw_noise_rad, yaw_noise_rad))
            half = 0.5 * dyaw
            qw = float(np.cos(half))
            qz = float(np.sin(half))
            w0 = float(robot.data.qpos[3])
            x0 = float(robot.data.qpos[4])
            y0 = float(robot.data.qpos[5])
            z0 = float(robot.data.qpos[6])
            # Compose q_yaw * q_base (apply yaw in world frame)
            nw = qw * w0 - qz * z0
            nx = qw * x0 - qz * y0
            ny = qw * y0 + qz * x0
            nz = qw * z0 + qz * w0
            robot.data.qpos[3] = nw
            robot.data.qpos[4] = nx
            robot.data.qpos[5] = ny
            robot.data.qpos[6] = nz
        _mj.mj_forward(robot.model, robot.data)
        robot._sync_state_from_sim(time.time())


def run_episode(eval_cfg: EvalConfig, *, policy_dir: Path) -> EpisodeMetrics:
    config_path = policy_dir / "config.yaml"
    policy_path = policy_dir / "policy.onnx"
    if not config_path.is_file() or not policy_path.is_file():
        raise FileNotFoundError(f"missing config.yaml or policy.onnx in {policy_dir}")

    robot = SimBipedalRobotController(
        control_hz=eval_cfg.control_hz,
        auto_reset_on_flip=False,
        auto_reset_on_divergence=False,
    )
    robot.start(mode="control", auto_enable=True)
    _apply_initial_perturbation(robot, eval_cfg)

    any_obs_noise = (
        eval_cfg.obs_joint_pos_noise_deg > 0.0
        or eval_cfg.obs_joint_vel_noise_rad_s > 0.0
        or eval_cfg.obs_gyro_noise_rad_s > 0.0
        or eval_cfg.obs_lin_vel_noise_mps > 0.0
        or eval_cfg.obs_quat_tilt_noise_deg > 0.0
    )
    any_action_pert = eval_cfg.action_latency_ms > 0.0 or eval_cfg.action_noise_deg > 0.0
    lin_vel_nonstd = eval_cfg.lin_vel_mode != "sim"
    any_drift = eval_cfg.gyro_bias_drift_std > 0.0 or eval_cfg.tilt_bias_drift_deg_per_sqrt_s > 0.0
    any_dropout = eval_cfg.obs_dropout_prob > 0.0
    if any_obs_noise or any_action_pert or lin_vel_nonstd or any_drift or any_dropout:
        agent_robot = PerturbedRobot(
            robot,
            obs_joint_pos_noise_deg=eval_cfg.obs_joint_pos_noise_deg,
            obs_joint_vel_noise_rad_s=eval_cfg.obs_joint_vel_noise_rad_s,
            obs_gyro_noise_rad_s=eval_cfg.obs_gyro_noise_rad_s,
            obs_lin_vel_noise_mps=eval_cfg.obs_lin_vel_noise_mps,
            obs_quat_tilt_noise_deg=eval_cfg.obs_quat_tilt_noise_deg,
            action_latency_ms=eval_cfg.action_latency_ms,
            action_noise_deg=eval_cfg.action_noise_deg,
            lin_vel_mode=eval_cfg.lin_vel_mode,
            leaky_lin_vel_tau_s=eval_cfg.leaky_lin_vel_tau_s,
            gyro_bias_drift_std=eval_cfg.gyro_bias_drift_std,
            gyro_bias_tau_s=eval_cfg.gyro_bias_tau_s,
            tilt_bias_drift_deg_per_sqrt_s=eval_cfg.tilt_bias_drift_deg_per_sqrt_s,
            tilt_bias_tau_s=eval_cfg.tilt_bias_tau_s,
            obs_dropout_prob=eval_cfg.obs_dropout_prob,
            seed=eval_cfg.seed,
        )
    else:
        agent_robot = robot

    pusher: Optional[RandomPusher] = None
    if eval_cfg.push_interval_s > 0.0 and eval_cfg.push_force_n > 0.0:
        pusher = RandomPusher(
            robot,
            mean_interval_s=eval_cfg.push_interval_s,
            force_n=eval_cfg.push_force_n,
            seed=eval_cfg.seed,
        )

    agent = RLAgent.from_files(
        robot=agent_robot,
        config_path=str(config_path),
        policy_path=str(policy_path),
    )
    agent.set_command_source(ConstantCommandSource(
        lin_x=eval_cfg.command_vx,
        lin_y=eval_cfg.command_vy,
        yaw_rate=eval_cfg.command_yaw,
    ))
    agent.start()
    if pusher is not None:
        pusher.start()

    sample_dt = 1.0 / max(1.0, eval_cfg.sample_hz)
    start_perf = time.perf_counter()
    next_sample = start_perf

    heights: List[float] = []
    abs_pitches_deg: List[float] = []
    abs_rolls_deg: List[float] = []
    vxs_body: List[float] = []
    x0: Optional[float] = None
    y0: Optional[float] = None
    final_xy = np.zeros(2, dtype=float)

    survived = True
    fall_reason = "timeout"
    survival_time = eval_cfg.duration_s

    try:
        while True:
            elapsed = time.perf_counter() - start_perf
            if elapsed >= eval_cfg.duration_s:
                break

            with robot._sim_lock:
                base_pos = np.asarray(robot.data.qpos[0:3], dtype=float).copy()
                base_q_wxyz = np.asarray(robot.data.qpos[3:7], dtype=float).copy()
                base_vel_world = np.asarray(robot.data.qvel[0:3], dtype=float).copy()
            qw, qx, qy, qz = (float(v) for v in base_q_wxyz)
            roll, pitch, yaw = _quat_wxyz_to_rpy(qw, qx, qy, qz)
            base_z = float(base_pos[2])

            cy, sy = float(np.cos(yaw)), float(np.sin(yaw))
            vx_body = cy * float(base_vel_world[0]) + sy * float(base_vel_world[1])

            if elapsed >= eval_cfg.warmup_s:
                heights.append(base_z)
                abs_pitches_deg.append(abs(float(np.rad2deg(pitch))))
                abs_rolls_deg.append(abs(float(np.rad2deg(roll))))
                vxs_body.append(vx_body)
                if x0 is None:
                    x0 = float(base_pos[0])
                    y0 = float(base_pos[1])
                final_xy[:] = base_pos[0:2]

                if base_z < eval_cfg.fall_height_m:
                    survived = False
                    fall_reason = f"low_base(z={base_z:.2f})"
                    survival_time = elapsed
                    break

                tilt = _base_tilt_rad(qw, qx, qy, qz)
                if tilt > eval_cfg.flip_angle_rad:
                    survived = False
                    fall_reason = f"flipped(tilt={np.rad2deg(tilt):.0f}deg)"
                    survival_time = elapsed
                    break

            next_sample += sample_dt
            sleep_s = next_sample - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_sample = time.perf_counter()
    finally:
        try:
            if pusher is not None:
                pusher.stop()
        finally:
            try:
                agent.stop()
            finally:
                robot.stop()

    if x0 is None or y0 is None:
        dx = dy = 0.0
    else:
        dx = float(final_xy[0] - x0)
        dy = float(final_xy[1] - y0)

    mean_vx = float(np.mean(vxs_body)) if vxs_body else 0.0
    return EpisodeMetrics(
        policy_name=eval_cfg.policy_name,
        command_vx=eval_cfg.command_vx,
        seed=int(eval_cfg.seed),
        duration_s=eval_cfg.duration_s,
        survived=survived,
        survival_time_s=float(survival_time),
        fall_reason=fall_reason,
        distance_x=dx,
        distance_y=dy,
        mean_base_height=float(np.mean(heights)) if heights else 0.0,
        std_base_height=float(np.std(heights)) if heights else 0.0,
        mean_abs_pitch_deg=float(np.mean(abs_pitches_deg)) if abs_pitches_deg else 0.0,
        mean_abs_roll_deg=float(np.mean(abs_rolls_deg)) if abs_rolls_deg else 0.0,
        mean_measured_vx=mean_vx,
        tracking_err_vx=float(mean_vx - eval_cfg.command_vx),
        num_samples=len(heights),
    )


def _print_summary(results: List[EpisodeMetrics]) -> None:
    hdr = (
        f"{'policy':<42} {'vx':>5} {'seed':>4} {'surv_s':>7} {'reason':<20} "
        f"{'dx':>6} {'dy':>6} {'z_mean':>7} {'z_std':>6} "
        f"{'|p|deg':>7} {'|r|deg':>7} {'vx_meas':>8} {'trk_err':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r.policy_name[:41]:<42} {r.command_vx:>5.2f} {r.seed:>4d} {r.survival_time_s:>7.2f} "
            f"{r.fall_reason[:19]:<20} {r.distance_x:>6.2f} {r.distance_y:>6.2f} "
            f"{r.mean_base_height:>7.2f} {r.std_base_height:>6.3f} "
            f"{r.mean_abs_pitch_deg:>7.1f} {r.mean_abs_roll_deg:>7.1f} "
            f"{r.mean_measured_vx:>8.2f} {r.tracking_err_vx:>8.2f}"
        )


def _print_aggregate(results: List[EpisodeMetrics]) -> None:
    from collections import defaultdict

    groups: dict = defaultdict(list)
    for r in results:
        groups[(r.policy_name, r.command_vx)].append(r)

    hdr = (
        f"{'policy':<42} {'vx_cmd':>6} {'N':>3} {'pass':>5} "
        f"{'surv_med':>9} {'surv_min':>9} {'surv_max':>9} "
        f"{'dx_med':>7} {'vx_med':>7} {'|p|_med':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for (name, vx), rs in sorted(groups.items()):
        n = len(rs)
        duration = rs[0].duration_s
        n_pass = sum(1 for r in rs if r.survived)
        survivals = sorted(r.survival_time_s for r in rs)
        dxs = sorted(r.distance_x for r in rs)
        vxs = sorted(r.mean_measured_vx for r in rs)
        pitches = sorted(r.mean_abs_pitch_deg for r in rs)

        def _med(xs):
            if not xs:
                return 0.0
            mid = len(xs) // 2
            if len(xs) % 2:
                return float(xs[mid])
            return float(0.5 * (xs[mid - 1] + xs[mid]))

        pass_rate = f"{n_pass}/{n}"
        print(
            f"{name[:41]:<42} {vx:>6.2f} {n:>3d} {pass_rate:>5} "
            f"{_med(survivals):>9.2f} {min(survivals):>9.2f} {max(survivals):>9.2f} "
            f"{_med(dxs):>7.2f} {_med(vxs):>7.2f} {_med(pitches):>8.1f}"
        )
        _ = duration


def _discover_policies(policy_root: Path) -> List[str]:
    out = []
    for p in sorted(policy_root.iterdir()):
        if p.is_dir() and (p / "policy.onnx").is_file() and (p / "config.yaml").is_file():
            out.append(p.name)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Headless sim-eval for RL walking policies.")
    ap.add_argument("--policy-root", default=str(DEFAULT_POLICY_ROOT))
    ap.add_argument("--policies", nargs="*", default=None,
                    help="Subdirectories under --policy-root. Default: all with policy.onnx + config.yaml.")
    ap.add_argument("--hf-repo", default=None,
                    help="HuggingFace repo id (e.g. CarolinePascal/lerobot-humanoid-noarms-velocity-v16). "
                         "Downloads the snapshot, finds policy.onnx + env.yaml (or config.yaml), and uses it "
                         "as the policy root. Overrides --policy-root/--policies.")
    ap.add_argument("--hf-subdir", default=None,
                    help="Subdirectory inside the HF repo that contains policy.onnx + env.yaml. "
                         "Auto-detected if there is exactly one such subdir.")
    ap.add_argument("--cmd-vx", nargs="*", type=float, default=[0.0, 0.3, 0.6])
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--warmup", type=float, default=1.0)
    ap.add_argument("--fall-height", type=float, default=0.35)
    ap.add_argument("--flip-deg", type=float, default=70.0)
    ap.add_argument("--control-hz", type=float, default=200.0)
    ap.add_argument("--seeds", type=int, default=1,
                    help="Number of seeds (trials) per (policy, vx). Default 1.")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--init-joint-noise-deg", type=float, default=2.0,
                    help="Per-joint uniform noise applied at reset. Set 0 to disable.")
    ap.add_argument("--init-base-yaw-noise-deg", type=float, default=5.0,
                    help="Uniform noise on initial base yaw. Set 0 to disable.")
    # OOD runtime perturbations (0 by default; exceed training levels to stress-test).
    ap.add_argument("--obs-joint-pos-noise-deg", type=float, default=0.0,
                    help="Gaussian noise stdev added to joint_pos obs. Training trained on uniform +-2.5 deg.")
    ap.add_argument("--obs-joint-vel-noise-rad-s", type=float, default=0.0,
                    help="Gaussian noise stdev added to joint_vel obs. Training: uniform +-0.05 rad/s.")
    ap.add_argument("--obs-gyro-noise-rad-s", type=float, default=0.0,
                    help="Gaussian noise stdev added to gyro/ang_vel obs. Training: +-0.06 rad/s.")
    ap.add_argument("--obs-lin-vel-noise-mps", type=float, default=0.0,
                    help="Gaussian noise stdev added to imu lin_vel obs. Training: +-0.075 m/s.")
    ap.add_argument("--obs-quat-tilt-noise-deg", type=float, default=0.0,
                    help="Gaussian random-axis tilt stdev added to IMU quaternion.")
    ap.add_argument("--action-latency-ms", type=float, default=0.0,
                    help="Sleep this many ms inside set_action to simulate motor-command latency (OOD: training has 0).")
    ap.add_argument("--action-noise-deg", type=float, default=0.0,
                    help="Gaussian noise stdev added to each action command in degrees.")
    ap.add_argument("--push-interval-s", type=float, default=0.0,
                    help="Mean interval between external pushes (exponential). 0 disables.")
    ap.add_argument("--push-force-n", type=float, default=0.0,
                    help="Magnitude of random horizontal push force on torso (Newtons).")
    ap.add_argument("--lin-vel-mode", choices=["sim", "zero", "leaky"], default="sim",
                    help="How to populate base_lin_vel obs. 'sim' = ground-truth velocimeter (training). "
                         "'zero' = force to [0,0,0]. 'leaky' = leaky-integrate body accel (mimics real JY901).")
    ap.add_argument("--leaky-lin-vel-tau-s", type=float, default=1.5,
                    help="Time constant for the leaky integrator in --lin-vel-mode=leaky.")
    ap.add_argument("--gyro-bias-drift-std", type=float, default=0.0,
                    help="Ornstein-Uhlenbeck diffusion rate for gyro bias drift (rad/s/sqrt(s)). "
                         "JY901-like: ~0.001. Set >0 to enable drift.")
    ap.add_argument("--gyro-bias-tau-s", type=float, default=100.0,
                    help="Mean-reversion time constant for gyro bias (seconds).")
    ap.add_argument("--tilt-bias-drift-deg-per-sqrt-s", type=float, default=0.0,
                    help="Orientation drift diffusion rate (deg/sqrt(s)). JY901-like: ~0.3-0.5.")
    ap.add_argument("--tilt-bias-tau-s", type=float, default=200.0,
                    help="Mean-reversion time constant for tilt bias (seconds).")
    ap.add_argument("--obs-dropout-prob", type=float, default=0.0,
                    help="Per-step probability of returning stale obs (simulates dropped serial frame).")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if args.hf_repo:
        from huggingface_hub import snapshot_download
        snap = Path(snapshot_download(args.hf_repo))
        def _looks_like_policy_dir(d: Path) -> bool:
            return (d / "policy.onnx").is_file() and (
                (d / "config.yaml").is_file() or (d / "env.yaml").is_file()
            )
        if args.hf_subdir:
            policy_dir = snap / args.hf_subdir
        elif _looks_like_policy_dir(snap):
            policy_dir = snap
        else:
            candidates = [d for d in snap.iterdir() if d.is_dir() and _looks_like_policy_dir(d)]
            if len(candidates) != 1:
                print(f"[eval][ERR] expected exactly one policy subdir in {snap}, got {[d.name for d in candidates]}")
                return 2
            policy_dir = candidates[0]

        # env.yaml (AGILE / Isaac Lab) and config.yaml (mjlab) use different schemas.
        # RL_agent_isolated expects mjlab layout; translate when only env.yaml exists.
        if not (policy_dir / "config.yaml").is_file() and (policy_dir / "env.yaml").is_file():
            _adapt_agile_env_to_mjlab_config(
                policy_dir / "env.yaml", policy_dir / "config.yaml"
            )
        policy_root = policy_dir.parent
        policies = [policy_dir.name]
        print(f"[eval] hf_repo = {args.hf_repo}  resolved → {policy_dir}")
    else:
        policy_root = Path(args.policy_root).resolve()
        if not policy_root.is_dir():
            print(f"[eval][ERR] policy root not found: {policy_root}")
            return 2

        policies = args.policies if args.policies else _discover_policies(policy_root)
    if not policies:
        print(f"[eval][ERR] no policies found under {policy_root}")
        return 2

    print(f"[eval] policy_root = {policy_root}")
    print(f"[eval] policies    = {policies}")
    print(f"[eval] cmd_vx list = {args.cmd_vx}")
    print(f"[eval] duration_s  = {args.duration}  warmup_s = {args.warmup}")
    print()

    n_seeds = max(1, int(args.seeds))
    seed_start = int(args.seed_start)

    results: List[EpisodeMetrics] = []
    for name in policies:
        pdir = policy_root / name
        for vx in args.cmd_vx:
            for k in range(n_seeds):
                seed = seed_start + k
                cfg = EvalConfig(
                    policy_name=name,
                    command_vx=float(vx),
                    duration_s=float(args.duration),
                    warmup_s=float(args.warmup),
                    fall_height_m=float(args.fall_height),
                    flip_angle_rad=float(np.deg2rad(args.flip_deg)),
                    control_hz=float(args.control_hz),
                    seed=seed,
                    init_joint_noise_deg=float(args.init_joint_noise_deg),
                    init_base_yaw_noise_deg=float(args.init_base_yaw_noise_deg),
                    obs_joint_pos_noise_deg=float(args.obs_joint_pos_noise_deg),
                    obs_joint_vel_noise_rad_s=float(args.obs_joint_vel_noise_rad_s),
                    obs_gyro_noise_rad_s=float(args.obs_gyro_noise_rad_s),
                    obs_lin_vel_noise_mps=float(args.obs_lin_vel_noise_mps),
                    obs_quat_tilt_noise_deg=float(args.obs_quat_tilt_noise_deg),
                    action_latency_ms=float(args.action_latency_ms),
                    action_noise_deg=float(args.action_noise_deg),
                    push_interval_s=float(args.push_interval_s),
                    push_force_n=float(args.push_force_n),
                    lin_vel_mode=str(args.lin_vel_mode),
                    leaky_lin_vel_tau_s=float(args.leaky_lin_vel_tau_s),
                    gyro_bias_drift_std=float(args.gyro_bias_drift_std),
                    gyro_bias_tau_s=float(args.gyro_bias_tau_s),
                    tilt_bias_drift_deg_per_sqrt_s=float(args.tilt_bias_drift_deg_per_sqrt_s),
                    tilt_bias_tau_s=float(args.tilt_bias_tau_s),
                    obs_dropout_prob=float(args.obs_dropout_prob),
                )
                print(f"[eval] {name:<42} vx={vx:>5.2f} seed={seed} ...", flush=True)
                try:
                    m = run_episode(cfg, policy_dir=pdir)
                except Exception as exc:
                    print(f"[eval][ERR] {name} vx={vx} seed={seed}: {type(exc).__name__}: {exc}")
                    continue
                results.append(m)
                ok = "OK " if m.survived else "FELL"
                print(
                    f"       -> {ok} t={m.survival_time_s:5.2f}s  dx={m.distance_x:5.2f}m  "
                    f"vx={m.mean_measured_vx:5.2f}  reason={m.fall_reason}"
                )

    print()
    print("=== Per-run detail ===")
    _print_summary(results)

    if n_seeds > 1:
        print()
        print("=== Aggregated by (policy, vx) ===")
        _print_aggregate(results)

    if args.json_out:
        with open(args.json_out, "w") as fp:
            json.dump([asdict(r) for r in results], fp, indent=2)
        print(f"\n[eval] JSON written to {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
