import math
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import serial


@dataclass(frozen=True)
class QuaternionSample:
    """Quaternion in (w, x, y, z) + metadata."""
    w: float
    x: float
    y: float
    z: float
    timestamp: float  # time.time() when sample was updated
    source: str       # "quat" if from 0x59, "euler" if derived from angles


@dataclass(frozen=True)
class VectorSample:
    """3D vector sample (x,y,z) + timestamp."""
    x: float
    y: float
    z: float
    timestamp: float


@dataclass(frozen=True)
class ImuObservation:
    """Convenience bundle for RL observations."""
    quat: Optional[QuaternionSample]
    ang_vel_rad_s: Optional[VectorSample]   # body-frame angular velocity (rad/s)
    lin_acc_m_s2: Optional[VectorSample]    # body-frame linear acceleration (m/s^2) with gravity removed
    lin_vel_m_s: Optional[VectorSample]     # body-frame linear velocity (m/s), integrated (drifts!)
    timestamp: float


class JY901IMU:
    """
    Non-blocking JY901/WitMotion UART reader.
    - Background thread reads frames continuously.
    - Exposes latest quaternion, angular velocity, linear acceleration, and (integrated) linear velocity.
    - Angular velocity comes from gyro frame 0x52.
    - Linear acceleration is accel frame 0x51 with gravity removed using latest quaternion.
    - Linear velocity is trapezoidal integration of linear acceleration (WILL drift).

    Notes for RL:
      - ang_vel is generally great.
      - lin_acc (gravity removed) is often useful.
      - lin_vel will drift; consider using it only over short horizons or with resets/filters.
    """

    FRAME_LEN = 11
    HEAD = 0x55
    G = 9.80665  # m/s^2

    def __init__(
        self,
        port: str = "/dev/ttyAMA0",
        baudrate: int = 9600,
        timeout_s: float = 0.2,
        thread_name: str = "JY901IMUReader",
        accel_full_scale_g: float = 16.0,          # JY901 typical: ±16g
        gyro_full_scale_dps: float = 2000.0,       # JY901 typical: ±2000 deg/s
        vel_leak: float = 0.0,                      # 0=no leak; small leak (e.g. 0.02) can tame drift
        max_dt_s: float = 0.2,                      # clamp dt to avoid huge jumps (0.05 is too strict at 9600 baud)
        vel_kf_accel_proc_std_m_s2: float = 3.0,    # process noise on acceleration driving velocity prediction
        vel_kf_zero_meas_std_m_s: float = 0.15,      # pseudo-measurement z=0 (pull velocity toward zero)
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout_s = timeout_s
        self.thread_name = thread_name

        self.accel_full_scale_g = accel_full_scale_g
        self.gyro_full_scale_dps = gyro_full_scale_dps
        self.vel_leak = float(vel_leak)
        self.max_dt_s = float(max_dt_s)
        self.vel_kf_accel_proc_std_m_s2 = float(vel_kf_accel_proc_std_m_s2)
        self.vel_kf_zero_meas_std_m_s = float(vel_kf_zero_meas_std_m_s)

        self._ser: Optional[serial.Serial] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

        self._lock = threading.Lock()
        self._latest_quat: Optional[QuaternionSample] = None
        self._latest_ang_vel: Optional[VectorSample] = None
        self._latest_lin_acc: Optional[VectorSample] = None
        self._latest_lin_vel: Optional[VectorSample] = None

        # Integration state
        self._prev_lin_acc: Optional[Tuple[float, float, float, float]] = None  # ax,ay,az,t
        self._vel_xyz = [0.0, 0.0, 0.0]
        self._vel_var_xyz = [1.0, 1.0, 1.0]

        self._stats_lock = threading.Lock()
        self._frames_ok = 0
        self._frames_bad = 0
        self._last_frame_ts = 0.0
        self._lin_vel_updates = 0
        self._lin_vel_skip_dt = 0
        self._lin_vel_skip_no_prev = 0
        self._lin_vel_kf_updates = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout_s)
        self._thread = threading.Thread(target=self._run, name=self.thread_name, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=1.0)
        self._thread = None
        if self._ser:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    # ---------- non-blocking getters ----------

    def get_quaternion(self) -> Optional[QuaternionSample]:
        with self._lock:
            return self._latest_quat

    def get_angular_velocity(self) -> Optional[VectorSample]:
        """Body-frame angular velocity (rad/s)."""
        with self._lock:
            return self._latest_ang_vel

    def get_linear_acceleration(self) -> Optional[VectorSample]:
        """Body-frame linear acceleration (m/s^2), gravity removed using latest quaternion."""
        with self._lock:
            return self._latest_lin_acc

    def get_linear_velocity(self) -> Optional[VectorSample]:
        """Body-frame linear velocity (m/s), integrated from lin_acc (drifts!)."""
        with self._lock:
            return self._latest_lin_vel

    def get_observation(self) -> ImuObservation:
        """Single non-blocking snapshot (handy for RL)."""
        with self._lock:
            quat = self._latest_quat
            ang = self._latest_ang_vel
            acc = self._latest_lin_acc
            vel = self._latest_lin_vel
        ts = time.time()
        return ImuObservation(quat=quat, ang_vel_rad_s=ang, lin_acc_m_s2=acc, lin_vel_m_s=vel, timestamp=ts)

    def reset_velocity(self) -> None:
        """Reset integrated linear velocity (useful at episode reset / known standstill)."""
        with self._lock:
            self._vel_xyz = [0.0, 0.0, 0.0]
            self._vel_var_xyz = [1.0, 1.0, 1.0]
            self._latest_lin_vel = VectorSample(0.0, 0.0, 0.0, time.time())
        self._prev_lin_acc = None

    def get_stats(self) -> dict:
        with self._stats_lock:
            return {
                "frames_ok": self._frames_ok,
                "frames_bad": self._frames_bad,
                "last_frame_age_s": (time.time() - self._last_frame_ts) if self._last_frame_ts else None,
                "lin_vel_updates": self._lin_vel_updates,
                "lin_vel_skip_dt": self._lin_vel_skip_dt,
                "lin_vel_skip_no_prev": self._lin_vel_skip_no_prev,
                "lin_vel_kf_updates": self._lin_vel_kf_updates,
                "max_dt_s": self.max_dt_s,
                "vel_kf_accel_proc_std_m_s2": self.vel_kf_accel_proc_std_m_s2,
                "vel_kf_zero_meas_std_m_s": self.vel_kf_zero_meas_std_m_s,
                "port": self.port,
                "baudrate": self.baudrate,
            }

    # ---------- internal helpers ----------

    @staticmethod
    def _checksum_ok(frame: bytes) -> bool:
        return len(frame) == JY901IMU.FRAME_LEN and ((sum(frame[:10]) & 0xFF) == frame[10])

    @staticmethod
    def _i16(lo: int, hi: int) -> int:
        v = (hi << 8) | lo
        return v - 0x10000 if v & 0x8000 else v

    @staticmethod
    def _quat_from_euler_deg(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
        # ZYX convention: yaw (Z), pitch (Y), roll (X)
        r = math.radians(roll)
        p = math.radians(pitch)
        y = math.radians(yaw)

        cy = math.cos(y * 0.5)
        sy = math.sin(y * 0.5)
        cp = math.cos(p * 0.5)
        sp = math.sin(p * 0.5)
        cr = math.cos(r * 0.5)
        sr = math.sin(r * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        yq = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return (w, x, yq, z)

    @staticmethod
    def _quat_conj(w: float, x: float, y: float, z: float) -> Tuple[float, float, float, float]:
        return (w, -x, -y, -z)

    @staticmethod
    def _quat_mul(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )

    @classmethod
    def _rotate_world_to_body(cls, q_wxyz: Tuple[float, float, float, float], v_xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Rotate vector from world to body frame using quaternion (w,x,y,z).

        Assumption: q describes body orientation in world (body->world rotation).
        Then world->body is q_conj * v * q.

        If your convention differs, swap conjugation.
        """
        w, x, y, z = q_wxyz
        qc = cls._quat_conj(w, x, y, z)
        vx, vy, vz = v_xyz
        vq = (0.0, vx, vy, vz)
        out = cls._quat_mul(cls._quat_mul(qc, vq), (w, x, y, z))
        return (out[1], out[2], out[3])

    def _update_quat(self, sample: QuaternionSample) -> None:
        with self._lock:
            self._latest_quat = sample

    # ---------- reader thread ----------

    def _run(self) -> None:
        assert self._ser is not None
        ser = self._ser

        while not self._stop_evt.is_set():
            try:
                b = ser.read(1)
                if not b or b[0] != self.HEAD:
                    continue

                rest = ser.read(self.FRAME_LEN - 1)
                if len(rest) != self.FRAME_LEN - 1:
                    with self._stats_lock:
                        self._frames_bad += 1
                    continue

                frame = bytes([self.HEAD]) + rest
                if not self._checksum_ok(frame):
                    with self._stats_lock:
                        self._frames_bad += 1
                    continue

                ftype = frame[1]
                payload = frame[2:10]
                now = time.time()

                with self._stats_lock:
                    self._frames_ok += 1
                    self._last_frame_ts = now

                # 0x59: quaternion (q0,q1,q2,q3) int16, scale 1/32768
                if ftype == 0x59:
                    q0 = self._i16(payload[0], payload[1]) / 32768.0
                    q1 = self._i16(payload[2], payload[3]) / 32768.0
                    q2 = self._i16(payload[4], payload[5]) / 32768.0
                    q3 = self._i16(payload[6], payload[7]) / 32768.0
                    self._update_quat(QuaternionSample(w=q0, x=q1, y=q2, z=q3, timestamp=now, source="quat"))
                    continue

                # 0x53: Euler angles (deg). Use as fallback for quaternion.
                if ftype == 0x53:
                    roll = self._i16(payload[0], payload[1]) / 32768.0 * 180.0
                    pitch = self._i16(payload[2], payload[3]) / 32768.0 * 180.0
                    yaw = self._i16(payload[4], payload[5]) / 32768.0 * 180.0
                    w, x, yq, z = self._quat_from_euler_deg(roll, pitch, yaw)
                    self._update_quat(QuaternionSample(w=w, x=x, y=yq, z=z, timestamp=now, source="euler"))
                    continue

                # 0x52: gyro (deg/s), scale int16 / 32768 * full_scale_dps
                if ftype == 0x52:
                    gx = self._i16(payload[0], payload[1]) / 32768.0 * self.gyro_full_scale_dps
                    gy = self._i16(payload[2], payload[3]) / 32768.0 * self.gyro_full_scale_dps
                    gz = self._i16(payload[4], payload[5]) / 32768.0 * self.gyro_full_scale_dps
                    # convert to rad/s
                    gx *= math.pi / 180.0
                    gy *= math.pi / 180.0
                    gz *= math.pi / 180.0
                    with self._lock:
                        self._latest_ang_vel = VectorSample(gx, gy, gz, now)
                    continue

                # 0x51: accel (g), scale int16 / 32768 * full_scale_g
                if ftype == 0x51:
                    ax_g = self._i16(payload[0], payload[1]) / 32768.0 * self.accel_full_scale_g
                    ay_g = self._i16(payload[2], payload[3]) / 32768.0 * self.accel_full_scale_g
                    az_g = self._i16(payload[4], payload[5]) / 32768.0 * self.accel_full_scale_g

                    # Convert to m/s^2 (still includes gravity)
                    ax = ax_g * self.G
                    ay = ay_g * self.G
                    az = az_g * self.G

                    # Remove gravity using latest quaternion.
                    # Empirically with JY901 quaternion convention, using world gravity
                    # as +Z gives near-zero linear acceleration at rest.
                    # If this regresses on your setup, revisit quaternion convention.
                    with self._lock:
                        q = self._latest_quat

                    if q is not None:
                        g_bx, g_by, g_bz = self._rotate_world_to_body((q.w, q.x, q.y, q.z), (0.0, 0.0, +self.G))
                        lax = ax - g_bx
                        lay = ay - g_by
                        laz = az - g_bz
                    else:
                        # No orientation yet; best effort: don't remove gravity.
                        lax, lay, laz = ax, ay, az

                    # Update latest linear acceleration (body frame)
                    with self._lock:
                        self._latest_lin_acc = VectorSample(lax, lay, laz, now)

                    # Estimate linear velocity (body frame) from linear acceleration.
                    # We use a simple per-axis Kalman-style filter:
                    #   predict: v <- v + a*dt
                    #   update:  pseudo-measurement z=0 (pull toward standstill)
                    # This acts like a drift-tolerant integrator with an explicit zero-velocity prior.
                    prev = self._prev_lin_acc
                    if prev is not None:
                        pax, pay, paz, pt = prev
                        dt = now - pt
                        if 0.0 < dt <= self.max_dt_s:
                            # Trapezoidal acceleration for the prediction step.
                            a_mid = (
                                0.5 * (pax + lax),
                                0.5 * (pay + lay),
                                0.5 * (paz + laz),
                            )
                            vx, vy, vz = self._vel_xyz
                            pvx, pvy, pvz = self._vel_var_xyz
                            v_list = [vx, vy, vz]
                            p_list = [pvx, pvy, pvz]

                            q_var = (max(1e-6, self.vel_kf_accel_proc_std_m_s2) * dt) ** 2
                            r_var = max(1e-8, self.vel_kf_zero_meas_std_m_s) ** 2

                            for i, a_i in enumerate(a_mid):
                                # Predict velocity from acceleration.
                                v_pred = float(v_list[i]) + float(a_i) * float(dt)
                                p_pred = float(p_list[i]) + float(q_var)

                                # Optional leak remains as an extra damping knob.
                                if self.vel_leak > 0.0:
                                    decay = max(0.0, 1.0 - self.vel_leak * dt)
                                    v_pred *= decay

                                # Pseudo-measurement z=0 m/s.
                                k = p_pred / (p_pred + r_var)
                                v_upd = (1.0 - k) * v_pred
                                p_upd = max(1e-9, (1.0 - k) * p_pred)
                                v_list[i] = float(v_upd)
                                p_list[i] = float(p_upd)

                            self._vel_xyz = [float(v_list[0]), float(v_list[1]), float(v_list[2])]
                            self._vel_var_xyz = [float(p_list[0]), float(p_list[1]), float(p_list[2])]
                            with self._lock:
                                self._latest_lin_vel = VectorSample(
                                    self._vel_xyz[0], self._vel_xyz[1], self._vel_xyz[2], now
                                )
                            with self._stats_lock:
                                self._lin_vel_updates += 1
                                self._lin_vel_kf_updates += 1
                        else:
                            with self._stats_lock:
                                self._lin_vel_skip_dt += 1
                    else:
                        # Seed output with zero velocity on first valid accel sample so
                        # callers don't see `None` forever while waiting for the next frame.
                        with self._lock:
                            self._latest_lin_vel = VectorSample(self._vel_xyz[0], self._vel_xyz[1], self._vel_xyz[2], now)
                        with self._stats_lock:
                            self._lin_vel_skip_no_prev += 1

                    self._prev_lin_acc = (lax, lay, laz, now)
                    continue

                # ignore other types
            except (serial.SerialException, OSError):
                time.sleep(0.1)
            except Exception:
                with self._stats_lock:
                    self._frames_bad += 1
                time.sleep(0.01)


if __name__ == "__main__":
    imu = JY901IMU("/dev/ttyAMA0", 9600, vel_leak=0.02)
    imu.start()
    try:
        while True:
            obs = imu.get_observation()
            if obs.quat and obs.ang_vel_rad_s and obs.lin_acc_m_s2:
                q = obs.quat
                w = obs.ang_vel_rad_s
                a = obs.lin_acc_m_s2
                v = obs.lin_vel_m_s
                print(
                    f"q=({q.w:+.3f},{q.x:+.3f},{q.y:+.3f},{q.z:+.3f}) "
                    f"w(rad/s)=({w.x:+.2f},{w.y:+.2f},{w.z:+.2f}) "
                    f"a(m/s2)=({a.x:+.2f},{a.y:+.2f},{a.z:+.2f}) "
                    f"v(m/s)=({(v.x if v else 0):+.2f},{(v.y if v else 0):+.2f},{(v.z if v else 0):+.2f})"
                )
            time.sleep(0.02)
    finally:
        imu.stop()
