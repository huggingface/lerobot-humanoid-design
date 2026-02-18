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


class JY901IMU:
    """
    Non-blocking JY901/WitMotion UART reader.
    - Spawns a background thread that continuously reads frames.
    - Exposes latest quaternion via get_quaternion() without blocking.
    - Supports native quaternion frame (0x59) when available.
      Falls back to quaternion computed from angles (0x53) if needed.

    Typical:
        imu = JY901IMU("/dev/ttyAMA0", 9600)
        imu.start()
        q = imu.get_quaternion()
        imu.stop()
    """

    # Frame: 0x55 + TYPE + 8 payload bytes + checksum (sum of first 10 bytes & 0xFF)
    FRAME_LEN = 11
    HEAD = 0x55

    def __init__(
        self,
        port: str = "/dev/ttyAMA0",
        baudrate: int = 9600,
        timeout_s: float = 0.2,
        thread_name: str = "JY901IMUReader",
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout_s = timeout_s
        self.thread_name = thread_name

        self._ser: Optional[serial.Serial] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

        self._lock = threading.Lock()
        self._latest_quat: Optional[QuaternionSample] = None
        self._latest_angles_deg: Optional[Tuple[float, float, float, float]] = None  # (roll,pitch,yaw,timestamp)

        self._stats_lock = threading.Lock()
        self._frames_ok = 0
        self._frames_bad = 0
        self._last_frame_ts = 0.0

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

    def get_quaternion(self) -> Optional[QuaternionSample]:
        """Non-blocking: returns latest quaternion sample, or None if not yet available."""
        with self._lock:
            return self._latest_quat

    def get_stats(self) -> dict:
        """Non-blocking stats snapshot."""
        with self._stats_lock:
            return {
                "frames_ok": self._frames_ok,
                "frames_bad": self._frames_bad,
                "last_frame_age_s": (time.time() - self._last_frame_ts) if self._last_frame_ts else None,
                "port": self.port,
                "baudrate": self.baudrate,
            }

    # ---------------- internal ----------------

    @staticmethod
    def _checksum_ok(frame: bytes) -> bool:
        if len(frame) != JY901IMU.FRAME_LEN:
            return False
        return (sum(frame[:10]) & 0xFF) == frame[10]

    @staticmethod
    def _i16(lo: int, hi: int) -> int:
        v = (hi << 8) | lo
        return v - 0x10000 if v & 0x8000 else v

    @staticmethod
    def _quat_from_euler_deg(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
        # ZYX convention: yaw (Z), pitch (Y), roll (X)
        import math
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

    def _update_quat(self, sample: QuaternionSample) -> None:
        with self._lock:
            self._latest_quat = sample

    def _run(self) -> None:
        assert self._ser is not None
        ser = self._ser

        # Read byte-by-byte sync on 0x55; robust to misalignment
        while not self._stop_evt.is_set():
            try:
                b = ser.read(1)
                if not b:
                    continue
                if b[0] != self.HEAD:
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

                # 0x59: quaternion (common on WT901/JY901 firmwares)
                # payload: q0,q1,q2,q3 as int16 (little endian), scaled by 1/32768
                if ftype == 0x59:
                    q0 = self._i16(payload[0], payload[1]) / 32768.0
                    q1 = self._i16(payload[2], payload[3]) / 32768.0
                    q2 = self._i16(payload[4], payload[5]) / 32768.0
                    q3 = self._i16(payload[6], payload[7]) / 32768.0
                    self._update_quat(QuaternionSample(w=q0, x=q1, y=q2, z=q3, timestamp=now, source="quat"))
                    continue

                # 0x53: angles (roll,pitch,yaw) in degrees, scaled: int16 / 32768 * 180
                if ftype == 0x53:
                    roll = self._i16(payload[0], payload[1]) / 32768.0 * 180.0
                    pitch = self._i16(payload[2], payload[3]) / 32768.0 * 180.0
                    yaw = self._i16(payload[4], payload[5]) / 32768.0 * 180.0
                    with self._lock:
                        self._latest_angles_deg = (roll, pitch, yaw, now)

                    # If we don't get native quaternion frames, provide quaternion derived from euler
                    # (still useful for quick integration; yaw may drift depending on firmware/calibration)
                    w, x, yq, z = self._quat_from_euler_deg(roll, pitch, yaw)
                    self._update_quat(QuaternionSample(w=w, x=x, y=yq, z=z, timestamp=now, source="euler"))
                    continue

                # ignore other frame types (0x51 accel, 0x52 gyro, etc.)
            except (serial.SerialException, OSError):
                # If cable is unplugged etc., don't spin at 100% CPU
                time.sleep(0.1)
            except Exception:
                # Keep the reader alive even if parsing hiccups
                with self._stats_lock:
                    self._frames_bad += 1
                time.sleep(0.01)


if __name__ == "__main__":
    imu = JY901IMU("/dev/ttyAMA0", 9600)
    imu.start()
    try:
        while True:
            q = imu.get_quaternion()
            if q is not None:
                print(f"{q.timestamp:.3f}  source={q.source:5s}  "
                      f"wxyz=({q.w:+.4f}, {q.x:+.4f}, {q.y:+.4f}, {q.z:+.4f})")
            time.sleep(0.02)
    finally:
        imu.stop()
