from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import time
import math
from pathlib import Path
import importlib.util
from importlib.machinery import SourceFileLoader


DEFAULT_BNO085_REPORTS: tuple[str, ...] = (
    "rotation_vector",
    "accelerometer",
    "gyroscope",
    "linear_acceleration",
    "gravity",
)


@dataclass
class IMUState:
    timestamp_s: float
    quaternion_xyzw: Optional[Tuple[float, float, float, float]] = None
    acceleration_mps2: Optional[Tuple[float, float, float]] = None
    gyro_rads: Optional[Tuple[float, float, float]] = None
    linear_velocity_mps: Optional[Tuple[float, float, float]] = None
    linear_acceleration_mps2: Optional[Tuple[float, float, float]] = None
    gravity_mps2: Optional[Tuple[float, float, float]] = None
    calibration: Optional[Any] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_s": float(self.timestamp_s),
            "quaternion_xyzw": self.quaternion_xyzw,
            "acceleration_mps2": self.acceleration_mps2,
            "gyro_rads": self.gyro_rads,
            "linear_velocity_mps": self.linear_velocity_mps,
            "linear_acceleration_mps2": self.linear_acceleration_mps2,
            "gravity_mps2": self.gravity_mps2,
            "calibration": self.calibration,
        }


class BNO085IMU:
    """
    Lightweight BNO085 wrapper for Raspberry Pi using Adafruit's CircuitPython driver.
    """

    def __init__(
        self,
        *,
        address: int = 0x4A,
        reports: tuple[str, ...] = DEFAULT_BNO085_REPORTS,
    ) -> None:
        self.address = int(address)
        self.reports = tuple(reports)
        self._sensor = None
        self._import_error: Optional[Exception] = None
        self._init_sensor()

    def _init_sensor(self) -> None:
        try:
            import board
            import busio
            from adafruit_bno08x import (
                BNO_REPORT_ACCELEROMETER,
                BNO_REPORT_GYROSCOPE,
                BNO_REPORT_LINEAR_ACCELERATION,
                BNO_REPORT_GRAVITY,
                BNO_REPORT_ROTATION_VECTOR,
            )
            from adafruit_bno08x.i2c import BNO08X_I2C
        except Exception as exc:
            self._import_error = exc
            return

        i2c = busio.I2C(board.SCL, board.SDA)
        sensor = BNO08X_I2C(i2c, address=self.address)

        report_map = {
            "rotation_vector": BNO_REPORT_ROTATION_VECTOR,
            "accelerometer": BNO_REPORT_ACCELEROMETER,
            "gyroscope": BNO_REPORT_GYROSCOPE,
            "linear_acceleration": BNO_REPORT_LINEAR_ACCELERATION,
            "gravity": BNO_REPORT_GRAVITY,
        }
        for report_name in self.reports:
            report = report_map.get(report_name)
            if report is None:
                continue
            sensor.enable_feature(report)

        self._sensor = sensor
        self._import_error = None

    @property
    def available(self) -> bool:
        return self._sensor is not None

    @property
    def last_error(self) -> Optional[str]:
        if self._import_error is not None:
            return f"{type(self._import_error).__name__}: {self._import_error}"
        return None

    def _read_attr(self, name: str) -> Optional[Tuple[float, ...]]:
        if self._sensor is None:
            return None
        try:
            value = getattr(self._sensor, name)
            if value is None:
                return None
            return tuple(float(v) for v in value)
        except Exception:
            return None

    def read(self) -> IMUState:
        calibration = None
        if self._sensor is not None and hasattr(self._sensor, "calibration_status"):
            try:
                calibration = self._sensor.calibration_status
            except Exception:
                calibration = None

        return IMUState(
            timestamp_s=time.time(),
            quaternion_xyzw=self._read_attr("quaternion"),
            acceleration_mps2=self._read_attr("acceleration"),
            gyro_rads=self._read_attr("gyro"),
            linear_acceleration_mps2=self._read_attr("linear_acceleration"),
            gravity_mps2=self._read_attr("gravity"),
            calibration=calibration,
        )

    def read_dict(self) -> Dict[str, Any]:
        state = self.read()
        out = state.as_dict()
        out["available"] = self.available
        out["error"] = self.last_error
        return out


class JY901UARTIMU:
    """
    JY901 UART adapter that exposes the same read/read_dict interface.
    """

    def __init__(
        self,
        *,
        port: str = "/dev/ttyAMA0",
        baudrate: int = 9600,
        timeout_s: float = 0.2,
        autostart: bool = True,
    ) -> None:
        self.port = str(port)
        self.baudrate = int(baudrate)
        self.timeout_s = float(timeout_s)
        self._imu = None
        self._import_error: Optional[Exception] = None
        try:
            from IMU_JY901 import JY901IMU

            self._imu = JY901IMU(port=self.port, baudrate=self.baudrate, timeout_s=self.timeout_s)
            if autostart:
                self.start()
        except Exception as exc:
            self._import_error = exc

    def start(self) -> None:
        if self._imu is not None:
            self._imu.start()

    def stop(self) -> None:
        if self._imu is not None:
            self._imu.stop()

    @property
    def available(self) -> bool:
        return self._imu is not None

    @property
    def last_error(self) -> Optional[str]:
        if self._import_error is not None:
            return f"{type(self._import_error).__name__}: {self._import_error}"
        return None

    def read(self) -> IMUState:
        if self._imu is None:
            return IMUState(timestamp_s=time.time())
        obs = self._imu.get_observation()
        quat = obs.quat
        ang = obs.ang_vel_rad_s
        lin_acc = obs.lin_acc_m_s2
        lin_vel = obs.lin_vel_m_s

        quaternion_xyzw = None
        calibration: Optional[Dict[str, Any]] = None
        if quat is not None:
            # JY901 exposes wxyz, controller expects xyzw.
            quaternion_xyzw = (float(quat.x), float(quat.y), float(quat.z), float(quat.w))
            calibration = {"source": str(quat.source), "stats": self._imu.get_stats()}
        else:
            calibration = {"stats": self._imu.get_stats()}

        gyro_rads = None
        if ang is not None:
            gyro_rads = (float(ang.x), float(ang.y), float(ang.z))

        linear_acceleration_mps2 = None
        if lin_acc is not None:
            linear_acceleration_mps2 = (float(lin_acc.x), float(lin_acc.y), float(lin_acc.z))

        linear_velocity_mps = None
        if lin_vel is not None:
            linear_velocity_mps = (float(lin_vel.x), float(lin_vel.y), float(lin_vel.z))

        return IMUState(
            timestamp_s=float(obs.timestamp),
            quaternion_xyzw=quaternion_xyzw,
            gyro_rads=gyro_rads,
            linear_velocity_mps=linear_velocity_mps,
            linear_acceleration_mps2=linear_acceleration_mps2,
            calibration=calibration,
        )

    def read_dict(self) -> Dict[str, Any]:
        state = self.read()
        out = state.as_dict()
        out["available"] = self.available
        out["error"] = self.last_error
        return out


class BNO055I2CIMU:
    """
    BNO055 I2C adapter that exposes the same read/read_dict interface.
    Reuses the local IMU_BNO055 implementation file.
    """

    def __init__(
        self,
        *,
        i2c_bus: int = 1,
        address: int = 0x28,
        rate_hz: float = 100.0,
        vel_leak: float = 0.0,
        max_dt_s: float = 0.2,
        vel_kf_accel_proc_std_m_s2: float = 3.0,
        vel_kf_zero_meas_std_m_s: float = 0.15,
        opr_mode: int = 0x0C,
        reset_on_start: bool = False,
        i2c_retries: int = 3,
        autostart: bool = True,
        frame_yaw_deg: float = 0.0,
    ) -> None:
        self._imu = None
        self._import_error: Optional[Exception] = None
        self._frame_yaw_rad = math.radians(float(frame_yaw_deg))
        try:
            BNO055IMU = self._load_impl_class()
            self._imu = BNO055IMU(
                i2c_bus=int(i2c_bus),
                address=int(address),
                rate_hz=float(rate_hz),
                vel_leak=float(vel_leak),
                max_dt_s=float(max_dt_s),
                vel_kf_accel_proc_std_m_s2=float(vel_kf_accel_proc_std_m_s2),
                vel_kf_zero_meas_std_m_s=float(vel_kf_zero_meas_std_m_s),
                opr_mode=int(opr_mode),
                reset_on_start=bool(reset_on_start),
                i2c_retries=int(i2c_retries),
            )
            if autostart:
                self.start()
        except Exception as exc:
            self._import_error = exc

    @staticmethod
    def _load_impl_class():
        impl_path = Path(__file__).with_name("IMU_BNO055")
        if not impl_path.exists():
            raise FileNotFoundError(f"Missing BNO055 implementation file: {impl_path}")
        # IMU_BNO055 is an extensionless Python source file.
        loader = SourceFileLoader("imu_bno055_impl", str(impl_path))
        spec = importlib.util.spec_from_loader("imu_bno055_impl", loader)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load module from {impl_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = getattr(module, "BNO055IMU", None)
        if cls is None:
            raise RuntimeError("BNO055IMU class not found in IMU_BNO055")
        return cls

    def start(self) -> None:
        if self._imu is not None:
            self._imu.start()

    def stop(self) -> None:
        if self._imu is not None:
            self._imu.stop()

    @property
    def available(self) -> bool:
        return self._imu is not None

    @property
    def last_error(self) -> Optional[str]:
        if self._import_error is not None:
            return f"{type(self._import_error).__name__}: {self._import_error}"
        return None

    def _rotate_vec_sensor_to_robot(self, vec: Optional[Tuple[float, float, float]]) -> Optional[Tuple[float, float, float]]:
        if vec is None:
            return None
        x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
        if abs(self._frame_yaw_rad) < 1e-12:
            return (x, y, z)
        c = float(math.cos(self._frame_yaw_rad))
        s = float(math.sin(self._frame_yaw_rad))
        # Sensor -> robot frame rotation around +Z by frame_yaw_deg.
        xr = c * x - s * y
        yr = s * x + c * y
        return (xr, yr, z)

    @staticmethod
    def _quat_xyzw_mul(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )

    @staticmethod
    def _quat_xyzw_conj(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        x, y, z, w = q
        return (-x, -y, -z, w)

    def _map_quaternion_sensor_to_robot(
        self, q_xyzw_sensor: Optional[Tuple[float, float, float, float]]
    ) -> Optional[Tuple[float, float, float, float]]:
        if q_xyzw_sensor is None:
            return None
        if abs(self._frame_yaw_rad) < 1e-12:
            return q_xyzw_sensor
        half = 0.5 * self._frame_yaw_rad
        # q_rs: sensor->robot rotation around +Z.
        q_rs = (0.0, 0.0, float(math.sin(half)), float(math.cos(half)))
        # q_sr is inverse of q_rs.
        q_sr = self._quat_xyzw_conj(q_rs)
        # q_ws (sensor->world) -> q_wr (robot->world): q_wr = q_ws * q_sr
        return self._quat_xyzw_mul(q_xyzw_sensor, q_sr)

    def read(self) -> IMUState:
        if self._imu is None:
            return IMUState(timestamp_s=time.time())
        obs = self._imu.get_observation()
        quat = obs.quat
        ang = obs.ang_vel_rad_s
        lin_acc = obs.lin_acc_m_s2
        lin_vel = obs.lin_vel_m_s

        quaternion_xyzw = None
        if quat is not None:
            # BNO055 implementation exposes wxyz; controller expects xyzw.
            q_sensor = (float(quat.x), float(quat.y), float(quat.z), float(quat.w))
            quaternion_xyzw = self._map_quaternion_sensor_to_robot(q_sensor)

        gyro_rads = None
        if ang is not None:
            gyro_rads = self._rotate_vec_sensor_to_robot((float(ang.x), float(ang.y), float(ang.z)))

        linear_acceleration_mps2 = None
        if lin_acc is not None:
            linear_acceleration_mps2 = self._rotate_vec_sensor_to_robot(
                (float(lin_acc.x), float(lin_acc.y), float(lin_acc.z))
            )

        linear_velocity_mps = None
        if lin_vel is not None:
            linear_velocity_mps = self._rotate_vec_sensor_to_robot((float(lin_vel.x), float(lin_vel.y), float(lin_vel.z)))

        calibration: Optional[Dict[str, Any]] = None
        if hasattr(self._imu, "get_stats"):
            try:
                calibration = {"stats": self._imu.get_stats()}
            except Exception:
                calibration = None

        return IMUState(
            timestamp_s=float(obs.timestamp),
            quaternion_xyzw=quaternion_xyzw,
            gyro_rads=gyro_rads,
            linear_velocity_mps=linear_velocity_mps,
            linear_acceleration_mps2=linear_acceleration_mps2,
            calibration=calibration,
        )

    def read_dict(self) -> Dict[str, Any]:
        state = self.read()
        out = state.as_dict()
        out["available"] = self.available
        out["error"] = self.last_error
        out["frame_yaw_deg"] = float(math.degrees(self._frame_yaw_rad))
        return out


class MockIMU:
    def __init__(
        self,
        *,
        quaternion_xyzw: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        acceleration_mps2: Tuple[float, float, float] = (0.0, 0.0, 9.81),
        gyro_rads: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        linear_velocity_mps: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        linear_acceleration_mps2: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        gravity_mps2: Tuple[float, float, float] = (0.0, 0.0, -9.81),
    ) -> None:
        self.set_state(
            quaternion_xyzw=quaternion_xyzw,
            acceleration_mps2=acceleration_mps2,
            gyro_rads=gyro_rads,
            linear_velocity_mps=linear_velocity_mps,
            linear_acceleration_mps2=linear_acceleration_mps2,
            gravity_mps2=gravity_mps2,
        )

    def set_state(
        self,
        *,
        quaternion_xyzw: Optional[Tuple[float, float, float, float]] = None,
        acceleration_mps2: Optional[Tuple[float, float, float]] = None,
        gyro_rads: Optional[Tuple[float, float, float]] = None,
        linear_velocity_mps: Optional[Tuple[float, float, float]] = None,
        linear_acceleration_mps2: Optional[Tuple[float, float, float]] = None,
        gravity_mps2: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        if quaternion_xyzw is not None:
            self.quaternion_xyzw = tuple(float(v) for v in quaternion_xyzw)
        if acceleration_mps2 is not None:
            self.acceleration_mps2 = tuple(float(v) for v in acceleration_mps2)
        if gyro_rads is not None:
            self.gyro_rads = tuple(float(v) for v in gyro_rads)
        if linear_velocity_mps is not None:
            self.linear_velocity_mps = tuple(float(v) for v in linear_velocity_mps)
        if linear_acceleration_mps2 is not None:
            self.linear_acceleration_mps2 = tuple(float(v) for v in linear_acceleration_mps2)
        if gravity_mps2 is not None:
            self.gravity_mps2 = tuple(float(v) for v in gravity_mps2)

    def read(self) -> IMUState:
        return IMUState(
            timestamp_s=time.time(),
            quaternion_xyzw=self.quaternion_xyzw,
            acceleration_mps2=self.acceleration_mps2,
            gyro_rads=self.gyro_rads,
            linear_velocity_mps=self.linear_velocity_mps,
            linear_acceleration_mps2=self.linear_acceleration_mps2,
            gravity_mps2=self.gravity_mps2,
            calibration={"mock": True},
        )

    def read_dict(self) -> Dict[str, Any]:
        out = self.read().as_dict()
        out["available"] = True
        out["error"] = None
        out["mock"] = True
        return out


class IMU:
    """
    Parent IMU selector used by LeRobot.
    - sensor: "bno085", "bno055", or "jy901"
    - mock=True forces mock values regardless of selected sensor
    """

    def __init__(self, *, sensor: str = "bno085", mock: bool = False, **sensor_kwargs: Any) -> None:
        self.sensor = str(sensor).strip().lower()
        self.mock = bool(mock)
        self._sensor_kwargs: Dict[str, Any] = dict(sensor_kwargs)
        self._mock_backend = MockIMU()
        self._sensor_backend: Optional[Any] = None
        self._build_sensor_backend()

    def _build_sensor_backend(self) -> None:
        if self.sensor == "bno085":
            address = int(self._sensor_kwargs.get("address", 0x4A))
            reports = tuple(self._sensor_kwargs.get("reports", DEFAULT_BNO085_REPORTS))
            self._sensor_backend = BNO085IMU(address=address, reports=reports)
            return
        if self.sensor == "jy901":
            self._sensor_backend = JY901UARTIMU(
                port=str(self._sensor_kwargs.get("port", "/dev/ttyAMA0")),
                baudrate=int(self._sensor_kwargs.get("baudrate", 9600)),
                timeout_s=float(self._sensor_kwargs.get("timeout_s", 0.2)),
                autostart=bool(self._sensor_kwargs.get("autostart", True)),
            )
            return
        if self.sensor == "bno055":
            self._sensor_backend = BNO055I2CIMU(
                i2c_bus=int(self._sensor_kwargs.get("i2c_bus", 1)),
                address=int(self._sensor_kwargs.get("address", 0x28)),
                rate_hz=float(self._sensor_kwargs.get("rate_hz", 100.0)),
                vel_leak=float(self._sensor_kwargs.get("vel_leak", 0.0)),
                max_dt_s=float(self._sensor_kwargs.get("max_dt_s", 0.2)),
                vel_kf_accel_proc_std_m_s2=float(self._sensor_kwargs.get("vel_kf_accel_proc_std_m_s2", 3.0)),
                vel_kf_zero_meas_std_m_s=float(self._sensor_kwargs.get("vel_kf_zero_meas_std_m_s", 0.15)),
                opr_mode=int(self._sensor_kwargs.get("opr_mode", 0x0C)),
                reset_on_start=bool(self._sensor_kwargs.get("reset_on_start", False)),
                i2c_retries=int(self._sensor_kwargs.get("i2c_retries", 3)),
                autostart=bool(self._sensor_kwargs.get("autostart", True)),
                frame_yaw_deg=float(self._sensor_kwargs.get("frame_yaw_deg", 0.0)),
            )
            return
        raise ValueError("Unsupported IMU sensor. Use 'bno085', 'bno055', or 'jy901'.")

    def use_sensor(self, sensor: str, **sensor_kwargs: Any) -> None:
        self.stop()
        self.sensor = str(sensor).strip().lower()
        self._sensor_kwargs = dict(sensor_kwargs)
        self._build_sensor_backend()

    def set_mock(self, enabled: bool, **mock_state: Any) -> None:
        self.mock = bool(enabled)
        if mock_state:
            self._mock_backend.set_state(**mock_state)

    def start(self) -> None:
        backend = self._sensor_backend
        if backend is not None and hasattr(backend, "start"):
            backend.start()

    def stop(self) -> None:
        backend = self._sensor_backend
        if backend is not None and hasattr(backend, "stop"):
            backend.stop()

    def read(self) -> IMUState:
        if self.mock:
            return self._mock_backend.read()
        if self._sensor_backend is None:
            return IMUState(timestamp_s=time.time())
        if hasattr(self._sensor_backend, "read"):
            return self._sensor_backend.read()
        return IMUState(timestamp_s=time.time())

    def read_dict(self) -> Dict[str, Any]:
        if self.mock:
            out = self._mock_backend.read_dict()
            out["sensor"] = self.sensor
            return out
        if self._sensor_backend is None:
            return {"timestamp_s": time.time(), "sensor": self.sensor, "available": False, "error": "No IMU backend"}
        if hasattr(self._sensor_backend, "read_dict"):
            out = self._sensor_backend.read_dict()
        elif hasattr(self._sensor_backend, "read"):
            raw = self._sensor_backend.read()
            out = raw.as_dict() if isinstance(raw, IMUState) else (raw if isinstance(raw, dict) else {"data": raw})
            out.setdefault("available", True)
            out.setdefault("error", None)
        else:
            out = {"timestamp_s": time.time(), "available": False, "error": "Backend has no read/read_dict"}
        out["sensor"] = self.sensor
        return out


# Backward compatibility with existing imports in bipedal_robot.py.
BNO085I2CWrapper = BNO085IMU
