from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import time


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
    linear_acceleration_mps2: Optional[Tuple[float, float, float]] = None
    gravity_mps2: Optional[Tuple[float, float, float]] = None
    calibration: Optional[Any] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_s": float(self.timestamp_s),
            "quaternion_xyzw": self.quaternion_xyzw,
            "acceleration_mps2": self.acceleration_mps2,
            "gyro_rads": self.gyro_rads,
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
        sample = self._imu.get_quaternion()
        if sample is None:
            return IMUState(timestamp_s=time.time())
        # JY901 exposes wxyz, controller expects xyzw.
        return IMUState(
            timestamp_s=float(sample.timestamp),
            quaternion_xyzw=(float(sample.x), float(sample.y), float(sample.z), float(sample.w)),
            calibration={"source": str(sample.source), "stats": self._imu.get_stats()},
        )

    def read_dict(self) -> Dict[str, Any]:
        state = self.read()
        out = state.as_dict()
        out["available"] = self.available
        out["error"] = self.last_error
        return out


class MockIMU:
    def __init__(
        self,
        *,
        quaternion_xyzw: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        acceleration_mps2: Tuple[float, float, float] = (0.0, 0.0, 9.81),
        gyro_rads: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        linear_acceleration_mps2: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        gravity_mps2: Tuple[float, float, float] = (0.0, 0.0, -9.81),
    ) -> None:
        self.set_state(
            quaternion_xyzw=quaternion_xyzw,
            acceleration_mps2=acceleration_mps2,
            gyro_rads=gyro_rads,
            linear_acceleration_mps2=linear_acceleration_mps2,
            gravity_mps2=gravity_mps2,
        )

    def set_state(
        self,
        *,
        quaternion_xyzw: Optional[Tuple[float, float, float, float]] = None,
        acceleration_mps2: Optional[Tuple[float, float, float]] = None,
        gyro_rads: Optional[Tuple[float, float, float]] = None,
        linear_acceleration_mps2: Optional[Tuple[float, float, float]] = None,
        gravity_mps2: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        if quaternion_xyzw is not None:
            self.quaternion_xyzw = tuple(float(v) for v in quaternion_xyzw)
        if acceleration_mps2 is not None:
            self.acceleration_mps2 = tuple(float(v) for v in acceleration_mps2)
        if gyro_rads is not None:
            self.gyro_rads = tuple(float(v) for v in gyro_rads)
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
    - sensor: "bno085" or "jy901"
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
        raise ValueError("Unsupported IMU sensor. Use 'bno085' or 'jy901'.")

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
