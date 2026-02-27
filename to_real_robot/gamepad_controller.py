from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import argparse
import select
import threading
import time


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


TRAINING_MAX_YAW_RATE = 0.5


@dataclass
class AxisState:
    value: float = 0.0
    raw: int = 0
    min_raw: int = -32768
    max_raw: int = 32767
    center_raw: int = 0

    def normalize(self, raw: int) -> float:
        self.raw = int(raw)
        left_span = max(1, self.center_raw - self.min_raw)
        right_span = max(1, self.max_raw - self.center_raw)
        if self.raw >= self.center_raw:
            return float(self.raw - self.center_raw) / float(right_span)
        return float(self.raw - self.center_raw) / float(left_span)


@dataclass
class GamepadController:
    """
    Linux Bluetooth/USB gamepad reader for RL command twist.
    Expected backend: `evdev` package.
    """

    name_substring: str = "8bitdo"
    deadzone: float = 0.10
    max_lin_x: float = 0.6
    max_lin_y: float = 0.4
    max_yaw_rate: float = TRAINING_MAX_YAW_RATE
    invert_lin_x: bool = True
    invert_lin_y: bool = False
    invert_yaw: bool = False
    use_right_stick_for_yaw: bool = True
    yaw_axis_code: Optional[int] = None
    _device: Optional[Any] = field(default=None, init=False)
    _ecodes: Optional[Any] = field(default=None, init=False)
    _running: bool = field(default=False, init=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _twist: Tuple[float, float, float] = field(default=(0.0, 0.0, 0.0), init=False)
    _last_update_s: float = field(default=0.0, init=False)
    _warn_last_s: float = field(default=0.0, init=False)
    _warn_count: int = field(default=0, init=False)
    _axes: Dict[str, AxisState] = field(default_factory=dict, init=False)
    _axis_codes: Dict[str, int] = field(default_factory=dict, init=False)
    _raw_abs_latest: Dict[int, int] = field(default_factory=dict, init=False)

    def _warn(self, msg: str) -> None:
        now = time.time()
        self._warn_count += 1
        if (now - self._warn_last_s) < 1.0:
            return
        self._warn_last_s = now
        print(f"[GamepadController][WARN] {msg} (count={self._warn_count})")

    @staticmethod
    def list_devices() -> List[Tuple[str, str]]:
        try:
            from evdev import InputDevice, list_devices  # type: ignore
        except Exception:
            return []
        out: List[Tuple[str, str]] = []
        for path in list_devices():
            try:
                dev = InputDevice(path)
                out.append((path, str(dev.name)))
            except Exception:
                continue
        return out

    def connect(self, device_path: Optional[str] = None) -> None:
        try:
            from evdev import InputDevice, ecodes, list_devices  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "evdev is required for gamepad support. Install with `pip install evdev`."
            ) from exc

        chosen = device_path
        if chosen is None:
            needle = self.name_substring.strip().lower()
            for path in list_devices():
                try:
                    dev = InputDevice(path)
                except Exception:
                    continue
                name = str(dev.name).lower()
                if needle in name:
                    chosen = path
                    break
        if chosen is None:
            devices = ", ".join([f"{p}:{n}" for p, n in self.list_devices()]) or "<none>"
            raise RuntimeError(f"No matching gamepad found for '{self.name_substring}'. devices={devices}")

        dev = InputDevice(chosen)
        self._device = dev
        self._ecodes = ecodes
        self._setup_axis_ranges()
        print(f"[GamepadController] connected path={chosen} name={dev.name}")

    def _setup_axis_ranges(self) -> None:
        dev = self._device
        ecodes = self._ecodes
        if dev is None or ecodes is None:
            return

        # 8BitDo mappings vary by mode/OS:
        # right-stick horizontal can show up as ABS_RX / ABS_Z / ABS_RZ (and rarely others).
        axis_map = {
            "lx": [ecodes.ABS_X],  # left stick horizontal
            "ly": [ecodes.ABS_Y],  # left stick vertical
            "rx": [ecodes.ABS_RX, ecodes.ABS_Z, ecodes.ABS_RZ, ecodes.ABS_RY, ecodes.ABS_HAT0X],  # yaw source
        }
        self._axes = {}
        self._axis_codes = {}
        for key, candidates in axis_map.items():
            if key == "rx" and self.yaw_axis_code is not None:
                candidates = [int(self.yaw_axis_code)] + [c for c in candidates if int(c) != int(self.yaw_axis_code)]
            code = None
            best_score = -1
            for candidate in candidates:
                try:
                    info = dev.absinfo(candidate)
                except Exception:
                    info = None
                if info is not None:
                    # Prefer full-range signed axes (sticks) over triggers/HAT.
                    lo = int(info.min)
                    hi = int(info.max)
                    signed = 1 if (lo < 0 < hi) else 0
                    span = max(1, hi - lo)
                    # Penalize tiny-span digital HAT-style axes.
                    if span <= 4:
                        score = -100_000 + span
                    else:
                        score = signed * 1_000 + span
                    if score > best_score:
                        best_score = score
                        code = int(candidate)
            if code is None:
                continue
            st = AxisState()
            info = dev.absinfo(code)
            if info is not None:
                st.min_raw = int(info.min)
                st.max_raw = int(info.max)
                st.center_raw = int((info.min + info.max) // 2)
            self._axes[key] = st
            self._axis_codes[key] = code

    def start(self) -> None:
        if self._device is None:
            self.connect()
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
        self._device = None

    def _apply_deadzone(self, v: float) -> float:
        d = abs(float(self.deadzone))
        x = float(v)
        if abs(x) <= d:
            return 0.0
        scaled = (abs(x) - d) / max(1e-6, 1.0 - d)
        return scaled if x > 0 else -scaled

    def _update_twist_from_axes(self) -> None:
        lx = self._apply_deadzone(self._axes.get("lx", AxisState()).value)
        ly = self._apply_deadzone(self._axes.get("ly", AxisState()).value)
        rx = self._apply_deadzone(self._axes.get("rx", AxisState()).value)

        lin_x = (-ly if self.invert_lin_x else ly) * float(self.max_lin_x)
        lin_y = (-lx if self.invert_lin_y else lx) * float(self.max_lin_y)
        yaw_scale = min(abs(float(self.max_yaw_rate)), TRAINING_MAX_YAW_RATE)
        yaw_input = rx if self.use_right_stick_for_yaw else 0.0
        yaw = (-yaw_input if self.invert_yaw else yaw_input) * yaw_scale
        twist = (
            _clamp(lin_x, -self.max_lin_x, self.max_lin_x),
            _clamp(lin_y, -self.max_lin_y, self.max_lin_y),
            _clamp(yaw, -TRAINING_MAX_YAW_RATE, TRAINING_MAX_YAW_RATE),
        )
        with self._lock:
            self._twist = twist
            self._last_update_s = time.time()

    def _loop(self) -> None:
        dev = self._device
        ecodes = self._ecodes
        if dev is None or ecodes is None:
            return
        while self._running:
            try:
                ready, _, _ = select.select([dev.fd], [], [], 0.1)
                if not ready:
                    continue
                for event in dev.read():
                    if event.type != ecodes.EV_ABS:
                        continue
                    self._raw_abs_latest[int(event.code)] = int(event.value)
                    if event.code == self._axis_codes.get("lx") and "lx" in self._axes:
                        self._axes["lx"].value = self._axes["lx"].normalize(int(event.value))
                    elif event.code == self._axis_codes.get("ly") and "ly" in self._axes:
                        self._axes["ly"].value = self._axes["ly"].normalize(int(event.value))
                    elif event.code == self._axis_codes.get("rx") and "rx" in self._axes:
                        self._axes["rx"].value = self._axes["rx"].normalize(int(event.value))
                    self._update_twist_from_axes()
            except Exception as exc:
                self._warn(f"read loop exception: {type(exc).__name__}: {exc}")
                time.sleep(0.1)

    def get_command_twist(self) -> Tuple[float, float, float]:
        with self._lock:
            return self._twist

    def set_yaw_axis_code(self, axis_code: int) -> None:
        self.yaw_axis_code = int(axis_code)
        if self._device is not None and self._ecodes is not None:
            self._setup_axis_ranges()
            self._warn(f"yaw axis set to code={self.yaw_axis_code}")

    def get_debug_state(self) -> Dict[str, Any]:
        with self._lock:
            twist = self._twist
            last_update_s = self._last_update_s
        return {
            "running": bool(self._running),
            "device_connected": bool(self._device is not None),
            "device_name": getattr(self._device, "name", None) if self._device is not None else None,
            "command_twist": [float(twist[0]), float(twist[1]), float(twist[2])],
            "last_update_s": float(last_update_s),
            "warn_count": int(self._warn_count),
            "deadzone": float(self.deadzone),
            "max_lin_x": float(self.max_lin_x),
            "max_lin_y": float(self.max_lin_y),
            "max_yaw_rate": float(self.max_yaw_rate),
            "axis_codes": dict(self._axis_codes),
            "raw_abs_latest": {int(k): int(v) for k, v in self._raw_abs_latest.items()},
        }


def _main() -> None:
    parser = argparse.ArgumentParser(description="Quick 8BitDo gamepad check.")
    parser.add_argument("--list", action="store_true", help="List input devices and exit.")
    parser.add_argument("--name", type=str, default="8bitdo", help="Case-insensitive gamepad name substring.")
    parser.add_argument("--path", type=str, default=None, help="Explicit /dev/input/eventX path.")
    parser.add_argument("--seconds", type=float, default=10.0, help="Run duration for test mode.")
    args = parser.parse_args()

    if args.list:
        devices = GamepadController.list_devices()
        if not devices:
            print("No input devices found, or evdev is not installed.")
            return
        for path, name in devices:
            print(f"{path}: {name}")
        return

    ctrl = GamepadController(name_substring=args.name)
    ctrl.connect(device_path=args.path)
    ctrl.start()
    print("[GamepadController] move sticks now. Ctrl+C to stop.")
    t0 = time.time()
    try:
        while (time.time() - t0) < float(args.seconds):
            lin_x, lin_y, yaw = ctrl.get_command_twist()
            print(f"twist lin_x={lin_x:+.3f} lin_y={lin_y:+.3f} yaw={yaw:+.3f}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.stop()


if __name__ == "__main__":
    _main()
