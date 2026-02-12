from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional

import time
import can
import numpy as np

from leg_test.mit import MotorState, float_to_uint, uint_to_float
from root_constant import MOTORS, CAN_CMD_CLEAR_FAULT


def _pack_state_frame(
    *,
    motor_id: int,
    position_deg: float,
    velocity_deg_s: float,
    torque_nm: float,
    temp_c: float,
    pmax: float,
    vmax: float,
    tmax: float,
) -> bytes:
    pos_rad = float(np.radians(position_deg))
    vel_rad = float(np.radians(velocity_deg_s))

    q_uint = float_to_uint(pos_rad, -pmax, pmax, 16)
    dq_uint = float_to_uint(vel_rad, -vmax, vmax, 12)
    tau_uint = float_to_uint(float(torque_nm), -tmax, tmax, 12)
    t_mos = int(max(0, min(65535, round(float(temp_c) * 10.0))))

    data = [0] * 8
    data[0] = int(motor_id) & 0xFF
    data[1] = (q_uint >> 8) & 0xFF
    data[2] = q_uint & 0xFF
    data[3] = (dq_uint >> 4) & 0xFF
    data[4] = ((dq_uint & 0x0F) << 4) | ((tau_uint >> 8) & 0x0F)
    data[5] = tau_uint & 0xFF
    data[6] = (t_mos >> 8) & 0xFF
    data[7] = t_mos & 0xFF
    return bytes(data)


class MockBus:
    """
    Basic CAN mock for MIT motors.
    Behavior:
      - MIT command frame: instant motor (zero inertia), state jumps to requested pos/vel/tau.
      - CLEAR_FAULT cmd: returns zero state.
      - Other command frames [FF..FF, cmd]: returns zero state + warning.
    """

    def __init__(self, *, default_temp_c: float = 30.0):
        self._rx_queue: Deque[can.Message] = deque()
        self._state: Dict[int, MotorState] = {}
        self._default_temp_c = float(default_temp_c)

    def send(self, msg: can.Message) -> None:
        data = bytes(msg.data)
        if len(data) < 8:
            return
        motor_id = int(msg.arbitration_id)
        spec = MOTORS.get(motor_id)
        pmax = float(spec.pmax_rad) if spec is not None else 12.57
        vmax = float(spec.vmax_rad_s) if spec is not None else 33.0
        tmax = float(spec.tmax_nm) if spec is not None else 20.0

        is_cmd_ff = (len(data) == 8 and all(b == 0xFF for b in data[:7]))
        if is_cmd_ff:
            cmd = int(data[7])
            # CLEAR_FAULT -> explicit zero state.
            if cmd == int(CAN_CMD_CLEAR_FAULT):
                st = MotorState(position_deg=0.0, velocity_deg_s=0.0, torque_nm=0.0, temp_mos_c=self._default_temp_c, stamp=time.time())
                self._state[motor_id] = st
                self._enqueue_state(motor_id, st, pmax=pmax, vmax=vmax, tmax=tmax)
                return

            print(f"[WARN] MockBus: unusual FF command 0x{cmd:02X} on m{motor_id}, returning zero state.")
            st = MotorState(position_deg=0.0, velocity_deg_s=0.0, torque_nm=0.0, temp_mos_c=self._default_temp_c, stamp=time.time())
            self._state[motor_id] = st
            self._enqueue_state(motor_id, st, pmax=pmax, vmax=vmax, tmax=tmax)
            return

        # MIT control frame: decode requested pos/vel/tau and mirror as state.
        q_uint = (data[0] << 8) | data[1]
        dq_uint = (data[2] << 4) | (data[3] >> 4)
        tau_uint = ((data[6] & 0x0F) << 8) | data[7]
        pos_deg = float(np.degrees(uint_to_float(q_uint, -pmax, pmax, 16)))
        vel_deg_s = float(np.degrees(uint_to_float(dq_uint, -vmax, vmax, 12)))
        tau_nm = float(uint_to_float(tau_uint, -tmax, tmax, 12))

        prev = self._state.get(motor_id, MotorState(temp_mos_c=self._default_temp_c))
        st = MotorState(
            position_deg=pos_deg,
            velocity_deg_s=vel_deg_s,
            torque_nm=tau_nm,
            temp_mos_c=prev.temp_mos_c if prev.temp_mos_c > 0 else self._default_temp_c,
            stamp=time.time(),
        )
        self._state[motor_id] = st
        self._enqueue_state(motor_id, st, pmax=pmax, vmax=vmax, tmax=tmax)

    def recv(self, timeout: Optional[float] = None) -> Optional[can.Message]:
        # Match python-can semantics:
        # - timeout=None: block indefinitely until one message is available
        # - timeout<=0: non-blocking poll, return one message or None
        # - timeout>0: wait up to timeout for one message
        if timeout is None:
            while True:
                if self._rx_queue:
                    return self._rx_queue.popleft()
                time.sleep(0.0001)

        timeout_s = float(timeout)
        if timeout_s <= 0.0:
            return self._rx_queue.popleft() if self._rx_queue else None

        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            if self._rx_queue:
                return self._rx_queue.popleft()
            time.sleep(0.0001)
        return None

    def shutdown(self) -> None:
        self._rx_queue.clear()

    def _enqueue_state(self, motor_id: int, st: MotorState, *, pmax: float, vmax: float, tmax: float) -> None:
        payload = _pack_state_frame(
            motor_id=motor_id,
            position_deg=float(st.position_deg),
            velocity_deg_s=float(st.velocity_deg_s),
            torque_nm=float(st.torque_nm),
            temp_c=float(st.temp_mos_c),
            pmax=pmax,
            vmax=vmax,
            tmax=tmax,
        )
        self._rx_queue.append(
            can.Message(
                arbitration_id=int(motor_id),
                data=payload,
                is_extended_id=False,
            )
        )
