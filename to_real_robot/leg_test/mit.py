# leg_test/mit.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    x = max(x_min, min(x_max, x))
    span = x_max - x_min
    data_norm = (x - x_min) / span if span > 0 else 0.0
    return int(data_norm * ((1 << bits) - 1))


def uint_to_float(x: int, x_min: float, x_max: float, bits: int) -> float:
    span = x_max - x_min
    data_norm = float(x) / ((1 << bits) - 1)
    return data_norm * span + x_min


@dataclass
class MotorState:
    position_deg: float = 0.0
    velocity_deg_s: float = 0.0
    torque_nm: float = 0.0
    temp_mos_c: float = 0.0
    stamp: float = 0.0  # time.time()


def decode_state_frame(data: bytes, *, pmax: float, vmax: float, tmax: float) -> tuple[int, MotorState]:
    """
    Returns (motor_id, MotorState).
    """
    if len(data) < 8:
        raise ValueError(f"Expected 8 bytes, got {len(data)}")

    motor_id = int(data[0])
    q_uint = (data[1] << 8) | data[2]
    dq_uint = (data[3] << 4) | (data[4] >> 4)
    tau_uint = ((data[4] & 0x0F) << 8) | data[5]
    t_mos = (data[6] << 8) | data[7]

    pos_rad = uint_to_float(q_uint,  -pmax, pmax, 16)
    vel_rad = uint_to_float(dq_uint, -vmax, vmax, 12)
    tau_nm  = uint_to_float(tau_uint, -tmax, tmax, 12)

    st = MotorState(
        position_deg=float(np.degrees(pos_rad)),
        velocity_deg_s=float(np.degrees(vel_rad)),
        torque_nm=float(tau_nm),
        temp_mos_c=float(t_mos) / 10.0,
    )
    return motor_id, st


def pack_mit_command(
    *,
    position_deg: float,
    velocity_deg_s: float,
    kp: float,
    kd: float,
    torque_nm: float,
    pmax: float,
    vmax: float,
    tmax: float,
) -> bytes:
    pos_rad = np.radians(position_deg)
    vel_rad = np.radians(velocity_deg_s)

    kp_uint  = float_to_uint(kp, 0, 500, 12)
    kd_uint  = float_to_uint(kd, 0, 5,   12)
    q_uint   = float_to_uint(pos_rad, -pmax, pmax, 16)
    dq_uint  = float_to_uint(vel_rad, -vmax, vmax, 12)
    tau_uint = float_to_uint(torque_nm, -tmax, tmax, 12)

    data = [0] * 8
    data[0] = (q_uint >> 8) & 0xFF
    data[1] = q_uint & 0xFF
    data[2] = dq_uint >> 4
    data[3] = ((dq_uint & 0xF) << 4) | ((kp_uint >> 8) & 0xF)
    data[4] = kp_uint & 0xFF
    data[5] = kd_uint >> 4
    data[6] = ((kd_uint & 0xF) << 4) | ((tau_uint >> 8) & 0xF)
    data[7] = tau_uint & 0xFF
    return bytes(data)
