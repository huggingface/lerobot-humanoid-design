from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


# -------------------------
# Robot model path
# -------------------------
REPO_ROOT = Path(__file__).resolve().parent
MODEL_URDF_DIR = REPO_ROOT / "model" / "urdf"
MODEL_URDF_PATH = MODEL_URDF_DIR / "robot.urdf"


# -------------------------
# CAN protocol commands
# -------------------------
CAN_CMD_CLEAR_FAULT = 0xFB
CAN_CMD_ZERO = 0xFE
CAN_CMD_ENABLE = 0xFC
CAN_CMD_DISABLE = 0xFD


# -------------------------
# Motor definitions
# -------------------------
MOTOR_TYPE = "MIT-CAN servo"


@dataclass(frozen=True)
class MotorConstants:
    motor_id: int
    name: str
    pmax_rad: float
    vmax_rad_s: float
    tmax_nm: float
    motor_type: str = MOTOR_TYPE


MOTORS: Dict[int, MotorConstants] = {
    # Left leg on can0: IDs 1..6
    1: MotorConstants(1, "left_hipz", 12.57, 33.0, 14.0),
    2: MotorConstants(2, "left_hipx", 12.57, 33.0, 20.0),
    3: MotorConstants(3, "left_hipy", 12.57, 33.0, 60.0),
    4: MotorConstants(4, "left_knee", 12.57, 33.0, 60.0),
    5: MotorConstants(5, "left_ankle1", 12.57, 50.0, 5.5),
    6: MotorConstants(6, "left_ankle2", 12.57, 50.0, 5.5),
    # Right leg on can1: IDs 7..12
    7: MotorConstants(7, "right_hipz", 12.57, 33.0, 14.0),
    8: MotorConstants(8, "right_hipx", 12.57, 33.0, 20.0),
    9: MotorConstants(9, "right_hipy", 12.57, 33.0, 60.0),
    10: MotorConstants(10, "right_knee", 12.57, 33.0, 60.0),
    11: MotorConstants(11, "right_ankle1", 12.57, 50.0, 5.5),
    12: MotorConstants(12, "right_ankle2", 12.57, 50.0, 5.5),
}
MOTOR_IDS: Tuple[int, ...] = tuple(sorted(MOTORS.keys()))
CAN0_MOTOR_IDS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
CAN1_MOTOR_IDS: Tuple[int, ...] = (7, 8, 9, 10, 11, 12)
LEFT_MOTOR_IDS: Tuple[int, ...] = CAN0_MOTOR_IDS
RIGHT_MOTOR_IDS: Tuple[int, ...] = CAN1_MOTOR_IDS


# -------------------------
# Joint <-> motor association
# Per-leg model joint order:
# [hipz, hipx, hipy, knee, ankle_pitch, ankle_roll]
# -------------------------
MODEL_JOINT_TO_MOTOR_RIGHT = {
    "hipz": 7,
    "hipx": 8,
    "hipy": 9,
    "knee": 10,
    "ankle_pitch": (11, 12),  # coupled
    "ankle_roll": (11, 12),   # coupled
}

MODEL_JOINT_TO_MOTOR_LEFT = {
    "hipz": 1,
    "hipx": 2,
    "hipy": 3,
    "knee": 4,
    "ankle_pitch": (5, 6),  # coupled
    "ankle_roll": (5, 6),   # coupled
}


# -------------------------
# Real <-> model sign/offset calibration
# q_model_deg = sign * q_real_deg + offset_deg
# (for direct joints only)
# -------------------------
DIRECT_JOINT_CALIBRATION_RIGHT = {
    "hipz": {"motor_id": 7, "sign": +1.0, "offset_deg": -132.68},
    "hipx": {"motor_id": 8, "sign": +1.0, "offset_deg": -19.394},
    "hipy": {"motor_id": 9, "sign": +1.0, "offset_deg": -88.096},
    "knee": {"motor_id": 10, "sign": +1.0, "offset_deg": 57.352},
}

DIRECT_JOINT_CALIBRATION_LEFT = {
    "hipz": {"motor_id": 1, "sign": +1.0, "offset_deg": 132.68},
    "hipx": {"motor_id": 2, "sign": +1.0, "offset_deg": 19.394},
    "hipy": {"motor_id": 3, "sign": +1.0, "offset_deg": 88.096},
    "knee": {"motor_id": 4, "sign": +1.0, "offset_deg": 57.352},
}


# Coupled ankle mapping from real motor angles (deg) to model joint angles (deg):
# ankle_pitch = sign_pitch * ((m5 - m6) / 2) + offset_pitch
# ankle_roll  = sign_roll  * ((m5 + m6) / 2) + offset_roll
# This matches current code in robot.py:
# q[4] = -deg2rad((m5 - m6)/2), q[5] = +deg2rad((m5 + m6)/2)
ANKLE_COUPLING_CALIBRATION_RIGHT = {
    "motors": (11, 12),
    # Right ankle is mirrored: signs are reversed vs left side.
    "pitch": {"sign": -1.0, "offset_deg": 0.0},
    "roll": {"sign": +1.0, "offset_deg": 0.0},
}

ANKLE_COUPLING_CALIBRATION_LEFT = {
    "motors": (5, 6),
    "pitch": {"sign": -1.0, "offset_deg": 0.0},
    "roll": {"sign": +1.0, "offset_deg": 0.0},
}

# Flat per-motor calibration table used by the controller.
# calibrated_deg = sign * raw_motor_deg + offset_deg
MOTOR_SIGN = {mid: 1.0 for mid in MOTOR_IDS}
MOTOR_OFFSET_DEG = {mid: 0.0 for mid in MOTOR_IDS}
for cfg in DIRECT_JOINT_CALIBRATION_LEFT.values():
    MOTOR_SIGN[cfg["motor_id"]] = float(cfg["sign"])
    MOTOR_OFFSET_DEG[cfg["motor_id"]] = float(cfg["offset_deg"])
for cfg in DIRECT_JOINT_CALIBRATION_RIGHT.values():
    MOTOR_SIGN[cfg["motor_id"]] = float(cfg["sign"])
    MOTOR_OFFSET_DEG[cfg["motor_id"]] = float(cfg["offset_deg"])

# Explicit per-motor sign overrides from current calibration session.
MOTOR_SIGN[4] = -1.0
MOTOR_SIGN[5] = -1.0
MOTOR_SIGN[6] = -1.0
MOTOR_SIGN[11] = -1.0
MOTOR_SIGN[12] = -1.0


# -------------------------
# Control defaults
# -------------------------
DEFAULT_GAINS = {
    1: (5.0, 0.5), 2: (5.0, 0.5), 3: (15.0, 1.5), 4: (15.0, 1.5), 5: (8.0, 0.5), 6: (8.0, 0.5),
    7: (5.0, 0.5), 8: (5.0, 0.5), 9: (15.0, 1.5), 10: (15.0, 1.5), 11: (8.0, 0.5), 12: (8.0, 0.5),
}


# -------------------------
# Safety limits (deg)
# Updated from first raw limit scan in bipdeal_config.py comments.
# Values flagged as "-360" were wrap-corrected before being entered here.
# -------------------------
JOINT_LIMITS_DEG = {
    1: (-209.123, -47.685),   # raw m1 minus 360
    2: (-129.887, 12.934),    # raw m2 minus 360
    3: (-168.065, 1.967),     # raw m3 minus 360
    4: (0.978, 112.172),      # raw m4
    5: (-89.753, 11.542),     # raw m5
    6: (-18.14, 285.830),    # raw m6
    7: (64.894, 215.255),     # raw m7
    8: (-55.311, 40.409),     # raw m8 minus 360
    9: (-0.055, 162.438),     # raw m9
    10: (-98.105, -0.693),    # raw m10 minus 360
    11: (-12.50, 87.972),     # raw m11
    12: (-78.191, 13.9),   # raw m12
}

COMMAND_MARGIN_DEG = 1.0
STATE_MARGIN_DEG = 3.0
NEAR_STOP_MARGIN_DEG = 1.0
DAMPING_KD = 1.5
