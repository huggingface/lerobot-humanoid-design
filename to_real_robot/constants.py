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
# 0xFB: request current motor state (officially "clear fault")
# 0xFC: enable motor torque
# 0xFD: disable motor torque
# 0xFE: set current position as encoder zero
# -------------------------
CAN_CMD_REQUEST_STATE = 0xFB  # also known as CLEAR_FAULT in firmware docs
CAN_CMD_ENABLE = 0xFC
CAN_CMD_DISABLE = 0xFD
CAN_CMD_SET_ZERO = 0xFE

# Backward-compat aliases (used by bipedal_robot.py which must not be modified)
CAN_CMD_CLEAR_FAULT = CAN_CMD_REQUEST_STATE
CAN_CMD_ZERO = CAN_CMD_SET_ZERO


# -------------------------
# Motor definitions
# ankle_a / ankle_b are the two parallel motors that drive the ankle
# differential: ankle_pitch = -(a - b)/2, ankle_roll = (a + b)/2
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
    1: MotorConstants(1, "left_hipz",    12.57, 33.0, 14.0),
    2: MotorConstants(2, "left_hipx",    12.57, 33.0, 20.0),
    3: MotorConstants(3, "left_hipy",    12.57, 33.0, 60.0),
    4: MotorConstants(4, "left_knee",    12.57, 33.0, 60.0),
    5: MotorConstants(5, "left_ankle_a", 12.57, 50.0,  5.5),  # ankle differential motor A
    6: MotorConstants(6, "left_ankle_b", 12.57, 50.0,  5.5),  # ankle differential motor B
    # Right leg on can1: IDs 7..12
    7:  MotorConstants(7,  "right_hipz",    12.57, 33.0, 14.0),
    8:  MotorConstants(8,  "right_hipx",    12.57, 33.0, 20.0),
    9:  MotorConstants(9,  "right_hipy",    12.57, 33.0, 60.0),
    10: MotorConstants(10, "right_knee",    12.57, 33.0, 60.0),
    11: MotorConstants(11, "right_ankle_a", 12.57, 50.0,  5.5),
    12: MotorConstants(12, "right_ankle_b", 12.57, 50.0,  5.5),
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
# ankle_pitch and ankle_roll are both driven by the (a, b) motor pair
# -------------------------
MODEL_JOINT_TO_MOTOR_RIGHT = {
    "hipz":         7,
    "hipx":         8,
    "hipy":         9,
    "knee":         10,
    "ankle_pitch":  (11, 12),  # coupled differential
    "ankle_roll":   (11, 12),  # coupled differential
}

MODEL_JOINT_TO_MOTOR_LEFT = {
    "hipz":         1,
    "hipx":         2,
    "hipy":         3,
    "knee":         4,
    "ankle_pitch":  (5, 6),  # coupled differential
    "ankle_roll":   (5, 6),  # coupled differential
}


# -------------------------
# Motor calibration: raw encoder -> model joint space
# calibrated_deg = sign * raw_motor_deg + offset_deg
#
# Direct joints (hipz, hipx, hipy, knee):
# -------------------------
DIRECT_JOINT_CALIBRATION_RIGHT = {
    "hipz": {"motor_id": 7,  "sign": +1.0, "offset_deg": -132.68},
    "hipx": {"motor_id": 8,  "sign": +1.0, "offset_deg": -19.394},
    "hipy": {"motor_id": 9,  "sign": +1.0, "offset_deg": -88.096},
    "knee": {"motor_id": 10, "sign": +1.0, "offset_deg":  57.352},
}

DIRECT_JOINT_CALIBRATION_LEFT = {
    "hipz": {"motor_id": 1, "sign": +1.0, "offset_deg":  132.68},
    "hipx": {"motor_id": 2, "sign": +1.0, "offset_deg":   19.394},
    "hipy": {"motor_id": 3, "sign": +1.0, "offset_deg":   88.096},
    "knee": {"motor_id": 4, "sign": +1.0, "offset_deg":   57.352},
}


# Ankle differential (motors a, b -> pitch, roll):
# ankle_pitch = sign_pitch * ((a - b) / 2) + offset_pitch
# ankle_roll  = sign_roll  * ((a + b) / 2) + offset_roll
ANKLE_COUPLING_CALIBRATION_RIGHT = {
    "motors": (11, 12),
    "pitch": {"sign": -1.0, "offset_deg": 0.0},  # right mirrored vs left
    "roll":  {"sign": +1.0, "offset_deg": 0.0},
}

ANKLE_COUPLING_CALIBRATION_LEFT = {
    "motors": (5, 6),
    "pitch": {"sign": -1.0, "offset_deg": 0.0},
    "roll":  {"sign": +1.0, "offset_deg": 0.0},
}

# Flat per-motor calibration tables used by the controller.
# calibrated_deg = MOTOR_SIGN[mid] * raw_motor_deg + MOTOR_OFFSET_DEG[mid]
MOTOR_SIGN: Dict[int, float] = {mid: 1.0 for mid in MOTOR_IDS}
MOTOR_OFFSET_DEG: Dict[int, float] = {mid: 0.0 for mid in MOTOR_IDS}
for _cfg in DIRECT_JOINT_CALIBRATION_LEFT.values():
    MOTOR_SIGN[_cfg["motor_id"]] = float(_cfg["sign"])
    MOTOR_OFFSET_DEG[_cfg["motor_id"]] = float(_cfg["offset_deg"])
for _cfg in DIRECT_JOINT_CALIBRATION_RIGHT.values():
    MOTOR_SIGN[_cfg["motor_id"]] = float(_cfg["sign"])
    MOTOR_OFFSET_DEG[_cfg["motor_id"]] = float(_cfg["offset_deg"])

# Sign overrides established during calibration session
MOTOR_SIGN[4] = -1.0
MOTOR_SIGN[5] = -1.0
MOTOR_SIGN[6] = -1.0
MOTOR_SIGN[11] = -1.0
MOTOR_SIGN[12] = -1.0


# -------------------------
# Default PD gains  {motor_id: (kp_Nm_per_rad, kd_Nm_s_per_rad)}
# -------------------------
DEFAULT_PD_GAINS: Dict[int, Tuple[float, float]] = {
    1:  (5.0,  0.5),  2:  (5.0,  0.5),   # hipz  left/right  (low stiffness, yaw)
    3:  (15.0, 1.5),  4:  (15.0, 1.5),   # hipy  left / knee left
    5:  (8.0,  0.5),  6:  (8.0,  0.5),   # ankle_a/b left
    7:  (5.0,  0.5),  8:  (5.0,  0.5),   # hipz  right / hipx right
    9:  (15.0, 1.5),  10: (15.0, 1.5),   # hipy  right / knee right
    11: (8.0,  0.5),  12: (8.0,  0.5),   # ankle_a/b right
}
DEFAULT_GAINS = DEFAULT_PD_GAINS  # backward-compat alias (used by bipedal_robot.py)


# -------------------------
# Safety limits in RAW MOTOR degrees (from encoder min/max scan).
# These are raw encoder values, NOT model joint space.
# Values ending in "minus 360" were wrap-corrected from the scan log.
# -------------------------
MOTOR_RAW_LIMITS_DEG: Dict[int, Tuple[float, float]] = {
    1:  (-209.123,  -47.685),  # raw m1 − 360
    2:  ( -50.887,   50.934),  # raw m2 − 360
    3:  (-168.065,    0.0  ),  # raw m3 − 360
    4:  (   0.978,  112.172),  # raw m4
    5:  ( -89.753,   20.042),  # raw m5
    6:  ( -21.14,   285.830),  # raw m6
    7:  (  64.894,  215.255),  # raw m7
    8:  ( -55.311,   40.409),  # raw m8 − 360
    9:  (  -0.0,    162.438),  # raw m9
    10: ( -98.105,   -0.693),  # raw m10 − 360
    11: ( -21.50,    87.972),  # raw m11
    12: ( -78.191,   16.9  ),  # raw m12
}
JOINT_LIMITS_DEG = MOTOR_RAW_LIMITS_DEG  # backward-compat alias (used by bipedal_robot.py)

COMMAND_MARGIN_DEG = 1.0    # keep command this many deg away from limit
STATE_MARGIN_DEG = 0.5      # warn/stop this many deg before limit
NEAR_STOP_MARGIN_DEG = 0.5  # hysteresis for near-stop detection

EMERGENCY_DAMPING_KD = 1.5  # kd gain applied when near a joint limit (pure damping)
DAMPING_KD = EMERGENCY_DAMPING_KD  # backward-compat alias (used by bipedal_robot.py)
