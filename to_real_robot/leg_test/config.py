# leg_test/config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

CAN_CMD_CLEAR_FAULT = 0xFB
CAN_CMD_ZERO = 0xFE
CAN_CMD_ENABLE = 0xFC
CAN_CMD_DISABLE = 0xFD


@dataclass(frozen=True)
class MotorSpec:
    motor_id: int
    name: str
    pmax: float
    vmax: float
    tmax: float


LEG_MOTORS: Dict[int, MotorSpec] = {
    1: MotorSpec(1, "hipz",   12.57, 33, 14),
    2: MotorSpec(2, "hipx",   12.57, 33, 20),
    3: MotorSpec(3, "knee",   12.57, 33, 60),
    4: MotorSpec(4, "hipy",   12.57, 33, 60),
    5: MotorSpec(5, "ankle1", 12.57, 50, 5.5),
    6: MotorSpec(6, "ankle2", 12.57, 50, 5.5),
}
LEG_MOTOR_IDS = tuple(sorted(LEG_MOTORS.keys()))

DEFAULT_GAINS = {
    1: (5.0, 0.1),
    2: (5.0, 0.1),
    3: (5.0, 0.5),
    4: (5.0, 0.5),
    5: (5.0, 0.5),
    6: (5.0, 0.5),
}

# -------------------------
# SAFETY LIMITS (deg)
# -------------------------
# Put your current "true limits" here. Keep them conservative.
# (You can refine later once you trust calibration.)
JOINT_LIMITS_DEG: Dict[int, Tuple[float, float]] = {
    5: (-100.0, -0.0),   # ankle1
    6: (  0.0, 100.0),   # ankle2
    3: (-170.0, -0.0), # knee
    4: (-168.0, -0.0), # hipy
    2: (  0., 240.0), # hipx
    1: (  0., 260.0), # hipz (adjust if this is not correct!)
}

# Safety margins:
# - command_margin: shrink allowed command range a bit (avoids grazing hardstops)
# - state_margin: allow small measurement errors before e-stop
COMMAND_MARGIN_DEG = 1.0
STATE_MARGIN_DEG = 3.0
