# Backward-compatibility shim — the canonical module is now constants.py
# bipedal_robot.py imports from this file and cannot be modified,
# so all names it uses must remain importable from here.
from constants import *  # noqa: F401, F403
from constants import (  # noqa: F401  (explicit for static analysers)
    REPO_ROOT,
    MODEL_URDF_DIR,
    MODEL_URDF_PATH,
    # CAN commands
    CAN_CMD_REQUEST_STATE,
    CAN_CMD_ENABLE,
    CAN_CMD_DISABLE,
    CAN_CMD_SET_ZERO,
    CAN_CMD_CLEAR_FAULT,   # alias
    CAN_CMD_ZERO,          # alias
    # Motor definitions
    MOTOR_TYPE,
    MotorConstants,
    MOTORS,
    MOTOR_IDS,
    CAN0_MOTOR_IDS,
    CAN1_MOTOR_IDS,
    LEFT_MOTOR_IDS,
    RIGHT_MOTOR_IDS,
    # Joint↔motor mapping
    MODEL_JOINT_TO_MOTOR_RIGHT,
    MODEL_JOINT_TO_MOTOR_LEFT,
    # Calibration
    DIRECT_JOINT_CALIBRATION_RIGHT,
    DIRECT_JOINT_CALIBRATION_LEFT,
    ANKLE_COUPLING_CALIBRATION_RIGHT,
    ANKLE_COUPLING_CALIBRATION_LEFT,
    MOTOR_SIGN,
    MOTOR_OFFSET_DEG,
    # Gains and limits
    DEFAULT_PD_GAINS,
    DEFAULT_GAINS,         # alias
    MOTOR_RAW_LIMITS_DEG,
    JOINT_LIMITS_DEG,      # alias
    COMMAND_MARGIN_DEG,
    STATE_MARGIN_DEG,
    NEAR_STOP_MARGIN_DEG,
    EMERGENCY_DAMPING_KD,
    DAMPING_KD,            # alias
)
