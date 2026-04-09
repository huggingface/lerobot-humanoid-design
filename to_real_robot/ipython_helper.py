"""
IPython copy-paste snippets for common robot workflows.
Run from to_real_robot/ directory.

Sections:
  1. Mock robot  — hardware-free testing (MockBus + MockIMU)
  2. Sim robot   — MuJoCo physics + RL agent
  3. Real robot  — live hardware
  4. OCP motion  — trajectory following
  5. Utilities   — gains, gamepad, IMU, delay measurement
"""

# =============================================================================
# 1. MOCK ROBOT  (no hardware, instant feedback, no physics)
# =============================================================================

from mock_robot import make_mock_robot

robot = make_mock_robot()
robot.start(mode="state_only")

snap = robot.get_combined_state_snapshot()
print(snap["joint_state_rad"])          # 12 joints, model space (rad)
print(snap["imu"]["quaternion_xyzw"])   # (0,0,0,1) = upright

# Tweak IMU at runtime
robot.mock_imu.set_state(gyro_rads=(0.1, 0.0, 0.0))

robot.stop()


# --- mock robot in control mode with RL agent ---

from mock_robot import make_mock_robot
from rl_agent import RLAgent

robot = make_mock_robot()
robot.start(mode="control", auto_enable=True)

agent = RLAgent.from_files(
    robot,
    config_path="policies/less_noise_high_gain_torque_obs/config.yaml",
    policy_path="policies/less_noise_high_gain_torque_obs/policy.onnx",
)
agent.spec.action_scale = 0.0   # start at zero — safe
agent.start()

# agent.spec.action_scale = 0.1   # ramp up gradually
# agent.stop()
# robot.stop()


# =============================================================================
# 2. SIM ROBOT  (MuJoCo physics — best for debugging RL policies)
# =============================================================================

import importlib
import sim_robot, rl_agent
importlib.reload(sim_robot)
importlib.reload(rl_agent)
from sim_robot import SimBipedalRobotController
from rl_agent import RLAgent

robot = SimBipedalRobotController(control_hz=200.0)
robot.start(mode="control", auto_enable=True)
robot.start_viewer()   # opens MuJoCo viewer window
# NOTE: sim gains are baked into the MJCF <actuator> section — set_joint_gains() is a no-op.
# Call robot.set_joint_gains(1) to inspect what kp/kv the loaded model actually uses.
agent = RLAgent.from_files(
    robot,
    config_path="policies/less_noise_high_gain/config.yaml",
    policy_path="policies/less_noise_high_gain/policy.onnx",
    log_path="policies/less_noise_high_gain/sim_debug.csv",
    log_observation=True,
    log_action=True,
    log_every_n=1,
)
agent.spec.action_scale = 0.1
agent.start()

# agent.stop(); robot.stop()


# --- sim: fixed-base (no fall, good for joint-level debugging) ---

from sim_robot import SimBipedalRobotController

robot = SimBipedalRobotController(control_hz=200.0, fixed_base=True)
robot.start(mode="control", auto_enable=True)
robot.start_viewer()

# Send the knees-bent reference pose (degrees, model space — same as reset pose).
# After reset, action is already latched to this pose → torques ≈ 0 at equilibrium.
import numpy as np
robot.set_action(
    left={
        "hipz": 0.0, "hipx": 0.0, "hipy": float(np.rad2deg(-20.0535)),
        "knee": float(np.rad2deg(40.1070)), "ankle_pitch": float(np.rad2deg(-20.0535)), "ankle_roll": 0.0,
    },
    right={
        "hipz": 0.0, "hipx": 0.0, "hipy": float(np.rad2deg(20.0535)),
        "knee": float(np.rad2deg(40.1070)), "ankle_pitch": float(np.rad2deg(20.0535)), "ankle_roll": 0.0,
    },
)


# --- sim + gamepad ---

import importlib, sim_robot, rl_agent
importlib.reload(sim_robot); importlib.reload(rl_agent)
from sim_robot import SimBipedalRobotController
from rl_agent import RLAgent
from gamepad_controller import GamepadController

robot = SimBipedalRobotController(control_hz=200.0)
robot.start(mode="control", auto_enable=True)
robot.start_viewer()

pad = GamepadController(name_substring="8bitdo", deadzone=0.12,
                        max_lin_x=0.75, max_lin_y=0.5, max_yaw_rate=0.8)
pad.connect()
pad.start()

agent = RLAgent.from_files(
    robot,
    config_path="policies/less_noise_high_gain_torque_obs/config.yaml",
    policy_path="policies/less_noise_high_gain_torque_obs/policy.onnx",
    log_path="policies/less_noise_high_gain_torque_obs/sim_gamepad_debug.csv",
    log_observation=True, log_action=True, log_every_n=1,
)
agent.spec.action_scale = 1.0
agent.set_command_source(pad)
agent.start()


# =============================================================================
# 3. REAL ROBOT  (live hardware — start conservative, scale up)
# =============================================================================

from bipedal_robot import BipedalRobotController
from rl_agent import RLAgent
from imu import IMU

# With IMU (BNO055 on Raspberry Pi)
imu = IMU(sensor="bno055", i2c_bus=1, address=0x28, rate_hz=100.0, frame_yaw_deg=180.0)
# imu = IMU(sensor="jy901", port="/dev/ttyAMA0", baudrate=9600)
# imu = IMU(mock=True)   # no IMU hardware

robot = BipedalRobotController(control_hz=100.0, imu=imu,
                                log_path="bipedal_state_log.csv")
robot.set_max_command_delta(60.0)
robot.start(mode="state_only")   # read-only first — verify state
robot.request_state_once()       # ensure valid stamps


# --- real robot: switch to control mode ---

robot.set_mode("control")
robot.enable_all()
robot.set_action(
    left= {"hipz": 0.0, "hipx": 0.0, "hipy": 0.0, "knee": 0.0, "ankle_pitch": 0.0, "ankle_roll": 0.0},
    right={"hipz": 0.0, "hipx": 0.0, "hipy": 0.0, "knee": 0.0, "ankle_pitch": 0.0, "ankle_roll": 0.0},
)


# --- real robot + RL agent ---

from rl_agent import RLAgent
from gamepad_controller import GamepadController

pad = GamepadController(name_substring="8bitdo", deadzone=0.12,
                        max_lin_x=0.75, max_lin_y=0.5, max_yaw_rate=0.8)
pad.connect()
pad.start()

agent = RLAgent.from_files(
    robot,
    config_path="policies/less_noise_high_gain_torque_obs/config.yaml",
    policy_path="policies/less_noise_high_gain_torque_obs/policy.onnx",
    log_path="policies/less_noise_high_gain_torque_obs/real_debug.csv",
    log_observation=True, log_action=True, log_every_n=1,
    clamp_ankle_to_true_limits=False,
)
agent.spec.joint_vel_source = "auto"
agent.spec.action_scale = 0.0       # start at ZERO — ramp manually
agent.set_command_source(pad)
agent.start()

# agent.spec.action_scale = 0.1   # ramp up


# =============================================================================
# 4. UTILITIES
# =============================================================================

# --- PD gains ---
# Real robot: set_joint_gains() works (sent over CAN).
# Sim: gains are baked into the MJCF <actuator> section — set_joint_gains() prints the
#      actual value and warns; to change them, edit the scene XML and reload.

for mid in [1, 7]:    robot.set_joint_gains(mid, kp=30,  kd=3.0)   # hipz
for mid in [2, 8]:    robot.set_joint_gains(mid, kp=40, kd=3.0)   # hipx
for mid in [3, 9]:    robot.set_joint_gains(mid, kp=60, kd=4.0)   # hipy
for mid in [4, 10]:   robot.set_joint_gains(mid, kp=60, kd=4.0)   # knee
for mid in [5, 6, 11, 12]: robot.set_joint_gains(mid, kp=20, kd=1.5)  # ankles


# --- inspect current state ---

import time
while True:
    snap = robot.get_combined_state_snapshot()
    print(f"estop={snap['estop']}  q={[round(v,3) for v in snap['joint_state_rad']]}")
    time.sleep(0.2)


# --- IMU live read ---

from imu import IMU
imu = IMU(sensor="bno055", i2c_bus=1, address=0x28, rate_hz=100.0, frame_yaw_deg=180.0)
while True:
    print(imu.read_dict()["gyro_rads"])
    time.sleep(0.1)


# --- actuator delay measurement ---

from tools.measure_actuator_delay import measure_actuator_delays

results = measure_actuator_delays(
    robot,
    joint="left_hipz",
    all_joints=False,
    fps=100, duration_s=2.0, freq_hz=1.0, amp_deg=10.0,
    pre_roll_s=1.0, between_s=0.5,
    save_csv="delay_logs/delay_trace.csv",
    save_png="delay_logs/delay_trace.png",
)
print(results["left_hipz"]["metrics"])


# --- motor scan (find which motors respond on each CAN bus) ---

from tools.motor_utils import scan_motors
print(scan_motors())   # {'can0': [1..6], 'can1': [7..12]}


# --- MeshCat visualization (separate terminal) ---
# python tools/launch_meshcat_browser.py

robot.attach_default_meshcat()
robot._viz_hz = 20.0
