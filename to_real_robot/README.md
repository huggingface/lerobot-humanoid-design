# to_real_robot

Runtime stack for deploying trained RL policies on the bipedal humanoid robot (no arms). Covers hardware control, MuJoCo simulation, IMU integration, gamepad teleoperation, and OCP trajectory tracking.

---

## Directory layout

```
to_real_robot/
├── bipedal_robot.py          # Real hardware controller (CAN)
├── sim_robot.py              # MuJoCo sim — same API as bipedal_robot
├── RL_agent_isolated.py      # RL policy inference loop
├── IMU_integration.py        # IMU selector (BNO085 / BNO055 / JY901)
├── IMU_JY901.py              # JY901 UART driver
├── IMU_BNO055                # BNO055 I2C driver (extensionless Python source)
├── mock_bus.py               # CAN bus mock for hardware-free testing
├── gamepad_controller.py     # 8BitDo gamepad → (lin_x, lin_y, yaw_rate) twist
├── root_constant.py          # Motor IDs, calibration, gains, safety limits
├── measure_actuator_delay.py # Sinusoidal step-response delay measurement
├── launch_meshcat_browser.py # Helper to open Meshcat in browser
├── hip_imu_debug.py          # Hip/IMU debug script
├── bipdeal_config.py         # Legacy config (mostly superseded)
├── robstride_toolkit.py      # Low-level Robstride motor utilities
├── ipython_helper.py         # IPython snippets for interactive sessions
├── RL_policy/                # Trained policies (ONNX + config.yaml + gain.md)
├── bipedal_plateform_no_arms/# MuJoCo MJCF scene + STL assets
└── debug_logs/               # Delay trace outputs
```

---

## Hardware overview

12 MIT-CAN servo motors, split across two CAN buses:

| Bus  | Motor IDs | Side  | Joints (per leg)                        |
|------|-----------|-------|-----------------------------------------|
| can0 | 1 – 6    | Left  | hipz, hipx, hipy, knee, ankle_a, ankle_b |
| can1 | 7 – 12   | Right | hipz, hipx, hipy, knee, ankle_a, ankle_b |

Ankles are mechanically coupled: `ankle_pitch` and `ankle_roll` are computed from the differential/sum of the two ankle motors.

The robot optionally carries an IMU (BNO055 or JY901) for orientation and angular velocity feedback fed to the RL policy.

---

## Core modules

### `bipedal_robot.py` — `BipedalRobotController`

Real-hardware controller. Do not modify — it is the production binary.

```python
robot = BipedalRobotController(
    control_hz=200.0,
    log_path="bipedal_state_log.csv",
    imu=imu,          # optional
)
```

Key methods:

| Method | Description |
|--------|-------------|
| `robot.start(mode, auto_enable)` | `mode="state_only"` reads sensors only; `mode="control"` enables actuation |
| `robot.set_mode("control")` | Switch from state_only to control after startup |
| `robot.enable_all()` | Send enable command to all 12 motors |
| `robot.set_action(left={...}, right={...})` | Set joint position targets in degrees |
| `robot.set_joint_gains(motor_id, kp, kd)` | Override PD gains for one motor |
| `robot.set_joint_limit(motor_id, lo, hi)` | Override software safety limits |
| `robot.set_max_command_delta(deg)` | Max position step per control cycle |
| `robot.request_state_once()` | Force one state read (needed before first action) |
| `robot.attach_default_meshcat()` | Attach Meshcat visualizer |
| `robot.stop()` | Graceful shutdown |

**Startup sequence on real hardware:**

```python
robot.start(mode="state_only", auto_enable=False)   # read-only first
robot.set_action(left={all zeros}, right={all zeros})
robot.set_mode("control")
robot.enable_all()
# now safe to set gains and run RL agent
```

### `sim_robot.py` — `SimBipedalRobotController`

MuJoCo simulation with the same API as `BipedalRobotController`. Uses `bipedal_plateform_no_arms/mjcf/sim_scene_safe.xml` by default.

```python
robot = SimBipedalRobotController(control_hz=200.0, fixed_base=False)
robot.start(mode="control", auto_enable=True)
robot.start_viewer()   # opens MuJoCo viewer window
```

Extra method: `robot.enable_debug_trace(path, every_n=1)` — logs sim state to CSV.

### `RL_agent_isolated.py` — `RLAgent`

Runs a trained ONNX policy in a background thread at `inference_hz` (default 50 Hz). Reads robot state, assembles the observation vector, runs the ONNX model, and calls `robot.set_action()`.

```python
agent = RLAgent.from_files(
    robot,
    config_path="RL_policy/<name>/config.yaml",
    policy_path="RL_policy/<name>/policy.onnx",
    log_path="RL_policy/<name>/debug_ctrl.csv",   # optional
    log_observation=True,
    log_action=True,
    log_every_n=1,
    clamp_ankle_to_true_limits=False,
)
agent.spec.action_scale = 0.0    # start at 0 — ramp up manually once stable
agent.spec.joint_vel_source = "auto"   # or "finite_difference"
agent.set_command_source(pad)    # gamepad or manual set_command_twist()
agent.start()
```

`action_scale` is a global multiplier on all policy outputs. Starting at `0.0` is the safe default — the robot holds its current pose while the policy runs in the background. Ramp it toward `1.0` once the pose looks stable.

**Policy action order:** `[right(6), left(6)]` — hipz, hipx, hipy, knee, ankle_pitch, ankle_roll per side.

### `IMU_integration.py` — `IMU`

Unified IMU selector. Three supported sensors:

| Sensor | Interface | Class |
|--------|-----------|-------|
| `"bno085"` | I2C (Adafruit CircuitPython) | `BNO085IMU` |
| `"bno055"` | I2C, local driver | `BNO055I2CIMU` |
| `"jy901"` | UART (`/dev/ttyAMA0`) | `JY901UARTIMU` |

```python
# BNO055 on Raspberry Pi (current setup on real robot)
imu = IMU(sensor="bno055", i2c_bus=1, address=0x28, rate_hz=100.0, frame_yaw_deg=-180.0)

# JY901 over UART
imu = IMU(sensor="jy901", port="/dev/ttyAMA0", baudrate=9600)
```

`frame_yaw_deg` rotates the sensor frame to match the robot body frame. `-180.0` means the IMU is mounted backwards.

### `mock_bus.py` — `MockBus`

CAN bus mock for hardware-free development. Motors instantly jump to commanded positions (zero inertia). Useful for testing the control loop, observation pipeline, and visualizer without hardware.

```python
robot = BipedalRobotController(
    bus_can0=MockBus(), bus_can1=MockBus(),
    control_hz=100.0,
)
# Required with mock:
for mid in range(1, 13):
    robot.set_joint_limit(mid, -720.0, 720.0)
robot.set_max_command_delta(1000.0)
```

> **Known issue:** `disable_safety_limits=True` (or wide limits as above) is required. Motor 1 raw position 0 falls outside its calibrated range and triggers an E-STOP deadlock inside the RX callback. This is a bug in `bipedal_robot.py` that cannot be patched externally.

### `gamepad_controller.py` — `GamepadController`

Reads a Linux `evdev` gamepad (tested with 8BitDo) and outputs a `(lin_x, lin_y, yaw_rate)` command twist consumed by `RLAgent`.

```python
pad = GamepadController(
    name_substring="8bitdo",
    deadzone=0.12,
    max_lin_x=0.75,
    max_lin_y=0.5,
    max_yaw_rate=0.8,
)
pad.connect()
pad.start()
agent.set_command_source(pad)
```

Axes: left stick → lin_x / lin_y, right stick → yaw_rate. All values are clamped to the training maximum yaw rate (`TRAINING_MAX_YAW_RATE = 0.5 rad/s`).

---

## RL policies (`RL_policy/`)

Each subdirectory contains:
- `policy.onnx` — exported policy network
- `config.yaml` — observation/action spec, inference Hz, scales
- `gain.md` — PD gains to use with this policy on the real robot

**Current best policies** (as of 2026-04):

| Directory | Notes |
|-----------|-------|
| `less_noice_high_gain_torque_obs` | Best overall; includes torque in observation |
| `less_noice_high_gain` | Strong baseline without torque obs |
| `less_noice_high_gain_2` | Variant of above |

Also included: `RL_policy/robot.xml` — MuJoCo training scene.

---

## Real-robot RL deployment (full sequence)

This is the workflow currently used on the real robot. Copy-paste from `ipython_helper.py` (last section):

```python
from bipedal_robot import BipedalRobotController
from IMU_integration import IMU
from RL_agent_isolated import RLAgent
from gamepad_controller import GamepadController

# 1. IMU — BNO055 mounted backwards (-180° yaw correction)
imu = IMU(sensor="bno055", i2c_bus=1, address=0x28, rate_hz=100.0, frame_yaw_deg=-180.0)

# 2. Robot — start read-only, no auto-enable
robot = BipedalRobotController(control_hz=100.0, imu=imu)
robot.attach_default_meshcat()        # optional Meshcat visualizer
robot.set_max_command_delta(60.0)
robot.start(mode="state_only", auto_enable=False)
robot._viz_hz = 20.0

# 3. Go to zero pose, then enable
robot.set_action(left={all zeros}, right={all zeros})
robot.set_mode("control")
robot.enable_all()

# 4. Set per-joint PD gains (from gain.md for this policy)
for mid in [1, 7]:     robot.set_joint_gains(mid, kp=30, kd=3.0)   # hipz
for mid in [5,6,11,12]: robot.set_joint_gains(mid, kp=10, kd=0.75) # ankles
robot.set_joint_gains(2, kp=40, kd=3.0)   # left  hipx
robot.set_joint_gains(3, kp=6,  kd=0.2)   # left  hipy
robot.set_joint_gains(4, kp=6,  kd=0.2)   # left  knee
robot.set_joint_gains(8, kp=40, kd=3.0)   # right hipx
robot.set_joint_gains(9, kp=6,  kd=0.2)   # right hipy
robot.set_joint_gains(10, kp=6, kd=0.2)   # right knee

# 5. Gamepad
pad = GamepadController(name_substring="8bitdo", deadzone=0.12,
                        max_lin_x=0.75, max_lin_y=0.5, max_yaw_rate=0.8)
pad.connect()
pad.start()

# 6. RL agent
agent = RLAgent.from_files(
    robot,
    config_path="RL_policy/less_noice_high_gain_torque_obs/config.yaml",
    policy_path="RL_policy/less_noice_high_gain_torque_obs/policy.onnx",
    log_path="RL_policy/less_noice_high_gain_torque_obs/debug_ctrl5.csv",
    log_observation=True,
    log_action=True,
    log_every_n=1,
)
agent.spec.joint_vel_source = "auto"
agent.spec.action_scale = 0.0          # safe default — ramp up manually in IPython once robot looks stable
agent.set_command_source(pad)
agent.start()
```

---

## Actuator delay measurement

Injects a sinusoidal command on one joint and cross-correlates the response to estimate round-trip delay.

```python
from measure_actuator_delay import measure_actuator_delays

results = measure_actuator_delays(
    robot,
    joint="left_hipz",
    fps=100, duration_s=2.0, freq_hz=1.0, amp_deg=10.0,
    save_csv="debug_logs/delay_trace.csv",
    save_png="debug_logs/delay_trace.png",
)
print(results["left_hipz"]["metrics"])
```

---

## MuJoCo simulation scene

`bipedal_plateform_no_arms/mjcf/sim_scene_safe.xml` — the simulation scene used by `SimBipedalRobotController`. STL assets are in `bipedal_plateform_no_arms/mjcf/assets/`.

Default spawn: hardcoded stable qpos from MJLab (`MJLAB_HARDCODED_SPAWN_QPOS_FREE` in `sim_robot.py`), height ≈ 0.77 m.

---

## Python environment

```
/home/virgile/micromamba/envs/lerobot/
```

Key dependencies: `python-can`, `mujoco`, `onnxruntime`, `pinocchio`, `meshcat`, `evdev`, `pyserial`, `PyYAML`, `numpy`.

---

## Notes

- **`RL_agent_isolated.py` is the canonical agent file.** Ignore any references to `RL_agent.py` in older snippets — it is archived.
- **`action_scale` ramp** — done manually in IPython while observing the robot. Start at `0.0`, increase incrementally (e.g. `agent.spec.action_scale = 0.1`) until the policy takes full authority.
- **IMU mounting** — the BNO055 `frame_yaw_deg=-180.0` corrects for the sensor being mounted backwards. This is fixed for the current robot.
- **Torque observation** — `less_noice_high_gain_torque_obs` reads torque directly from the MIT motor state frames inside `bipedal_robot.py`. No estimator involved.
- **Motor recalibration** — the `MOTOR_OFFSET_DEG` values in `root_constant.py` encode the zero-pose calibration. Recalibration after reassembly is a manual procedure; not yet documented.
