# to_real_robot

Runtime stack for deploying trained RL policies on the bipedal humanoid robot (no arms). Covers hardware control, MuJoCo simulation, IMU, and gamepad teleoperation.

For a getting-started walkthrough, see the top-level `README.md`. This file is a per-module reference.

---

## Directory layout

```
to_real_robot/
├── bipedal_robot.py          # Real hardware controller (CAN)
├── sim_robot.py              # MuJoCo sim — same API as bipedal_robot
├── RL_agent_isolated.py      # RL policy inference loop
├── IMU_JY901.py              # JY901 UART IMU driver
├── gamepad_controller.py     # 8BitDo gamepad → (lin_x, lin_y, yaw_rate) twist
├── root_constant.py          # Motor IDs, calibration, gains, safety limits
├── bipdeal_config.py         # Motor calibration script (run-as-script only)
├── robstride_toolkit.py      # Low-level Robstride motor CLI
├── leg_test/mit.py           # MIT-mode CAN motor encoding (shared dep)
├── RL_policy/                # Trained policies (ONNX + config.yaml + gain.md)
└── bipedal_plateform_no_arms/# MuJoCo MJCF scene + STL assets
```

---

## Hardware overview

12 MIT-CAN servo motors, split across two CAN buses:

| Bus  | Motor IDs | Side  | Joints (per leg)                        |
|------|-----------|-------|-----------------------------------------|
| can0 | 1 – 6    | Left  | hipz, hipx, hipy, knee, ankle_a, ankle_b |
| can1 | 7 – 12   | Right | hipz, hipx, hipy, knee, ankle_a, ankle_b |

Ankles are mechanically coupled: `ankle_pitch` and `ankle_roll` are computed from the differential/sum of the two ankle motors.

An optional JY901 IMU provides orientation and angular velocity feedback for the RL policy.

---

## Core modules

### `bipedal_robot.py` — `BipedalRobotController`

Real-hardware controller.

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
robot.start_viewer()   # opens MuJoCo viewer window (blocks)
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
)
agent.spec.action_scale = 0.0    # start at 0 — ramp up manually once stable
agent.spec.joint_vel_source = "auto"
agent.set_command_source(pad)    # gamepad or manual set_command_twist()
agent.start()
```

`action_scale` is a global multiplier on all policy outputs. Starting at `0.0` is the safe default — the robot holds its current pose while the policy runs in the background. Ramp it toward `1.0` once the pose looks stable.

**Policy action order:** `[right(6), left(6)]` — hipz, hipx, hipy, knee, ankle_pitch, ankle_roll per side.

### `IMU_JY901.py` — `JY901IMU`

UART IMU driver (tested on `/dev/ttyAMA0` at 9600 baud).

```python
imu = JY901IMU(port="/dev/ttyAMA0", baudrate=9600)
imu.start()
# pass to BipedalRobotController(imu=imu)
```

Exposes orientation (quaternion), angular velocity, and linear acceleration. BNO055/BNO085 drivers lived in the removed `IMU_integration.py` — pull from `main` if needed.

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

Axes: left stick → lin_x / lin_y, right stick → yaw_rate. Values clamp to the training maximum yaw rate (`TRAINING_MAX_YAW_RATE = 0.5 rad/s`).

### `bipdeal_config.py`

Motor calibration script — run directly, do not import.

> **Known issue:** log output was pasted into the source around line 406, which makes the file unimportable. The file is only invoked as a standalone script, so this doesn't break anything, but it should be cleaned up. Pre-existing on `main`.

### `robstride_toolkit.py`

Low-level Robstride motor CLI — `python robstride_toolkit.py --help`. Use for one-off motor queries / flashing IDs outside the main runtime.

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

## MuJoCo simulation scene

`bipedal_plateform_no_arms/mjcf/sim_scene_safe.xml` — the simulation scene used by `SimBipedalRobotController`. STL assets are in `bipedal_plateform_no_arms/mjcf/assets/`.

Default spawn: hardcoded stable qpos from MJLab (`MJLAB_HARDCODED_SPAWN_QPOS_FREE` in `sim_robot.py`), height ≈ 0.77 m.

---

## Python environment

Key dependencies: `python-can`, `mujoco`, `onnxruntime`, `pinocchio`, `meshcat`, `evdev`, `pyserial`, `PyYAML`, `numpy`. See top-level `environement.lock`.

---

## Notes

- **`RL_agent_isolated.py` is the canonical agent file.** Ignore any references to `RL_agent.py` in older snippets — it is archived.
- **`action_scale` ramp** — done manually in IPython while observing the robot. Start at `0.0`, increase incrementally (e.g. `agent.spec.action_scale = 0.1`) until the policy takes full authority.
- **Torque observation** — `less_noice_high_gain_torque_obs` reads torque directly from the MIT motor state frames inside `bipedal_robot.py`. No estimator involved.
- **Motor recalibration** — the `MOTOR_OFFSET_DEG` values in `root_constant.py` encode the zero-pose calibration. Recalibration after reassembly is a manual procedure; not yet documented.
