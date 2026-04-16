# LeRobot Humanoid — Minimal

Minimal branch: sim-to-real RL policy deployment on the bipedal platform. Design studies, OCP experiments, retired URDFs and debug/calibration tooling have been removed here — see `main` for the full repo.

---

## What's in here

```
to_real_robot/
├── sim_robot.py              # MuJoCo sim (same API as bipedal_robot)
├── bipedal_robot.py          # Real hardware controller (CAN)
├── bipdeal_config.py         # Motor calibration script (run-as-script only)
├── robstride_toolkit.py      # Low-level Robstride motor CLI
├── RL_agent_isolated.py      # Policy runner (ONNX)
├── gamepad_controller.py     # 8BitDo gamepad → twist command
├── IMU_JY901.py              # JY901 UART IMU driver
├── root_constant.py          # Motor IDs, calibration, gains, limits
├── leg_test/mit.py           # MIT-CAN motor encoding (shared dep)
├── RL_policy/                # Trained ONNX policies + configs
└── bipedal_plateform_no_arms/# MuJoCo MJCF + STL assets
```

See `to_real_robot/README.md` for per-module reference.

---

## Quick tutorial

Everything runs from `to_real_robot/` as CWD. Start there:

```bash
cd to_real_robot
python
```

### 1. Sim only — poke the robot

Safest first step. Opens a MuJoCo viewer and holds the robot at the default pose.

```python
from sim_robot import SimBipedalRobotController

robot = SimBipedalRobotController(control_hz=200.0)
robot.start(mode="control", auto_enable=True)
robot.start_viewer()   # opens a MuJoCo window
```

The viewer blocks the REPL. Ctrl-C or close the window to exit.

You can pause the viewer with space and drag-apply forces to test stability.

### 2. Sim + trained policy

Load one of the pretrained policies and watch it balance / walk.

```python
from sim_robot import SimBipedalRobotController
from RL_agent_isolated import RLAgent

robot = SimBipedalRobotController(control_hz=200.0)
robot.start(mode="control", auto_enable=True)

agent = RLAgent.from_files(
    robot,
    config_path="RL_policy/less_noice_high_gain_torque_obs/config.yaml",
    policy_path="RL_policy/less_noice_high_gain_torque_obs/policy.onnx",
)
agent.spec.action_scale = 0.0   # start safe; ramp up once stable
agent.spec.joint_vel_source = "auto"
agent.start()

robot.start_viewer()            # blocks
```

In the viewer, you can ramp the policy by setting `agent.spec.action_scale = 0.5` then `1.0` from another Python thread (or just restart with `action_scale=1.0` if you trust the policy).

**To command velocities** without a gamepad:

```python
agent.set_command_twist(lin_x=0.3, lin_y=0.0, yaw_rate=0.0)  # walk forward
```

### 3. Sim + gamepad

8BitDo-style gamepad over evdev. Left stick = linear, right stick = yaw.

```python
from gamepad_controller import GamepadController

pad = GamepadController(
    name_substring="8bitdo", deadzone=0.12,
    max_lin_x=0.75, max_lin_y=0.5, max_yaw_rate=0.8,
)
pad.connect()
pad.start()
agent.set_command_source(pad)
```

### 4. Real robot

Hardware prerequisites: 12 Robstride MIT-CAN motors on `can0` + `can1`, optional JY901 IMU on `/dev/ttyAMA0`, 8BitDo gamepad via evdev.

```python
from bipedal_robot import BipedalRobotController
from IMU_JY901 import JY901IMU
from RL_agent_isolated import RLAgent
from gamepad_controller import GamepadController

# 1. IMU (optional but recommended)
imu = JY901IMU(port="/dev/ttyAMA0", baudrate=9600)
imu.start()

# 2. Robot — read-only first, never auto-enable on startup
robot = BipedalRobotController(control_hz=100.0, imu=imu)
robot.set_max_command_delta(60.0)
robot.start(mode="state_only", auto_enable=False)

# 3. Command zero pose, then enable
robot.set_action(
    left  = {k: 0.0 for k in ["hipz","hipx","hipy","knee","ankle_pitch","ankle_roll"]},
    right = {k: 0.0 for k in ["hipz","hipx","hipy","knee","ankle_pitch","ankle_roll"]},
)
robot.set_mode("control")
robot.enable_all()

# 4. Per-joint gains — copy from RL_policy/<policy>/gain.md
#    Example values for less_noice_high_gain_torque_obs:
for mid in [1, 7]:       robot.set_joint_gains(mid, kp=30, kd=3.0)   # hipz
for mid in [5, 6, 11, 12]: robot.set_joint_gains(mid, kp=10, kd=0.75) # ankles
robot.set_joint_gains(2, kp=40, kd=3.0);  robot.set_joint_gains(8, kp=40, kd=3.0)   # hipx
robot.set_joint_gains(3, kp=6,  kd=0.2);  robot.set_joint_gains(9, kp=6,  kd=0.2)   # hipy
robot.set_joint_gains(4, kp=6,  kd=0.2);  robot.set_joint_gains(10, kp=6, kd=0.2)   # knee

# 5. Gamepad + agent
pad = GamepadController(name_substring="8bitdo")
pad.connect(); pad.start()

agent = RLAgent.from_files(
    robot,
    config_path="RL_policy/less_noice_high_gain_torque_obs/config.yaml",
    policy_path="RL_policy/less_noice_high_gain_torque_obs/policy.onnx",
)
agent.spec.action_scale = 0.0   # ALWAYS start at 0 on real hardware
agent.spec.joint_vel_source = "auto"
agent.set_command_source(pad)
agent.start()

# Once the robot looks stable and tracks the zero pose, ramp up interactively:
#   agent.spec.action_scale = 0.3
#   agent.spec.action_scale = 1.0
```

**Shutdown:**

```python
agent.stop(); robot.stop(); pad.stop(); imu.stop()
```

---

## Motor layout

| Bus  | IDs   | Side  | Joints (hipz, hipx, hipy, knee, ankle_a, ankle_b) |
|------|-------|-------|----------------------------------------------------|
| can0 | 1–6   | Left  | m1, m2, m3, m4, m5, m6                             |
| can1 | 7–12  | Right | m7, m8, m9, m10, m11, m12                          |

Ankle motors are mechanically coupled — `ankle_pitch`/`ankle_roll` derive from the sum/diff of ankle_a and ankle_b.

Policy action order: `[right(6), left(6)]`, each `[hipz, hipx, hipy, knee, ankle_pitch, ankle_roll]`.

---

## Environment

See `environement.lock`. Key deps: `mujoco`, `onnxruntime`, `python-can`, `pinocchio`, `evdev`, `pyserial`, `numpy`, `PyYAML`.

For motor control on real hardware you also need the kernel CAN modules loaded and `can0` / `can1` interfaces up.

---

## Notes

- `bipdeal_config.py` has log output pasted into the source (around line 406) that prevents `import` of it. It's only used as a standalone calibration script, so this is harmless in practice — pre-existing on `main`.
- Only the JY901 IMU driver is included in this branch. BNO055/BNO085 support lived in the removed `IMU_integration.py` — pull it back from `main` if needed.
