# LeRobot Humanoid — Minimal

Minimal branch focused on the bipedal platform: sim-to-real RL policy deployment.

The co-design studies, OCP experiments, retired URDF variants, and calibration/debug tooling have been removed here. See `main` for the full design repo.

## What's in here

- `to_real_robot/` — controller, sim + real hardware interface
  - `sim_robot.py` — Mujoco simulation wrapper
  - `bipedal_robot.py` — real-robot controller (CAN + Robstride motors)
  - `bipdeal_config.py` — motor calibration / config
  - `robstride_toolkit.py` — motor protocol
  - `RL_agent_isolated.py` — policy runner (ONNX)
  - `gamepad_controller.py` — teleop
  - `IMU_JY901.py` — IMU driver
  - `root_constant.py` — motor / path constants
  - `leg_test/mit.py` — MIT-mode CAN motor encoding (shared dep)
  - `bipedal_plateform_no_arms/` — MJCF + URDF + meshes
  - `RL_policy/` — trained ONNX policies

## Requirements

See `environement.lock` for the Python environment.

## Known issues

- `to_real_robot/bipdeal_config.py` has log output pasted into the source (lines ~406+) that prevents `import`. Pre-existing on `main`; the file is used as a script, not imported.
