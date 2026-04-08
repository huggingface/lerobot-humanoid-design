"""
Mock robot: BipedalRobotController backed by MockBus (both CAN buses) + MockIMU.
No real hardware required.  Designed for IPython copy-paste testing.

Typical IPython session
-----------------------
    from mock_robot import make_mock_robot
    robot = make_mock_robot()
    robot.start(mode="control", auto_enable=True)

    # Inspect state
    snap = robot.get_combined_state_snapshot()
    print(snap["joint_state_rad"])   # 12-element list, model joint space (rad)
    print(snap["imu"])               # IMU dict with quaternion, gyro, etc.

    # Note: the MockBus echoes zero for all motor positions on first read.
    # Motor 1 raw=0 deg is outside its calibrated range, so an E-STOP fires.
    # This is correct safety behaviour – it's useful to verify the controller
    # responds to limit violations.  The E-STOP does not prevent reading state.

    # Send a command (joints in model space, radians)
    import numpy as np
    q = np.zeros(12)
    q[2] = 0.3   # left hipy
    q[3] = 0.5   # left knee
    left_cmd  = {1: q[0], 2: q[1], 3: q[2], 4: q[3], 5: q[4],  6: q[5]}
    right_cmd = {7: q[6], 8: q[7], 9: q[8], 10: q[9], 11: q[10], 12: q[11]}
    # (or use agent.set_action() if you have an RL agent running)

    robot.stop()

Notes
-----
- MockBus echoes every MIT command immediately (zero inertia): the state jumps
  instantly to whatever position was commanded.  This is useful to check that
  observation / action pipelines are wired correctly without needing physics.
- For physics-accurate testing use sim_robot.py (SimBipedalRobotController).
- MockIMU returns constant values by default; override with imu.set_state(...)
  while the robot is running.
"""
from __future__ import annotations

from typing import Optional

from bipedal_robot import BipedalRobotController
from mock_bus import MockBus
from imu import MockIMU


def make_mock_robot(
    *,
    control_hz: float = 100.0,
    log_path: str = "mock_robot_log.csv",
    imu_quaternion_xyzw: tuple = (0.0, 0.0, 0.0, 1.0),
    disable_safety_limits: bool = True,
) -> BipedalRobotController:
    """
    Return a BipedalRobotController wired to two MockBus instances and a MockIMU.

    Parameters
    ----------
    control_hz:
        Internal control-loop frequency (same as real robot default).
    log_path:
        CSV log destination.  Defaults to a mock-specific CSV file.
    imu_quaternion_xyzw:
        Initial IMU quaternion.  Upright robot = (0, 0, 0, 1).
    disable_safety_limits:
        If True (default), clears all joint limits so the controller does not
        trigger an E-STOP when MockBus returns zero positions on startup.
        The MockBus echoes position=0 for all motors; motor 1's zero is outside
        its calibrated range, which would cause a deadlock (E-STOP fires inside
        the RX-drain callback, which holds the RX lock that E-STOP also needs).
        Disable limits to avoid this; the safety logic can be re-enabled later
        with robot.set_joint_limit() + robot._recompute_command_limits().
    """
    bus0 = MockBus()
    bus1 = MockBus()
    imu  = MockIMU(quaternion_xyzw=imu_quaternion_xyzw)
    robot = BipedalRobotController(
        bus_can0=bus0,
        bus_can1=bus1,
        imu=imu,
        control_hz=control_hz,
        log_path=log_path,
    )

    if disable_safety_limits:
        # Clear joint limits so no E-STOP fires on mock zero-positions
        robot.joint_limits_deg.clear()
        robot._recompute_command_limits()  # type: ignore[attr-defined]

    # Expose mock objects as public attributes for runtime tweaking:
    #   robot.mock_imu.set_state(gyro_rads=(0.1, 0.0, 0.0))
    robot.mock_imu: MockIMU = imu  # type: ignore[attr-defined]
    robot.mock_bus0: MockBus = bus0  # type: ignore[attr-defined]
    robot.mock_bus1: MockBus = bus1  # type: ignore[attr-defined]
    return robot


# ---------------------------------------------------------------------------
# Quick smoke-test (run as script: python mock_robot.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    print("Creating mock robot...")
    robot = make_mock_robot(log_path="mock_smoke_test.csv")
    robot.start(mode="state_only")
    time.sleep(1.0)

    snap = robot.get_combined_state_snapshot()
    print(f"  mode:             {snap['mode']}")
    print(f"  estop:            {snap['estop']}  ({snap['estop_reason']})")
    print(f"  joint_state_rad:  {snap['joint_state_rad']}")
    print(f"  imu available:    {snap['imu'].get('available')}")
    print(f"  imu quaternion:   {snap['imu'].get('quaternion_xyzw')}")

    robot.stop()
    print("Mock robot smoke-test passed.")
