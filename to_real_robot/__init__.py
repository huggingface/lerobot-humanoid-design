"""
to_real_robot — bipedal humanoid robot control stack.

Primary classes
---------------
    BipedalRobotController   real hardware (dual-CAN, MIT protocol)
    SimBipedalRobotController  MuJoCo simulation, identical API
    make_mock_robot          factory: MockBus + MockIMU, no hardware
    RLAgent                  RL policy inference loop
    AgentSpec                agent configuration dataclass

Quick start (IPython, cwd = to_real_robot/)
-------------------------------------------
    # Simulation + RL
    from sim_robot import SimBipedalRobotController
    from rl_agent import RLAgent
    robot = SimBipedalRobotController(control_hz=200.0)
    robot.start(mode="control"); robot.start_viewer()
    agent = RLAgent.from_files(robot,
        config_path="policies/less_noise_high_gain_torque_obs/config.yaml",
        policy_path="policies/less_noise_high_gain_torque_obs/policy.onnx")
    agent.start()

    # Hardware-free testing
    from mock_robot import make_mock_robot
    robot = make_mock_robot()
    robot.start(mode="state_only")
    print(robot.get_combined_state_snapshot()["joint_state_rad"])
"""
from bipedal_robot import BipedalRobotController  # noqa: F401
from sim_robot import SimBipedalRobotController   # noqa: F401
from mock_robot import make_mock_robot            # noqa: F401
from mock_bus import MockBus                      # noqa: F401
from rl_agent import RLAgent, AgentSpec           # noqa: F401
from imu import IMU, MockIMU                      # noqa: F401
from constants import (                           # noqa: F401
    MOTORS, MOTOR_IDS, DEFAULT_PD_GAINS,
    MOTOR_RAW_LIMITS_DEG, EMERGENCY_DAMPING_KD,
    CAN_CMD_REQUEST_STATE, CAN_CMD_ENABLE, CAN_CMD_DISABLE,
)
